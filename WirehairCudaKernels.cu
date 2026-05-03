#include "WirehairCudaKernels.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>

namespace {

__global__ void WirehairEncodeAssistKernel(uint8_t* data, uint32_t bytes)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < bytes) {
        data[index] ^= 0;
    }
}

__global__ void WirehairDecodeChecksumKernel(const uint8_t* data, uint32_t bytes, uint32_t* checksumOut)
{
    __shared__ uint32_t scratch[256];
    const uint32_t tid = threadIdx.x;
    const uint32_t index = blockIdx.x * blockDim.x + tid;
    uint32_t value = 0;
    if (index < bytes) {
        value = static_cast<uint32_t>(data[index]);
    }
    scratch[tid] = value;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) {
            scratch[tid] ^= scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicXor(checksumOut, scratch[0]);
    }
}

bool CheckCuda(cudaError_t status)
{
    return status == cudaSuccess;
}

static uint64_t ClockUs()
{
    const auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
}

struct SessionStats
{
    std::atomic<uint64_t> EncodeCalls{0};
    std::atomic<uint64_t> DecodeCalls{0};
    std::atomic<uint64_t> SetupUs{0};
    std::atomic<uint64_t> H2dUs{0};
    std::atomic<uint64_t> KernelUs{0};
    std::atomic<uint64_t> D2hUs{0};
    std::atomic<uint64_t> SyncUs{0};
    std::atomic<uint64_t> BytesH2d{0};
    std::atomic<uint64_t> BytesD2h{0};
};

struct CudaSession
{
    std::mutex Mutex;
    bool Initialized = false;
    int DeviceOrdinal = -1;
    uint32_t RequestedStreamCount = 1;
    bool UsePinnedMemory = true;
    cudaStream_t Stream = nullptr;
    uint8_t* DeviceData = nullptr;
    size_t DeviceDataBytes = 0;
    uint32_t* DeviceChecksum = nullptr;
    uint8_t* PinnedHost = nullptr;
    size_t PinnedHostBytes = 0;
};

SessionStats g_stats;
CudaSession g_session;

bool EnsureSessionLocked(int selectedDevice)
{
    const uint64_t setupStart = ClockUs();
    if (!CheckCuda(cudaSetDevice(selectedDevice))) {
        return false;
    }
    if (!g_session.Initialized)
    {
        if (!CheckCuda(cudaStreamCreateWithFlags(&g_session.Stream, cudaStreamNonBlocking))) {
            g_session.Stream = nullptr;
            return false;
        }
        g_session.Initialized = true;
    }
    g_session.DeviceOrdinal = selectedDevice;
    g_stats.SetupUs.fetch_add(ClockUs() - setupStart, std::memory_order_relaxed);
    return true;
}

bool EnsureDeviceBufferLocked(size_t bytes)
{
    if (bytes <= g_session.DeviceDataBytes) {
        return true;
    }
    if (g_session.DeviceData) {
        cudaFree(g_session.DeviceData);
        g_session.DeviceData = nullptr;
        g_session.DeviceDataBytes = 0;
    }
    if (!CheckCuda(cudaMalloc(&g_session.DeviceData, bytes))) {
        return false;
    }
    g_session.DeviceDataBytes = bytes;
    return true;
}

bool EnsurePinnedHostLocked(size_t bytes)
{
    if (!g_session.UsePinnedMemory) {
        return true;
    }
    if (bytes <= g_session.PinnedHostBytes) {
        return true;
    }
    if (g_session.PinnedHost) {
        cudaFreeHost(g_session.PinnedHost);
        g_session.PinnedHost = nullptr;
        g_session.PinnedHostBytes = 0;
    }
    if (!CheckCuda(cudaHostAlloc(&g_session.PinnedHost, bytes, cudaHostAllocPortable))) {
        return false;
    }
    g_session.PinnedHostBytes = bytes;
    return true;
}

bool EnsureChecksumBufferLocked()
{
    if (g_session.DeviceChecksum) {
        return true;
    }
    return CheckCuda(cudaMalloc(&g_session.DeviceChecksum, sizeof(uint32_t)));
}

} // namespace

bool WirehairCudaKernelProbe(uint32_t* deviceCountOut)
{
    int count = 0;
    if (!CheckCuda(cudaGetDeviceCount(&count)) || count <= 0) {
        if (deviceCountOut) {
            *deviceCountOut = 0;
        }
        return false;
    }
    if (deviceCountOut) {
        *deviceCountOut = static_cast<uint32_t>(count);
    }
    return true;
}

bool WirehairCudaKernelEnableDevice(int32_t deviceOrdinal)
{
    int count = 0;
    if (!CheckCuda(cudaGetDeviceCount(&count)) || count <= 0) {
        return false;
    }
    int selected = 0;
    if (deviceOrdinal >= 0 && deviceOrdinal < count) {
        selected = deviceOrdinal;
    }
    std::lock_guard<std::mutex> lock(g_session.Mutex);
    return EnsureSessionLocked(selected);
}

bool WirehairCudaKernelConfigure(uint32_t streamCount, uint32_t usePinnedMemory)
{
    std::lock_guard<std::mutex> lock(g_session.Mutex);
    g_session.RequestedStreamCount = std::max<uint32_t>(1, streamCount);
    const bool wantPinned = usePinnedMemory != 0;
    if (g_session.UsePinnedMemory != wantPinned)
    {
        g_session.UsePinnedMemory = wantPinned;
        if (!wantPinned && g_session.PinnedHost) {
            cudaFreeHost(g_session.PinnedHost);
            g_session.PinnedHost = nullptr;
            g_session.PinnedHostBytes = 0;
        }
    }
    (void)g_session.RequestedStreamCount;
    return true;
}

bool WirehairCudaKernelEncodeAssist(void* data, uint32_t bytes)
{
    if (!data || bytes == 0) {
        return false;
    }
    g_stats.EncodeCalls.fetch_add(1, std::memory_order_relaxed);
    const uint64_t bytes64 = bytes;
    const uint8_t* hostIn = reinterpret_cast<const uint8_t*>(data);
    uint8_t* hostOut = reinterpret_cast<uint8_t*>(data);
    const uint32_t threadsPerBlock = 256;
    const uint32_t blockCount = (bytes + threadsPerBlock - 1) / threadsPerBlock;

    std::lock_guard<std::mutex> lock(g_session.Mutex);
    int device = g_session.DeviceOrdinal;
    if (device < 0) {
        device = 0;
    }
    if (!EnsureSessionLocked(device) || !EnsureDeviceBufferLocked(bytes) || !EnsurePinnedHostLocked(bytes)) {
        return false;
    }

    const uint64_t h2dStart = ClockUs();
    const void* source = hostIn;
    if (g_session.UsePinnedMemory && g_session.PinnedHost) {
        std::memcpy(g_session.PinnedHost, hostIn, bytes);
        source = g_session.PinnedHost;
    }
    if (!CheckCuda(cudaMemcpyAsync(g_session.DeviceData, source, bytes, cudaMemcpyHostToDevice, g_session.Stream))) {
        return false;
    }
    g_stats.H2dUs.fetch_add(ClockUs() - h2dStart, std::memory_order_relaxed);
    g_stats.BytesH2d.fetch_add(bytes64, std::memory_order_relaxed);

    const uint64_t kernelStart = ClockUs();
    WirehairEncodeAssistKernel<<<blockCount, threadsPerBlock, 0, g_session.Stream>>>(g_session.DeviceData, bytes);
    if (!CheckCuda(cudaGetLastError())) {
        return false;
    }
    g_stats.KernelUs.fetch_add(ClockUs() - kernelStart, std::memory_order_relaxed);

    const uint64_t d2hStart = ClockUs();
    void* dest = hostOut;
    if (g_session.UsePinnedMemory && g_session.PinnedHost) {
        dest = g_session.PinnedHost;
    }
    if (!CheckCuda(cudaMemcpyAsync(dest, g_session.DeviceData, bytes, cudaMemcpyDeviceToHost, g_session.Stream))) {
        return false;
    }
    g_stats.D2hUs.fetch_add(ClockUs() - d2hStart, std::memory_order_relaxed);
    g_stats.BytesD2h.fetch_add(bytes64, std::memory_order_relaxed);

    const uint64_t syncStart = ClockUs();
    if (!CheckCuda(cudaStreamSynchronize(g_session.Stream))) {
        return false;
    }
    g_stats.SyncUs.fetch_add(ClockUs() - syncStart, std::memory_order_relaxed);
    if (g_session.UsePinnedMemory && g_session.PinnedHost) {
        std::memcpy(hostOut, g_session.PinnedHost, bytes);
    }
    return true;
}

bool WirehairCudaKernelDecodeAssist(const void* data, uint32_t bytes, uint32_t* checksumOut)
{
    if (!data || !checksumOut || bytes == 0) {
        return false;
    }

    g_stats.DecodeCalls.fetch_add(1, std::memory_order_relaxed);
    const uint64_t bytes64 = bytes;
    const uint8_t* hostIn = reinterpret_cast<const uint8_t*>(data);
    const uint32_t threadsPerBlock = 256;
    const uint32_t blockCount = (bytes + threadsPerBlock - 1) / threadsPerBlock;
    std::lock_guard<std::mutex> lock(g_session.Mutex);
    int device = g_session.DeviceOrdinal;
    if (device < 0) {
        device = 0;
    }
    if (!EnsureSessionLocked(device) || !EnsureDeviceBufferLocked(bytes) || !EnsureChecksumBufferLocked() ||
        !EnsurePinnedHostLocked(bytes))
    {
        return false;
    }

    const uint64_t h2dStart = ClockUs();
    const void* source = hostIn;
    if (g_session.UsePinnedMemory && g_session.PinnedHost) {
        std::memcpy(g_session.PinnedHost, hostIn, bytes);
        source = g_session.PinnedHost;
    }
    uint32_t checksumSeed = 0;
    const bool copiedIn = CheckCuda(cudaMemcpyAsync(g_session.DeviceData, source, bytes, cudaMemcpyHostToDevice, g_session.Stream)) &&
        CheckCuda(cudaMemcpyAsync(g_session.DeviceChecksum, &checksumSeed, sizeof(uint32_t), cudaMemcpyHostToDevice, g_session.Stream));
    if (!copiedIn) {
        return false;
    }
    g_stats.H2dUs.fetch_add(ClockUs() - h2dStart, std::memory_order_relaxed);
    g_stats.BytesH2d.fetch_add(bytes64 + sizeof(uint32_t), std::memory_order_relaxed);

    const uint64_t kernelStart = ClockUs();
    WirehairDecodeChecksumKernel<<<blockCount, threadsPerBlock, 0, g_session.Stream>>>(g_session.DeviceData, bytes, g_session.DeviceChecksum);
    if (!CheckCuda(cudaGetLastError())) {
        return false;
    }
    g_stats.KernelUs.fetch_add(ClockUs() - kernelStart, std::memory_order_relaxed);

    const uint64_t d2hStart = ClockUs();
    if (!CheckCuda(cudaMemcpyAsync(checksumOut, g_session.DeviceChecksum, sizeof(uint32_t), cudaMemcpyDeviceToHost, g_session.Stream))) {
        return false;
    }
    g_stats.D2hUs.fetch_add(ClockUs() - d2hStart, std::memory_order_relaxed);
    g_stats.BytesD2h.fetch_add(sizeof(uint32_t), std::memory_order_relaxed);

    const uint64_t syncStart = ClockUs();
    if (!CheckCuda(cudaStreamSynchronize(g_session.Stream))) {
        return false;
    }
    g_stats.SyncUs.fetch_add(ClockUs() - syncStart, std::memory_order_relaxed);
    return true;
}

void WirehairCudaKernelResetStats()
{
    g_stats.EncodeCalls.store(0, std::memory_order_relaxed);
    g_stats.DecodeCalls.store(0, std::memory_order_relaxed);
    g_stats.SetupUs.store(0, std::memory_order_relaxed);
    g_stats.H2dUs.store(0, std::memory_order_relaxed);
    g_stats.KernelUs.store(0, std::memory_order_relaxed);
    g_stats.D2hUs.store(0, std::memory_order_relaxed);
    g_stats.SyncUs.store(0, std::memory_order_relaxed);
    g_stats.BytesH2d.store(0, std::memory_order_relaxed);
    g_stats.BytesD2h.store(0, std::memory_order_relaxed);
}

bool WirehairCudaKernelGetStats(WirehairCudaKernelStats* statsOut)
{
    if (!statsOut) {
        return false;
    }
    statsOut->encode_calls = g_stats.EncodeCalls.load(std::memory_order_relaxed);
    statsOut->decode_calls = g_stats.DecodeCalls.load(std::memory_order_relaxed);
    statsOut->setup_us = g_stats.SetupUs.load(std::memory_order_relaxed);
    statsOut->h2d_us = g_stats.H2dUs.load(std::memory_order_relaxed);
    statsOut->kernel_us = g_stats.KernelUs.load(std::memory_order_relaxed);
    statsOut->d2h_us = g_stats.D2hUs.load(std::memory_order_relaxed);
    statsOut->sync_us = g_stats.SyncUs.load(std::memory_order_relaxed);
    statsOut->bytes_h2d = g_stats.BytesH2d.load(std::memory_order_relaxed);
    statsOut->bytes_d2h = g_stats.BytesD2h.load(std::memory_order_relaxed);
    return true;
}
