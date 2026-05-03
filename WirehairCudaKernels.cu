#include "WirehairCudaKernels.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace {

__device__ __forceinline__ uint32_t FoldUint4Xor(const uint4& value)
{
    return value.x ^ value.y ^ value.z ^ value.w;
}

__global__ void WirehairBatchEncodeParityKernel(
    const uint8_t* input,
    uint8_t* output,
    uint32_t itemStrideBytes,
    uint32_t itemBytes,
    uint32_t* checksums)
{
    const uint32_t itemIndex = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t blockXor[256];
    uint32_t localXor = 0;

    const uint8_t* itemIn = input + static_cast<size_t>(itemIndex) * itemStrideBytes;
    uint8_t* itemOut = output + static_cast<size_t>(itemIndex) * itemStrideBytes;
    const uintptr_t inAddr = reinterpret_cast<uintptr_t>(itemIn);
    const uintptr_t outAddr = reinterpret_cast<uintptr_t>(itemOut);
    const bool canVectorize = ((inAddr & 0xF) == 0) && ((outAddr & 0xF) == 0);
    const uint32_t vecBytes = canVectorize ? (itemBytes & ~15U) : 0U;

    for (uint32_t offset = tid * 16U; offset < vecBytes; offset += blockDim.x * 16U)
    {
        const uint4 value = *reinterpret_cast<const uint4*>(itemIn + offset);
        *reinterpret_cast<uint4*>(itemOut + offset) = value;
        localXor ^= FoldUint4Xor(value);
    }
    for (uint32_t offset = vecBytes + tid; offset < itemBytes; offset += blockDim.x)
    {
        const uint8_t value = itemIn[offset];
        itemOut[offset] = value;
        localXor ^= static_cast<uint32_t>(value) << ((offset & 3U) * 8U);
    }

    blockXor[tid] = localXor;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) {
            blockXor[tid] ^= blockXor[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        checksums[itemIndex] = blockXor[0];
    }
}

__global__ void WirehairBatchDecodeSolveKernel(
    const uint8_t* input,
    uint32_t itemStrideBytes,
    uint32_t itemBytes,
    uint32_t* checksums)
{
    const uint32_t itemIndex = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t blockXor[256];
    uint32_t localXor = 0;

    const uint8_t* itemIn = input + static_cast<size_t>(itemIndex) * itemStrideBytes;
    const uintptr_t inAddr = reinterpret_cast<uintptr_t>(itemIn);
    const bool canVectorize = ((inAddr & 0xF) == 0);
    const uint32_t vecBytes = canVectorize ? (itemBytes & ~15U) : 0U;

    for (uint32_t offset = tid * 16U; offset < vecBytes; offset += blockDim.x * 16U)
    {
        const uint4 value = *reinterpret_cast<const uint4*>(itemIn + offset);
        localXor ^= FoldUint4Xor(value);
    }
    for (uint32_t offset = vecBytes + tid; offset < itemBytes; offset += blockDim.x)
    {
        const uint8_t value = itemIn[offset];
        localXor ^= static_cast<uint32_t>(value) << ((offset & 3U) * 8U);
    }

    blockXor[tid] = localXor;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) {
            blockXor[tid] ^= blockXor[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        checksums[itemIndex] = blockXor[0];
    }
}

__global__ void WirehairXorInPlaceKernel(
    uint8_t* dest,
    const uint8_t* src,
    uint32_t bytes)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < bytes; i += stride) {
        dest[i] ^= src[i];
    }
}

bool CheckCuda(cudaError_t status)
{
    return status == cudaSuccess;
}

uint64_t EventElapsedUs(cudaEvent_t start, cudaEvent_t stop)
{
    float elapsedMs = 0.0f;
    if (!CheckCuda(cudaEventElapsedTime(&elapsedMs, start, stop))) {
        return 0;
    }
    return static_cast<uint64_t>(elapsedMs * 1000.0f);
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
    std::atomic<uint64_t> H2dEventUs{0};
    std::atomic<uint64_t> KernelEventUs{0};
    std::atomic<uint64_t> D2hEventUs{0};
    std::atomic<uint64_t> E2eEventUs{0};
    std::atomic<uint64_t> BytesH2d{0};
    std::atomic<uint64_t> BytesD2h{0};
};

struct CudaSession
{
    std::mutex Mutex;
    bool Initialized = false;
    int DeviceOrdinal = -1;
    uint32_t RequestedStreamCount = 1;
    uint32_t ActiveStreamCount = 1;
    bool UsePinnedMemory = true;
};

struct StreamContext
{
    std::mutex Mutex;
    cudaStream_t Stream = nullptr;
    uint8_t* DeviceData = nullptr;
    size_t DeviceDataBytes = 0;
    uint8_t* DeviceAux = nullptr;
    size_t DeviceAuxBytes = 0;
    uint32_t* DeviceChecksums = nullptr;
    size_t DeviceChecksumsCount = 0;
    uint8_t* PinnedHost = nullptr;
    size_t PinnedHostBytes = 0;
    cudaEvent_t CompletionEvent = nullptr;
    cudaEvent_t H2dStartEvent = nullptr;
    cudaEvent_t H2dStopEvent = nullptr;
    cudaEvent_t KernelStartEvent = nullptr;
    cudaEvent_t KernelStopEvent = nullptr;
    cudaEvent_t D2hStartEvent = nullptr;
    cudaEvent_t D2hStopEvent = nullptr;
    cudaEvent_t E2eStartEvent = nullptr;
    cudaEvent_t E2eStopEvent = nullptr;
    bool EventInitialized = false;
    bool TimingEventsInitialized = false;
    bool Initialized = false;
};

SessionStats g_stats;
CudaSession g_session;
static const uint32_t kMaxStreamPoolSize = 16;
StreamContext g_streams[kMaxStreamPoolSize];
std::atomic<uint32_t> g_nextStreamIndex(0);

bool EnsureSessionLocked(int selectedDevice)
{
    const uint64_t setupStart = ClockUs();
    if (!CheckCuda(cudaSetDevice(selectedDevice))) {
        return false;
    }
    g_session.Initialized = true;
    g_session.DeviceOrdinal = selectedDevice;
    g_stats.SetupUs.fetch_add(ClockUs() - setupStart, std::memory_order_relaxed);
    return true;
}

bool EnsureStreamContextInitialized(StreamContext& ctx)
{
    if (ctx.Initialized) {
        return true;
    }
    if (!CheckCuda(cudaStreamCreateWithFlags(&ctx.Stream, cudaStreamNonBlocking))) {
        ctx.Stream = nullptr;
        return false;
    }
    ctx.Initialized = true;
    return true;
}

bool EnsureStreamPoolLocked()
{
    const uint32_t targetCount = std::max<uint32_t>(1, std::min<uint32_t>(g_session.RequestedStreamCount, kMaxStreamPoolSize));
    for (uint32_t i = 0; i < targetCount; ++i)
    {
        StreamContext& ctx = g_streams[i];
        std::lock_guard<std::mutex> streamGuard(ctx.Mutex);
        if (!EnsureStreamContextInitialized(ctx)) {
            return false;
        }
    }
    g_session.ActiveStreamCount = targetCount;
    return true;
}

uint32_t AcquireStreamIndexLocked()
{
    const uint32_t active = std::max<uint32_t>(1, g_session.ActiveStreamCount);
    const uint32_t next = g_nextStreamIndex.fetch_add(1, std::memory_order_relaxed);
    return next % active;
}

bool EnsureDeviceBufferLocked(StreamContext& ctx, size_t bytes)
{
    if (bytes <= ctx.DeviceDataBytes) {
        return true;
    }
    if (ctx.DeviceData) {
        cudaFree(ctx.DeviceData);
        ctx.DeviceData = nullptr;
        ctx.DeviceDataBytes = 0;
    }
    if (!CheckCuda(cudaMalloc(&ctx.DeviceData, bytes))) {
        return false;
    }
    ctx.DeviceDataBytes = bytes;
    return true;
}

bool EnsureAuxBufferLocked(StreamContext& ctx, size_t bytes)
{
    if (bytes <= ctx.DeviceAuxBytes) {
        return true;
    }
    if (ctx.DeviceAux) {
        cudaFree(ctx.DeviceAux);
        ctx.DeviceAux = nullptr;
        ctx.DeviceAuxBytes = 0;
    }
    if (!CheckCuda(cudaMalloc(&ctx.DeviceAux, bytes))) {
        return false;
    }
    ctx.DeviceAuxBytes = bytes;
    return true;
}

bool EnsurePinnedHostLocked(StreamContext& ctx, bool usePinnedMemory, size_t bytes)
{
    if (!usePinnedMemory) {
        return true;
    }
    if (bytes <= ctx.PinnedHostBytes) {
        return true;
    }
    if (ctx.PinnedHost) {
        cudaFreeHost(ctx.PinnedHost);
        ctx.PinnedHost = nullptr;
        ctx.PinnedHostBytes = 0;
    }
    if (!CheckCuda(cudaHostAlloc(&ctx.PinnedHost, bytes, cudaHostAllocPortable))) {
        return false;
    }
    ctx.PinnedHostBytes = bytes;
    return true;
}

bool EnsureChecksumBufferLocked(StreamContext& ctx, size_t checksumCount)
{
    if (checksumCount <= ctx.DeviceChecksumsCount && ctx.DeviceChecksums) {
        return true;
    }
    if (ctx.DeviceChecksums) {
        cudaFree(ctx.DeviceChecksums);
        ctx.DeviceChecksums = nullptr;
        ctx.DeviceChecksumsCount = 0;
    }
    if (!CheckCuda(cudaMalloc(&ctx.DeviceChecksums, checksumCount * sizeof(uint32_t)))) {
        return false;
    }
    ctx.DeviceChecksumsCount = checksumCount;
    return true;
}

bool EnsureCompletionEventLocked(StreamContext& ctx)
{
    if (ctx.EventInitialized && ctx.CompletionEvent) {
        return true;
    }
    if (!CheckCuda(cudaEventCreateWithFlags(&ctx.CompletionEvent, cudaEventDisableTiming))) {
        ctx.CompletionEvent = nullptr;
        ctx.EventInitialized = false;
        return false;
    }
    ctx.EventInitialized = true;
    return true;
}

bool EnsureTimingEventsLocked(StreamContext& ctx)
{
    if (ctx.TimingEventsInitialized) {
        return true;
    }
    if (!CheckCuda(cudaEventCreate(&ctx.H2dStartEvent)) ||
        !CheckCuda(cudaEventCreate(&ctx.H2dStopEvent)) ||
        !CheckCuda(cudaEventCreate(&ctx.KernelStartEvent)) ||
        !CheckCuda(cudaEventCreate(&ctx.KernelStopEvent)) ||
        !CheckCuda(cudaEventCreate(&ctx.D2hStartEvent)) ||
        !CheckCuda(cudaEventCreate(&ctx.D2hStopEvent)) ||
        !CheckCuda(cudaEventCreate(&ctx.E2eStartEvent)) ||
        !CheckCuda(cudaEventCreate(&ctx.E2eStopEvent)))
    {
        return false;
    }
    ctx.TimingEventsInitialized = true;
    return true;
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
    g_session.RequestedStreamCount = std::max<uint32_t>(1, std::min<uint32_t>(streamCount, kMaxStreamPoolSize));
    const bool wantPinned = usePinnedMemory != 0;
    if (g_session.UsePinnedMemory != wantPinned)
    {
        g_session.UsePinnedMemory = wantPinned;
        if (!wantPinned)
        {
            for (uint32_t i = 0; i < kMaxStreamPoolSize; ++i)
            {
                StreamContext& ctx = g_streams[i];
                std::lock_guard<std::mutex> streamGuard(ctx.Mutex);
                if (ctx.PinnedHost) {
                    cudaFreeHost(ctx.PinnedHost);
                    ctx.PinnedHost = nullptr;
                    ctx.PinnedHostBytes = 0;
                }
            }
        }
    }
    return EnsureStreamPoolLocked();
}

bool WirehairCudaKernelEncodeAssist(void* data, uint32_t bytes)
{
    WirehairCudaCoreBatchParams params = {};
    params.struct_bytes = sizeof(WirehairCudaCoreBatchParams);
    params.op = WirehairCudaCoreOp_EncodeParity;
    params.item_count = 1;
    params.item_stride_bytes = bytes;
    params.item_bytes = bytes;
    params.options = WirehairCudaCoreBatchOption_AllowHostRegister;

    WirehairCudaCoreBatchResult result = {};
    result.struct_bytes = sizeof(WirehairCudaCoreBatchResult);
    return WirehairCudaKernelProcessBatch(&params, data, data, &result);
}

bool WirehairCudaKernelDecodeAssist(const void* data, uint32_t bytes, uint32_t* checksumOut)
{
    if (!checksumOut) {
        return false;
    }
    WirehairCudaCoreBatchParams params = {};
    params.struct_bytes = sizeof(WirehairCudaCoreBatchParams);
    params.op = WirehairCudaCoreOp_DecodeSolve;
    params.item_count = 1;
    params.item_stride_bytes = bytes;
    params.item_bytes = bytes;
    params.options = WirehairCudaCoreBatchOption_AllowHostRegister;

    WirehairCudaCoreBatchResult result = {};
    result.struct_bytes = sizeof(WirehairCudaCoreBatchResult);
    if (!WirehairCudaKernelProcessBatch(&params, data, nullptr, &result)) {
        return false;
    }
    *checksumOut = result.checksum;
    return true;
}

bool WirehairCudaKernelXorInPlace(void* destData, const void* srcData, uint32_t bytes)
{
    if (!destData || !srcData || bytes == 0) {
        return false;
    }
    const uint32_t threadsPerBlock = 256;
    const uint32_t blockCount = std::max<uint32_t>(1, (bytes + threadsPerBlock - 1) / threadsPerBlock);
    int device = 0;
    bool usePinnedMemory = true;
    uint32_t streamIndex = 0;
    {
        std::lock_guard<std::mutex> lock(g_session.Mutex);
        device = g_session.DeviceOrdinal >= 0 ? g_session.DeviceOrdinal : 0;
        usePinnedMemory = g_session.UsePinnedMemory;
        if (!EnsureSessionLocked(device) || !EnsureStreamPoolLocked()) {
            return false;
        }
        streamIndex = AcquireStreamIndexLocked();
    }
    StreamContext& ctx = g_streams[streamIndex];
    std::lock_guard<std::mutex> streamLock(ctx.Mutex);
    if (!CheckCuda(cudaSetDevice(device)) ||
        !EnsureStreamContextInitialized(ctx) ||
        !EnsureDeviceBufferLocked(ctx, bytes) ||
        !EnsureAuxBufferLocked(ctx, bytes) ||
        !EnsurePinnedHostLocked(ctx, usePinnedMemory, bytes) ||
        !EnsureCompletionEventLocked(ctx))
    {
        return false;
    }

    const uint8_t* srcHost = reinterpret_cast<const uint8_t*>(srcData);
    uint8_t* dstHost = reinterpret_cast<uint8_t*>(destData);
    const uint64_t h2dStart = ClockUs();
    if (!CheckCuda(cudaMemcpyAsync(ctx.DeviceData, dstHost, bytes, cudaMemcpyHostToDevice, ctx.Stream)) ||
        !CheckCuda(cudaMemcpyAsync(ctx.DeviceAux, srcHost, bytes, cudaMemcpyHostToDevice, ctx.Stream)))
    {
        return false;
    }
    g_stats.H2dUs.fetch_add(ClockUs() - h2dStart, std::memory_order_relaxed);
    g_stats.BytesH2d.fetch_add(static_cast<uint64_t>(bytes) * 2ULL, std::memory_order_relaxed);

    const uint64_t kernelStart = ClockUs();
    WirehairXorInPlaceKernel<<<blockCount, threadsPerBlock, 0, ctx.Stream>>>(ctx.DeviceData, ctx.DeviceAux, bytes);
    if (!CheckCuda(cudaGetLastError())) {
        return false;
    }
    g_stats.KernelUs.fetch_add(ClockUs() - kernelStart, std::memory_order_relaxed);
    g_stats.EncodeCalls.fetch_add(1, std::memory_order_relaxed);

    const uint64_t d2hStart = ClockUs();
    if (!CheckCuda(cudaMemcpyAsync(dstHost, ctx.DeviceData, bytes, cudaMemcpyDeviceToHost, ctx.Stream))) {
        return false;
    }
    g_stats.D2hUs.fetch_add(ClockUs() - d2hStart, std::memory_order_relaxed);
    g_stats.BytesD2h.fetch_add(bytes, std::memory_order_relaxed);

    const uint64_t syncStart = ClockUs();
    if (!CheckCuda(cudaStreamSynchronize(ctx.Stream))) {
        return false;
    }
    g_stats.SyncUs.fetch_add(ClockUs() - syncStart, std::memory_order_relaxed);
    return true;
}

bool WirehairCudaKernelProcessBatch(
    const WirehairCudaCoreBatchParams* params,
    const void* inputData,
    void* outputData,
    WirehairCudaCoreBatchResult* resultOut)
{
    if (!params || !inputData || !resultOut) {
        return false;
    }
    if (params->struct_bytes != sizeof(WirehairCudaCoreBatchParams) ||
        resultOut->struct_bytes != sizeof(WirehairCudaCoreBatchResult) ||
        params->item_count == 0 ||
        params->item_bytes == 0 ||
        params->item_stride_bytes < params->item_bytes ||
        (params->op != WirehairCudaCoreOp_EncodeParity && params->op != WirehairCudaCoreOp_DecodeSolve))
    {
        return false;
    }

    const size_t totalBytes = static_cast<size_t>(params->item_count) * params->item_stride_bytes;
    const uint64_t totalBytes64 = static_cast<uint64_t>(totalBytes);
    const uint32_t threadsPerBlock = 256;
    const uint32_t blockCount = params->item_count;

    int device = 0;
    bool usePinnedMemory = true;
    uint32_t streamIndex = 0;
    {
        std::lock_guard<std::mutex> lock(g_session.Mutex);
        device = g_session.DeviceOrdinal >= 0 ? g_session.DeviceOrdinal : 0;
        usePinnedMemory = g_session.UsePinnedMemory;
        if (!EnsureSessionLocked(device) || !EnsureStreamPoolLocked()) {
            return false;
        }
        streamIndex = AcquireStreamIndexLocked();
    }

    StreamContext& ctx = g_streams[streamIndex];
    std::lock_guard<std::mutex> streamLock(ctx.Mutex);
    if (!CheckCuda(cudaSetDevice(device)) ||
        !EnsureStreamContextInitialized(ctx) ||
        !EnsureDeviceBufferLocked(ctx, totalBytes) ||
        !EnsureChecksumBufferLocked(ctx, params->item_count) ||
        !EnsurePinnedHostLocked(ctx, usePinnedMemory, totalBytes) ||
        !EnsureCompletionEventLocked(ctx) ||
        !EnsureTimingEventsLocked(ctx))
    {
        return false;
    }

    const uint8_t* hostIn = reinterpret_cast<const uint8_t*>(inputData);
    uint8_t* hostOut = reinterpret_cast<uint8_t*>(outputData);
    bool registeredIn = false;
    bool registeredOut = false;
    const bool allowHostRegister = (params->options & WirehairCudaCoreBatchOption_AllowHostRegister) != 0;
    if (usePinnedMemory && allowHostRegister && totalBytes >= (64U * 1024U))
    {
        if (CheckCuda(cudaHostRegister(const_cast<uint8_t*>(hostIn), totalBytes, cudaHostRegisterPortable))) {
            registeredIn = true;
        }
        if (hostOut && hostOut != hostIn &&
            CheckCuda(cudaHostRegister(hostOut, totalBytes, cudaHostRegisterPortable)))
        {
            registeredOut = true;
        }
    }
    struct HostRegistrationGuard
    {
        const uint8_t* In = nullptr;
        uint8_t* Out = nullptr;
        bool* InRegistered = nullptr;
        bool* OutRegistered = nullptr;
        ~HostRegistrationGuard()
        {
            if (InRegistered && *InRegistered && In) {
                cudaHostUnregister(const_cast<uint8_t*>(In));
            }
            if (OutRegistered && *OutRegistered && Out) {
                cudaHostUnregister(Out);
            }
        }
    } hostRegGuard = {hostIn, hostOut, &registeredIn, &registeredOut};

    const uint8_t* source = hostIn;
    if (usePinnedMemory && !registeredIn && ctx.PinnedHost) {
        std::memcpy(ctx.PinnedHost, hostIn, totalBytes);
        source = ctx.PinnedHost;
    }
    const bool skipOutputCopy = ((params->options & WirehairCudaCoreBatchOption_SkipOutputCopy) != 0) ||
        (params->op == WirehairCudaCoreOp_EncodeParity && outputData == inputData);
    const bool copyOutput = hostOut && !skipOutputCopy;

    if (!CheckCuda(cudaEventRecord(ctx.E2eStartEvent, ctx.Stream))) {
        return false;
    }

    const uint64_t h2dStart = ClockUs();
    if (!CheckCuda(cudaEventRecord(ctx.H2dStartEvent, ctx.Stream))) {
        return false;
    }
    if (!CheckCuda(cudaMemcpyAsync(ctx.DeviceData, source, totalBytes, cudaMemcpyHostToDevice, ctx.Stream)) ||
        !CheckCuda(cudaMemsetAsync(ctx.DeviceChecksums, 0, params->item_count * sizeof(uint32_t), ctx.Stream)))
    {
        return false;
    }
    if (!CheckCuda(cudaEventRecord(ctx.H2dStopEvent, ctx.Stream))) {
        return false;
    }
    g_stats.H2dUs.fetch_add(ClockUs() - h2dStart, std::memory_order_relaxed);
    g_stats.BytesH2d.fetch_add(totalBytes64, std::memory_order_relaxed);

    const uint64_t kernelStart = ClockUs();
    if (!CheckCuda(cudaEventRecord(ctx.KernelStartEvent, ctx.Stream))) {
        return false;
    }
    if (params->op == WirehairCudaCoreOp_EncodeParity) {
        WirehairBatchEncodeParityKernel<<<blockCount, threadsPerBlock, 0, ctx.Stream>>>(
            ctx.DeviceData,
            ctx.DeviceData,
            params->item_stride_bytes,
            params->item_bytes,
            ctx.DeviceChecksums);
        g_stats.EncodeCalls.fetch_add(1, std::memory_order_relaxed);
    } else {
        WirehairBatchDecodeSolveKernel<<<blockCount, threadsPerBlock, 0, ctx.Stream>>>(
            ctx.DeviceData,
            params->item_stride_bytes,
            params->item_bytes,
            ctx.DeviceChecksums);
        g_stats.DecodeCalls.fetch_add(1, std::memory_order_relaxed);
    }
    if (!CheckCuda(cudaGetLastError())) {
        return false;
    }
    if (!CheckCuda(cudaEventRecord(ctx.KernelStopEvent, ctx.Stream))) {
        return false;
    }
    g_stats.KernelUs.fetch_add(ClockUs() - kernelStart, std::memory_order_relaxed);

    uint32_t checksum = 0;
    const uint64_t d2hStart = ClockUs();
    if (!CheckCuda(cudaEventRecord(ctx.D2hStartEvent, ctx.Stream))) {
        return false;
    }
    if (!CheckCuda(cudaMemcpyAsync(&checksum, ctx.DeviceChecksums, sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx.Stream))) {
        return false;
    }
    uint64_t bytesD2h = sizeof(uint32_t);
    if (copyOutput) {
        void* dest = hostOut;
        if (usePinnedMemory && !registeredOut && ctx.PinnedHost) {
            dest = ctx.PinnedHost;
        }
        if (!CheckCuda(cudaMemcpyAsync(dest, ctx.DeviceData, totalBytes, cudaMemcpyDeviceToHost, ctx.Stream))) {
            return false;
        }
        bytesD2h += totalBytes64;
    }
    if (!CheckCuda(cudaEventRecord(ctx.D2hStopEvent, ctx.Stream)) ||
        !CheckCuda(cudaEventRecord(ctx.E2eStopEvent, ctx.Stream)))
    {
        return false;
    }
    g_stats.D2hUs.fetch_add(ClockUs() - d2hStart, std::memory_order_relaxed);
    g_stats.BytesD2h.fetch_add(bytesD2h, std::memory_order_relaxed);

    const uint64_t syncStart = ClockUs();
    if (!CheckCuda(cudaEventRecord(ctx.CompletionEvent, ctx.Stream)) ||
        !CheckCuda(cudaEventSynchronize(ctx.CompletionEvent)))
    {
        return false;
    }
    g_stats.SyncUs.fetch_add(ClockUs() - syncStart, std::memory_order_relaxed);
    g_stats.H2dEventUs.fetch_add(EventElapsedUs(ctx.H2dStartEvent, ctx.H2dStopEvent), std::memory_order_relaxed);
    g_stats.KernelEventUs.fetch_add(EventElapsedUs(ctx.KernelStartEvent, ctx.KernelStopEvent), std::memory_order_relaxed);
    g_stats.D2hEventUs.fetch_add(EventElapsedUs(ctx.D2hStartEvent, ctx.D2hStopEvent), std::memory_order_relaxed);
    g_stats.E2eEventUs.fetch_add(EventElapsedUs(ctx.E2eStartEvent, ctx.E2eStopEvent), std::memory_order_relaxed);
    if (copyOutput && usePinnedMemory && !registeredOut && ctx.PinnedHost) {
        std::memcpy(hostOut, ctx.PinnedHost, totalBytes);
    }
    std::memset(resultOut, 0, sizeof(WirehairCudaCoreBatchResult));
    resultOut->struct_bytes = sizeof(WirehairCudaCoreBatchResult);
    resultOut->processed_count = params->item_count;
    resultOut->checksum = checksum;
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
    g_stats.H2dEventUs.store(0, std::memory_order_relaxed);
    g_stats.KernelEventUs.store(0, std::memory_order_relaxed);
    g_stats.D2hEventUs.store(0, std::memory_order_relaxed);
    g_stats.E2eEventUs.store(0, std::memory_order_relaxed);
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
    statsOut->h2d_event_us = g_stats.H2dEventUs.load(std::memory_order_relaxed);
    statsOut->kernel_event_us = g_stats.KernelEventUs.load(std::memory_order_relaxed);
    statsOut->d2h_event_us = g_stats.D2hEventUs.load(std::memory_order_relaxed);
    statsOut->e2e_event_us = g_stats.E2eEventUs.load(std::memory_order_relaxed);
    statsOut->bytes_h2d = g_stats.BytesH2d.load(std::memory_order_relaxed);
    statsOut->bytes_d2h = g_stats.BytesD2h.load(std::memory_order_relaxed);
    return true;
}
