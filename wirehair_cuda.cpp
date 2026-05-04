#include "WirehairCudaDispatch.h"

#include "WirehairCodec.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
#include "WirehairCudaKernels.cuh"
#else
enum WirehairCudaCoreOp : uint32_t
{
    WirehairCudaCoreOp_None = 0,
    WirehairCudaCoreOp_EncodeParity = 1,
    WirehairCudaCoreOp_DecodeSolve = 2,
};

enum WirehairCudaCoreBatchOption : uint32_t
{
    WirehairCudaCoreBatchOption_None = 0,
    WirehairCudaCoreBatchOption_AllowHostRegister = 1 << 0,
    WirehairCudaCoreBatchOption_SkipOutputCopy = 1 << 1,
    WirehairCudaCoreBatchOption_SkipChecksumCopy = 1 << 2,
};

struct WirehairCudaCoreBatchParams
{
    uint32_t struct_bytes;
    uint32_t op;
    uint32_t item_count;
    uint32_t item_stride_bytes;
    uint32_t item_bytes;
    uint32_t options;
};

struct WirehairCudaCoreBatchResult
{
    uint32_t struct_bytes;
    uint32_t processed_count;
    uint32_t checksum;
};
#endif

namespace {

std::atomic<uint32_t> g_backendMode(WirehairCudaBackend_Auto);
std::atomic<int32_t> g_deviceOrdinal(-1);
std::atomic<uint32_t> g_streamCount(1);
std::atomic<uint32_t> g_usePinnedMemory(1);
std::atomic<int32_t> g_cudaReadyState(0); // 0 unknown, 1 ready, -1 unavailable
std::atomic<int32_t> g_cudaReadyOrdinal((std::numeric_limits<int32_t>::min)());
std::atomic<uint32_t> g_dynamicOffloadEncodeBytes(2U * 1024U * 1024U);
std::atomic<uint32_t> g_dynamicOffloadDecodeBytes(1U * 1024U * 1024U);
std::atomic<uint32_t> g_cudaDecisionCounter(0);

static const uint32_t kMinCudaOffloadBytes = 256U * 1024U;
static const uint32_t kMaxCudaOffloadBytes = 64U * 1024U * 1024U;

thread_local WirehairCudaPath g_lastPath = WirehairCudaPath_CPU;

bool KernelProbe(uint32_t* deviceCountOut)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    return WirehairCudaKernelProbe(deviceCountOut);
#else
    if (deviceCountOut) {
        *deviceCountOut = 0;
    }
    return false;
#endif
}

bool KernelEnableDevice(int32_t deviceOrdinal)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    return WirehairCudaKernelEnableDevice(deviceOrdinal);
#else
    (void)deviceOrdinal;
    return false;
#endif
}

bool KernelEncodeAssist(void* data, uint32_t bytes)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    return WirehairCudaKernelEncodeAssist(data, bytes);
#else
    (void)data;
    (void)bytes;
    return false;
#endif
}

bool KernelDecodeAssist(const void* data, uint32_t bytes, uint32_t* checksumOut)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    return WirehairCudaKernelDecodeAssist(data, bytes, checksumOut);
#else
    (void)data;
    (void)bytes;
    if (checksumOut) {
        *checksumOut = 0;
    }
    return false;
#endif
}

bool KernelXorInPlace(void* destData, const void* srcData, uint32_t bytes)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    return WirehairCudaKernelXorInPlace(destData, srcData, bytes);
#else
    (void)destData;
    (void)srcData;
    (void)bytes;
    return false;
#endif
}

bool KernelProcessBatch(
    const WirehairCudaCoreBatchParams* params,
    const void* inputData,
    void* outputData,
    WirehairCudaCoreBatchResult* resultOut)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    return WirehairCudaKernelProcessBatch(params, inputData, outputData, resultOut);
#else
    (void)params;
    (void)inputData;
    (void)outputData;
    (void)resultOut;
    return false;
#endif
}

bool KernelConfigure(uint32_t streamCount, uint32_t usePinnedMemory)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    return WirehairCudaKernelConfigure(streamCount, usePinnedMemory);
#else
    (void)streamCount;
    (void)usePinnedMemory;
    return false;
#endif
}

void KernelResetStats()
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    WirehairCudaKernelResetStats();
#endif
}

bool KernelGetStats(WirehairCudaPerfStats* statsOut)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    if (!statsOut) {
        return false;
    }
    WirehairCudaKernelStats kernelStats = {};
    if (!WirehairCudaKernelGetStats(&kernelStats)) {
        return false;
    }
    statsOut->encode_calls = kernelStats.encode_calls;
    statsOut->decode_calls = kernelStats.decode_calls;
    statsOut->setup_us = kernelStats.setup_us;
    statsOut->h2d_us = kernelStats.h2d_us;
    statsOut->kernel_us = kernelStats.kernel_us;
    statsOut->d2h_us = kernelStats.d2h_us;
    statsOut->sync_us = kernelStats.sync_us;
    statsOut->h2d_event_us = kernelStats.h2d_event_us;
    statsOut->kernel_event_us = kernelStats.kernel_event_us;
    statsOut->d2h_event_us = kernelStats.d2h_event_us;
    statsOut->e2e_event_us = kernelStats.e2e_event_us;
    statsOut->enqueue_us = kernelStats.enqueue_us;
    statsOut->queue_stall_us = kernelStats.queue_stall_us;
    statsOut->queue_depth_samples = kernelStats.queue_depth_samples;
    statsOut->queue_depth_total = kernelStats.queue_depth_total;
    statsOut->device_idle_us = kernelStats.device_idle_us;
    statsOut->submit_batches = kernelStats.submit_batches;
    statsOut->submit_items = kernelStats.submit_items;
    statsOut->producer_wait_us = kernelStats.producer_wait_us;
    statsOut->transfer_wait_us = kernelStats.transfer_wait_us;
    statsOut->compute_wait_us = kernelStats.compute_wait_us;
    statsOut->completion_wait_us = kernelStats.completion_wait_us;
    statsOut->bytes_h2d = kernelStats.bytes_h2d;
    statsOut->bytes_d2h = kernelStats.bytes_d2h;
    return true;
#else
    (void)statsOut;
    return false;
#endif
}

WirehairCudaConfig SnapshotConfig()
{
    WirehairCudaConfig cfg = {};
    cfg.struct_bytes = sizeof(WirehairCudaConfig);
    cfg.backend_mode = g_backendMode.load(std::memory_order_relaxed);
    cfg.device_ordinal = g_deviceOrdinal.load(std::memory_order_relaxed);
    cfg.stream_count = g_streamCount.load(std::memory_order_relaxed);
    cfg.use_pinned_memory = g_usePinnedMemory.load(std::memory_order_relaxed);
    return cfg;
}

WirehairResult ValidateConfig(const WirehairCudaConfig& cfg)
{
    if (cfg.struct_bytes != sizeof(WirehairCudaConfig)) {
        return Wirehair_InvalidInput;
    }
    if (cfg.backend_mode >= WirehairCudaBackend_Count) {
        return Wirehair_InvalidInput;
    }
    if (cfg.stream_count == 0) {
        return Wirehair_InvalidInput;
    }
    return Wirehair_Success;
}

bool ProbeCuda(uint32_t* deviceCountOut)
{
    return KernelProbe(deviceCountOut);
}

bool PrepareCudaDevice()
{
    const int32_t deviceOrdinal = g_deviceOrdinal.load(std::memory_order_relaxed);
    return KernelEnableDevice(deviceOrdinal);
}

bool IsCudaReady()
{
    const int32_t currentOrdinal = g_deviceOrdinal.load(std::memory_order_relaxed);
    const int32_t cachedOrdinal = g_cudaReadyOrdinal.load(std::memory_order_relaxed);
    const int32_t cachedState = g_cudaReadyState.load(std::memory_order_relaxed);

    if (cachedOrdinal == currentOrdinal && cachedState != 0) {
        return cachedState > 0;
    }

    uint32_t deviceCount = 0;
    bool ready = ProbeCuda(&deviceCount) && deviceCount > 0 && PrepareCudaDevice();
    if (ready) {
        ready = KernelConfigure(
            g_streamCount.load(std::memory_order_relaxed),
            g_usePinnedMemory.load(std::memory_order_relaxed));
    }
    g_cudaReadyOrdinal.store(currentOrdinal, std::memory_order_relaxed);
    g_cudaReadyState.store(ready ? 1 : -1, std::memory_order_relaxed);
    return ready;
}

WirehairResult EncodeCpuOnly(
    wirehair::Codec* codec,
    unsigned blockId,
    void* blockDataOut,
    uint32_t outBytes,
    uint32_t* dataBytesOut)
{
    const uint32_t writtenBytes = codec->Encode(blockId, blockDataOut, outBytes);
    *dataBytesOut = writtenBytes;

    if (writtenBytes == 0) {
        return Wirehair_InvalidInput;
    }
    g_lastPath = WirehairCudaPath_CPU;
    return Wirehair_Success;
}

WirehairResult DecodeCpuOnly(
    wirehair::Codec* codec,
    unsigned blockId,
    const void* blockData,
    uint32_t dataBytes)
{
    g_lastPath = WirehairCudaPath_CPU;
    return codec->DecodeFeed(blockId, blockData, dataBytes);
}

bool ShouldAttemptCuda(
    bool* cudaRequiredOut,
    uint32_t bytesToProcess,
    bool decodePath,
    uint32_t itemCount,
    uint32_t itemBytes)
{
    const uint32_t backendMode = g_backendMode.load(std::memory_order_relaxed);
    const bool cudaRequired = backendMode == WirehairCudaBackend_CudaOnly;
    if (cudaRequiredOut) {
        *cudaRequiredOut = cudaRequired;
    }
    if (backendMode == WirehairCudaBackend_CpuOnly) {
        return false;
    }

    const uint32_t dynamicThreshold = decodePath
        ? g_dynamicOffloadDecodeBytes.load(std::memory_order_relaxed)
        : g_dynamicOffloadEncodeBytes.load(std::memory_order_relaxed);
    const uint32_t streamCount = std::max<uint32_t>(1, g_streamCount.load(std::memory_order_relaxed));
    const uint32_t shapeScore = std::min<uint32_t>(8U, (itemCount / 64U) + (itemBytes / 2048U));
    uint32_t shapeAdjustedThreshold = dynamicThreshold;
    if (shapeScore >= 5U && streamCount > 1U) {
        shapeAdjustedThreshold = std::max<uint32_t>(kMinCudaOffloadBytes, dynamicThreshold / 2U);
    } else if (shapeScore <= 2U) {
        shapeAdjustedThreshold = std::min<uint32_t>(kMaxCudaOffloadBytes, dynamicThreshold * 2U);
    }
    const uint32_t minItems = decodePath ? 48U : 128U;
    const uint32_t minItemBytes = decodePath ? 2048U : 4096U;
    if (!decodePath && !cudaRequired && shapeScore < 5U) {
        return false;
    }
    if (!cudaRequired && (itemCount < minItems || itemBytes < minItemBytes)) {
        return false;
    }
    if (!cudaRequired && bytesToProcess < shapeAdjustedThreshold) {
        return false;
    }
    const uint32_t decisionCount = g_cudaDecisionCounter.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((decisionCount % 32) == 0)
    {
        WirehairCudaPerfStats stats = {};
        stats.struct_bytes = sizeof(WirehairCudaPerfStats);
        if (KernelGetStats(&stats) && stats.encode_calls + stats.decode_calls > 8)
        {
            uint64_t transferUs = stats.h2d_event_us + stats.d2h_event_us;
            uint64_t kernelUs = stats.kernel_event_us;
            const uint64_t enqueueUs = stats.enqueue_us;
            const uint64_t queueDepth = (stats.queue_depth_samples > 0)
                ? (stats.queue_depth_total / stats.queue_depth_samples)
                : 1;
            if (transferUs == 0 || kernelUs == 0) {
                transferUs = stats.h2d_us + stats.d2h_us + stats.sync_us;
                kernelUs = stats.kernel_us;
            }
            uint32_t threshold = decodePath
                ? g_dynamicOffloadDecodeBytes.load(std::memory_order_relaxed)
                : g_dynamicOffloadEncodeBytes.load(std::memory_order_relaxed);
            if (transferUs > ((kernelUs * 3) / 2) && threshold < kMaxCudaOffloadBytes) {
                threshold = std::min<uint32_t>(kMaxCudaOffloadBytes, threshold * 2U);
            } else if (kernelUs > transferUs && threshold > kMinCudaOffloadBytes) {
                threshold = std::max<uint32_t>(kMinCudaOffloadBytes, threshold / 2U);
            }
            if (queueDepth <= 1 && enqueueUs > kernelUs && threshold < kMaxCudaOffloadBytes) {
                threshold = std::min<uint32_t>(kMaxCudaOffloadBytes, threshold * 2U);
            }
            if (queueDepth >= 3 && kernelUs > transferUs && threshold > kMinCudaOffloadBytes) {
                threshold = std::max<uint32_t>(kMinCudaOffloadBytes, threshold / 2U);
            }
            if (decodePath) {
                g_dynamicOffloadDecodeBytes.store(threshold, std::memory_order_relaxed);
            } else {
                g_dynamicOffloadEncodeBytes.store(threshold, std::memory_order_relaxed);
            }
        }
    }
    return IsCudaReady();
}

uint32_t ComputeChunkItemCount(
    bool decodePath,
    uint32_t totalItems,
    uint32_t itemStrideBytes)
{
    bool highMemoryProfile = g_streamCount.load(std::memory_order_relaxed) >= 8U;
    const char* envProfile = std::getenv("WIREHAIR_CUDA_HIGH_MEM");
    if (envProfile) {
        highMemoryProfile = (std::strcmp(envProfile, "0") != 0 && std::strcmp(envProfile, "false") != 0);
    }
    const uint32_t streams = std::max<uint32_t>(1, g_streamCount.load(std::memory_order_relaxed));
    const bool heavyShape = (totalItems >= 2048U) && (itemStrideBytes >= 4096U);
    const bool mediumShape = (totalItems >= 1024U) && (itemStrideBytes >= 2048U);
    const uint32_t targetChunkBytes = highMemoryProfile
        ? (heavyShape
            ? (decodePath ? (96U * 1024U * 1024U) : (64U * 1024U * 1024U))
            : (mediumShape
                ? (decodePath ? (24U * 1024U * 1024U) : (16U * 1024U * 1024U))
                : (decodePath ? (8U * 1024U * 1024U) : (6U * 1024U * 1024U))))
        : (heavyShape
            ? (decodePath ? (32U * 1024U * 1024U) : (24U * 1024U * 1024U))
            : (decodePath ? (8U * 1024U * 1024U) : (6U * 1024U * 1024U)));
    uint32_t items = itemStrideBytes > 0 ? (targetChunkBytes / itemStrideBytes) : 1U;
    if (items == 0) {
        items = 1;
    }
    const uint32_t minChunk = highMemoryProfile
        ? (heavyShape
            ? (decodePath ? (streams * 64U) : (streams * 48U))
            : (decodePath ? (streams * 10U) : (streams * 8U)))
        : (decodePath ? (streams * 8U) : (streams * 6U));
    const uint32_t maxChunk = highMemoryProfile
        ? (heavyShape
            ? (decodePath ? 32768U : 24576U)
            : (decodePath ? 4096U : 3072U))
        : (heavyShape
            ? (decodePath ? 8192U : 4096U)
            : (decodePath ? 2048U : 1536U));
    items = std::max<uint32_t>(minChunk, std::min<uint32_t>(maxChunk, items));
    return std::min<uint32_t>(totalItems, items);
}

} // namespace

WirehairResult WirehairCudaDispatchEncode(
    wirehair::Codec* codec,
    unsigned blockId,
    void* blockDataOut,
    uint32_t outBytes,
    uint32_t* dataBytesOut)
{
    if (!codec || !blockDataOut || !dataBytesOut) {
        return Wirehair_InvalidInput;
    }

    const uint32_t backendMode = g_backendMode.load(std::memory_order_relaxed);
    if (backendMode == WirehairCudaBackend_CudaOnly) {
        return Wirehair_UnsupportedPlatform;
    }
    return EncodeCpuOnly(codec, blockId, blockDataOut, outBytes, dataBytesOut);
}

WirehairResult WirehairCudaDispatchDecode(
    wirehair::Codec* codec,
    unsigned blockId,
    const void* blockData,
    uint32_t dataBytes)
{
    if (!codec || !blockData || dataBytes == 0) {
        return Wirehair_InvalidInput;
    }

    const uint32_t backendMode = g_backendMode.load(std::memory_order_relaxed);
    if (backendMode == WirehairCudaBackend_CudaOnly) {
        return Wirehair_UnsupportedPlatform;
    }
    return DecodeCpuOnly(codec, blockId, blockData, dataBytes);
}

bool WirehairCudaDispatchXorInPlace(
    void* destData,
    const void* srcData,
    uint32_t bytes)
{
    if (!destData || !srcData || bytes == 0) {
        return false;
    }
    bool cudaRequired = false;
    if (!ShouldAttemptCuda(&cudaRequired, bytes, false, 1, bytes)) {
        return false;
    }
    return KernelXorInPlace(destData, srcData, bytes);
}

extern "C" {

WIREHAIR_EXPORT void wirehair_cuda_get_default_config(
    WirehairCudaConfig* config_out
)
{
    if (!config_out) {
        return;
    }
    config_out->struct_bytes = sizeof(WirehairCudaConfig);
    config_out->backend_mode = WirehairCudaBackend_Auto;
    config_out->device_ordinal = -1;
    config_out->stream_count = 1;
    config_out->use_pinned_memory = 1;
}

WIREHAIR_EXPORT WirehairResult wirehair_cuda_set_config(
    const WirehairCudaConfig* config
)
{
    if (!config) {
        return Wirehair_InvalidInput;
    }
    const WirehairResult validation = ValidateConfig(*config);
    if (validation != Wirehair_Success) {
        return validation;
    }

    g_backendMode.store(config->backend_mode, std::memory_order_relaxed);
    g_deviceOrdinal.store(config->device_ordinal, std::memory_order_relaxed);
    g_streamCount.store(config->stream_count, std::memory_order_relaxed);
    g_usePinnedMemory.store(config->use_pinned_memory != 0 ? 1U : 0U, std::memory_order_relaxed);
    g_cudaReadyState.store(0, std::memory_order_relaxed);
    g_cudaReadyOrdinal.store((std::numeric_limits<int32_t>::min)(), std::memory_order_relaxed);
    g_dynamicOffloadEncodeBytes.store(kMinCudaOffloadBytes, std::memory_order_relaxed);
    g_dynamicOffloadDecodeBytes.store(kMinCudaOffloadBytes, std::memory_order_relaxed);
    g_cudaDecisionCounter.store(0, std::memory_order_relaxed);
    KernelConfigure(g_streamCount.load(std::memory_order_relaxed), g_usePinnedMemory.load(std::memory_order_relaxed));
    return Wirehair_Success;
}

WIREHAIR_EXPORT WirehairResult wirehair_cuda_get_config(
    WirehairCudaConfig* config_out
)
{
    if (!config_out) {
        return Wirehair_InvalidInput;
    }
    *config_out = SnapshotConfig();
    return Wirehair_Success;
}

WIREHAIR_EXPORT WirehairResult wirehair_cuda_is_available(
    uint32_t* device_count_out
)
{
    uint32_t count = 0;
    const bool available = ProbeCuda(&count);
    if (device_count_out) {
        *device_count_out = count;
    }
    return available ? Wirehair_Success : Wirehair_UnsupportedPlatform;
}

WIREHAIR_EXPORT WirehairCudaPath wirehair_cuda_get_last_path(
    void
)
{
    return g_lastPath;
}

WIREHAIR_EXPORT WirehairResult wirehair_cuda_get_perf_stats(
    WirehairCudaPerfStats* stats_out
)
{
    if (!stats_out) {
        return Wirehair_InvalidInput;
    }
    if (stats_out->struct_bytes != 0 && stats_out->struct_bytes != sizeof(WirehairCudaPerfStats)) {
        return Wirehair_InvalidInput;
    }
    std::memset(stats_out, 0, sizeof(WirehairCudaPerfStats));
    stats_out->struct_bytes = sizeof(WirehairCudaPerfStats);
    if (!KernelGetStats(stats_out)) {
        return Wirehair_UnsupportedPlatform;
    }
    return Wirehair_Success;
}

WIREHAIR_EXPORT void wirehair_cuda_reset_perf_stats(
    void
)
{
    KernelResetStats();
}

WIREHAIR_EXPORT WirehairResult wirehair_cuda_encode_batch(
    const WirehairEncodeBatchRequest* request
)
{
    if (!request || !request->message || request->message_bytes == 0 || request->block_bytes == 0 ||
        request->block_count == 0 || !request->block_data_out || request->block_stride_bytes < request->block_bytes ||
        !request->bytes_out)
    {
        return Wirehair_InvalidInput;
    }

    WirehairCodec codec = wirehair_encoder_create(nullptr, request->message, request->message_bytes, request->block_bytes);
    if (!codec) {
        return Wirehair_Error;
    }

    wirehair::Codec* encoder = reinterpret_cast<wirehair::Codec*>(codec);
    for (uint32_t i = 0; i < request->block_count; ++i)
    {
        uint8_t* output = reinterpret_cast<uint8_t*>(request->block_data_out) + static_cast<size_t>(i) * request->block_stride_bytes;
        const uint32_t written = encoder->Encode(request->start_block_id + i, output, request->block_bytes);
        request->bytes_out[i] = written;
        if (written == 0) {
            wirehair_free(codec);
            return Wirehair_InvalidInput;
        }
    }
    const uint32_t backendMode = g_backendMode.load(std::memory_order_relaxed);
    if (backendMode != WirehairCudaBackend_CudaOnly) {
        g_lastPath = WirehairCudaPath_CPU;
        wirehair_free(codec);
        return Wirehair_Success;
    }

    bool cudaRequired = false;
    const uint64_t totalBytes64 = static_cast<uint64_t>(request->block_count) * request->block_stride_bytes;
    const uint32_t maxU32 = (std::numeric_limits<uint32_t>::max)();
    const uint32_t totalBytes = totalBytes64 > maxU32 ? maxU32 : static_cast<uint32_t>(totalBytes64);
    if (ShouldAttemptCuda(&cudaRequired, totalBytes, false, request->block_count, request->block_bytes))
    {
        const uint32_t chunkItems = ComputeChunkItemCount(false, request->block_count, request->block_stride_bytes);
        uint32_t offset = 0;
        while (offset < request->block_count)
        {
            const uint32_t currentItems = std::min<uint32_t>(chunkItems, request->block_count - offset);
            WirehairCudaCoreBatchParams params = {};
            params.struct_bytes = sizeof(WirehairCudaCoreBatchParams);
            params.op = WirehairCudaCoreOp_EncodeParity;
            params.item_count = currentItems;
            params.item_stride_bytes = request->block_stride_bytes;
            params.item_bytes = request->block_bytes;
            params.options = WirehairCudaCoreBatchOption_AllowHostRegister |
                WirehairCudaCoreBatchOption_SkipChecksumCopy;
            WirehairCudaCoreBatchResult batchResult = {};
            batchResult.struct_bytes = sizeof(WirehairCudaCoreBatchResult);
            uint8_t* chunkBase = reinterpret_cast<uint8_t*>(request->block_data_out) +
                static_cast<size_t>(offset) * request->block_stride_bytes;
            if (!KernelProcessBatch(&params, chunkBase, chunkBase, &batchResult))
            {
                wirehair_free(codec);
                if (cudaRequired) {
                    return Wirehair_Error;
                }
                g_lastPath = WirehairCudaPath_CPU;
                return Wirehair_Success;
            }
            offset += currentItems;
        }
        g_lastPath = WirehairCudaPath_CUDA;
    }
    else if (cudaRequired) {
        wirehair_free(codec);
        return Wirehair_UnsupportedPlatform;
    }

    wirehair_free(codec);
    return Wirehair_Success;
}

WIREHAIR_EXPORT WirehairResult wirehair_cuda_decode_batch(
    const WirehairDecodeBatchRequest* request
)
{
    if (!request || request->message_bytes == 0 || request->block_bytes == 0 || request->symbol_count == 0 ||
        !request->block_ids || !request->block_data || !request->block_data_bytes ||
        request->block_stride_bytes < request->block_bytes || !request->message_out)
    {
        return Wirehair_InvalidInput;
    }

    WirehairCodec decoder = wirehair_decoder_create(nullptr, request->message_bytes, request->block_bytes);
    if (!decoder) {
        return Wirehair_Error;
    }

    bool cudaRequired = false;
    const uint64_t totalBytes64 = static_cast<uint64_t>(request->symbol_count) * request->block_stride_bytes;
    const uint32_t maxU32 = (std::numeric_limits<uint32_t>::max)();
    const uint32_t totalBytes = totalBytes64 > maxU32 ? maxU32 : static_cast<uint32_t>(totalBytes64);
    if (ShouldAttemptCuda(&cudaRequired, totalBytes, true, request->symbol_count, request->block_bytes))
    {
        const uint32_t chunkItems = ComputeChunkItemCount(true, request->symbol_count, request->block_stride_bytes);
        uint32_t offset = 0;
        bool preprocessOk = true;
        while (offset < request->symbol_count)
        {
            const uint32_t currentItems = std::min<uint32_t>(chunkItems, request->symbol_count - offset);
            WirehairCudaCoreBatchParams params = {};
            params.struct_bytes = sizeof(WirehairCudaCoreBatchParams);
            params.op = WirehairCudaCoreOp_DecodeSolve;
            params.item_count = currentItems;
            params.item_stride_bytes = request->block_stride_bytes;
            params.item_bytes = request->block_bytes;
            params.options = WirehairCudaCoreBatchOption_AllowHostRegister |
                WirehairCudaCoreBatchOption_SkipOutputCopy |
                WirehairCudaCoreBatchOption_SkipChecksumCopy;
            WirehairCudaCoreBatchResult batchResult = {};
            batchResult.struct_bytes = sizeof(WirehairCudaCoreBatchResult);
            const uint8_t* chunkBase = reinterpret_cast<const uint8_t*>(request->block_data) +
                static_cast<size_t>(offset) * request->block_stride_bytes;
            if (!KernelProcessBatch(&params, chunkBase, nullptr, &batchResult))
            {
                preprocessOk = false;
                break;
            }
            offset += currentItems;
        }
        if (!preprocessOk)
        {
            wirehair_free(decoder);
            if (cudaRequired) {
                return Wirehair_Error;
            }
            g_lastPath = WirehairCudaPath_CPU;
        }
        else {
            g_lastPath = WirehairCudaPath_CUDA;
        }
    }
    else if (cudaRequired) {
        wirehair_free(decoder);
        return Wirehair_UnsupportedPlatform;
    }

    wirehair::Codec* decoderImpl = reinterpret_cast<wirehair::Codec*>(decoder);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(request->block_data);
    WirehairResult decodeResult = Wirehair_NeedMore;
    for (uint32_t i = 0; i < request->symbol_count; ++i)
    {
        const uint8_t* symbol = base + static_cast<size_t>(i) * request->block_stride_bytes;
        decodeResult = decoderImpl->DecodeFeed(request->block_ids[i], symbol, request->block_data_bytes[i]);
        if (decodeResult == Wirehair_Success) {
            break;
        }
        if (decodeResult != Wirehair_NeedMore) {
            wirehair_free(decoder);
            return decodeResult;
        }
    }

    if (decodeResult != Wirehair_Success) {
        wirehair_free(decoder);
        return decodeResult;
    }

    const WirehairResult recover = wirehair_recover(decoder, request->message_out, request->message_bytes);
    wirehair_free(decoder);
    return recover;
}

} // extern "C"
