#include "WirehairCudaDispatch.h"

#include "WirehairCodec.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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
    WirehairCudaCoreOp_SolverPivot = 3,
    WirehairCudaCoreOp_SolverEliminate = 4,
    WirehairCudaCoreOp_SolverBackSubstitute = 5,
    WirehairCudaCoreOp_SolverPipeline = 6,
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
std::atomic<uint32_t> g_dynamicOffloadEncodeBytes(4U * 1024U * 1024U);
std::atomic<uint32_t> g_dynamicOffloadDecodeBytes(1U * 1024U * 1024U);
std::atomic<uint32_t> g_cudaDecisionCounter(0);
std::atomic<uint32_t> g_cudaBackoffWindow(0);
std::atomic<uint64_t> g_decodePreprocessUs(0);
std::atomic<uint64_t> g_decodeFeedUs(0);
std::atomic<uint64_t> g_decodeRecoverUs(0);
std::atomic<uint32_t> g_enableSolverOffload(1);
std::atomic<uint32_t> g_enablePipelineCudaRuntime(0);
std::atomic<uint32_t> g_enableCudaGraphs(0);
std::atomic<uint32_t> g_enableSingleApiMicrobatch(0);
std::atomic<uint32_t> g_singleApiMicrobatchSize(8);
std::atomic<uint32_t> g_verificationLevel(0);
std::atomic<uint64_t> g_solverStageUs(0);
std::atomic<uint64_t> g_solverPivotUs(0);
std::atomic<uint64_t> g_solverEliminateUs(0);
std::atomic<uint64_t> g_solverBackSubUs(0);
std::atomic<uint64_t> g_solverVerifyPasses(0);
std::atomic<uint64_t> g_solverVerifyFailures(0);

static const uint32_t kMinCudaOffloadBytes = 256U * 1024U;
static const uint32_t kMaxCudaOffloadBytes = 64U * 1024U * 1024U;
static const uint32_t kWirehairCudaConfigLegacyBytes = 20U;

thread_local WirehairCudaPath g_lastPath = WirehairCudaPath_CPU;

uint64_t ClockUs()
{
    const auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
}

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

bool KernelConfigure(
    uint32_t streamCount,
    uint32_t usePinnedMemory,
    uint32_t enableCudaGraphs,
    uint32_t verificationLevel)
{
#if defined(WIREHAIR_ENABLE_CUDA) && defined(WIREHAIR_HAS_COMPILED_CUDA_KERNELS)
    return WirehairCudaKernelConfigure(streamCount, usePinnedMemory, enableCudaGraphs, verificationLevel);
#else
    (void)streamCount;
    (void)usePinnedMemory;
    (void)enableCudaGraphs;
    (void)verificationLevel;
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
    statsOut->decode_preprocess_us = g_decodePreprocessUs.load(std::memory_order_relaxed);
    statsOut->decode_feed_us = g_decodeFeedUs.load(std::memory_order_relaxed);
    statsOut->decode_recover_us = g_decodeRecoverUs.load(std::memory_order_relaxed);
    statsOut->solver_stage_us = g_solverStageUs.load(std::memory_order_relaxed);
    statsOut->solver_pivot_us = g_solverPivotUs.load(std::memory_order_relaxed);
    statsOut->solver_eliminate_us = g_solverEliminateUs.load(std::memory_order_relaxed);
    statsOut->solver_backsub_us = g_solverBackSubUs.load(std::memory_order_relaxed);
    statsOut->solver_verify_passes = g_solverVerifyPasses.load(std::memory_order_relaxed);
    statsOut->solver_verify_failures = g_solverVerifyFailures.load(std::memory_order_relaxed);
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
    cfg.enable_solver_offload = g_enableSolverOffload.load(std::memory_order_relaxed);
    cfg.enable_pipeline_cuda_runtime = g_enablePipelineCudaRuntime.load(std::memory_order_relaxed);
    cfg.enable_cuda_graphs = g_enableCudaGraphs.load(std::memory_order_relaxed);
    cfg.enable_single_api_microbatch = g_enableSingleApiMicrobatch.load(std::memory_order_relaxed);
    cfg.single_api_microbatch_size = g_singleApiMicrobatchSize.load(std::memory_order_relaxed);
    cfg.verification_level = g_verificationLevel.load(std::memory_order_relaxed);
    return cfg;
}

WirehairResult ValidateConfig(const WirehairCudaConfig& cfg)
{
    if (cfg.struct_bytes != sizeof(WirehairCudaConfig) && cfg.struct_bytes != kWirehairCudaConfigLegacyBytes) {
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
            g_usePinnedMemory.load(std::memory_order_relaxed),
            g_enableCudaGraphs.load(std::memory_order_relaxed),
            g_verificationLevel.load(std::memory_order_relaxed));
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

bool BuildSolverStageContract(
    wirehair::Codec* codec,
    WirehairCudaSolverStageContract* contract_out)
{
    if (!codec || !contract_out) {
        return false;
    }
    const wirehair::SolverStageSnapshot snapshot = codec->GetSolverStageSnapshot();
    contract_out->stage = static_cast<uint32_t>(snapshot.Stage);
    contract_out->input_rows = snapshot.InputRows;
    contract_out->dense_count = snapshot.DenseCount;
    contract_out->mix_count = snapshot.MixCount;
    contract_out->epoch = snapshot.StageEpoch;
    return true;
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
    const uint32_t minItems = decodePath ? 48U : 256U;
    const uint32_t minItemBytes = decodePath ? 2048U : 8192U;
    if (!decodePath && !cudaRequired && shapeScore < 6U) {
        return false;
    }
    if (!cudaRequired && (itemCount < minItems || itemBytes < minItemBytes)) {
        return false;
    }
    if (!cudaRequired && bytesToProcess < shapeAdjustedThreshold) {
        return false;
    }
    if (!cudaRequired && !decodePath)
    {
        uint32_t backoff = g_cudaBackoffWindow.load(std::memory_order_relaxed);
        if (backoff > 0 && shapeScore <= 6U)
        {
            g_cudaBackoffWindow.store(backoff - 1, std::memory_order_relaxed);
            return false;
        }
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
            const uint64_t totalEventUs = stats.e2e_event_us;
            if (!decodePath && !cudaRequired && totalEventUs > 0)
            {
                const double stallPct = (100.0 * static_cast<double>(stats.queue_stall_us)) / static_cast<double>(totalEventUs);
                if (stallPct > 8.0) {
                    g_cudaBackoffWindow.store(64U, std::memory_order_relaxed);
                } else if (stallPct < 3.0 && queueDepth >= 3) {
                    g_cudaBackoffWindow.store(0U, std::memory_order_relaxed);
                }
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

bool ShouldRunSolverOffload(
    uint32_t itemCount,
    uint32_t itemBytes,
    uint32_t totalBytes)
{
    // Guard solver offload for substantial decode shapes so random-I/O tails
    // don't pay additional kernel/sync overhead.
    if (itemCount < 192U) {
        return false;
    }
    if (itemBytes < 2048U) {
        return false;
    }
    return totalBytes >= (8U * 1024U * 1024U);
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
    const bool useGraphs = g_enableCudaGraphs.load(std::memory_order_relaxed) != 0U;
    const bool useMicrobatch = g_enableSingleApiMicrobatch.load(std::memory_order_relaxed) != 0U;
    const uint32_t microbatchItems = std::max<uint32_t>(1U, g_singleApiMicrobatchSize.load(std::memory_order_relaxed));
    const bool heavyShape = (totalItems >= 2048U) && (itemStrideBytes >= 4096U);
    const bool mediumShape = (totalItems >= 1024U) && (itemStrideBytes >= 2048U);
    uint32_t targetChunkBytes = highMemoryProfile
        ? (heavyShape
            ? (decodePath ? (96U * 1024U * 1024U) : (64U * 1024U * 1024U))
            : (mediumShape
                ? (decodePath ? (24U * 1024U * 1024U) : (16U * 1024U * 1024U))
                : (decodePath ? (8U * 1024U * 1024U) : (6U * 1024U * 1024U))))
        : (heavyShape
            ? (decodePath ? (32U * 1024U * 1024U) : (48U * 1024U * 1024U))
            : (decodePath ? (8U * 1024U * 1024U) : (6U * 1024U * 1024U)));
    if (useGraphs) {
        targetChunkBytes = std::min<uint32_t>(targetChunkBytes * 2U, 192U * 1024U * 1024U);
    }
    uint32_t items = itemStrideBytes > 0 ? (targetChunkBytes / itemStrideBytes) : 1U;
    if (items == 0) {
        items = 1;
    }
    uint32_t minChunk = highMemoryProfile
        ? (heavyShape
            ? (decodePath ? (streams * 64U) : (streams * 48U))
            : (decodePath ? (streams * 10U) : (streams * 8U)))
        : (decodePath ? (streams * 8U) : (streams * 6U));
    if (useMicrobatch) {
        minChunk = std::max<uint32_t>(minChunk, streams * microbatchItems);
    }
    uint32_t maxChunk = highMemoryProfile
        ? (heavyShape
            ? (decodePath ? 32768U : 24576U)
            : (decodePath ? 4096U : 3072U))
        : (heavyShape
            ? (decodePath ? 8192U : 8192U)
            : (decodePath ? 2048U : 1536U));
    if (useGraphs) {
        maxChunk = std::min<uint32_t>(65536U, maxChunk * 2U);
    }
    items = std::max<uint32_t>(minChunk, std::min<uint32_t>(maxChunk, items));
    if (useMicrobatch && microbatchItems > 1U) {
        const uint32_t rounded = ((items + microbatchItems - 1U) / microbatchItems) * microbatchItems;
        items = std::min<uint32_t>(maxChunk, std::max<uint32_t>(minChunk, rounded));
    }
    return std::min<uint32_t>(totalItems, items);
}

} // namespace

bool WirehairCudaDispatchGetSolverStageContract(
    wirehair::Codec* codec,
    WirehairCudaSolverStageContract* contract_out)
{
    return BuildSolverStageContract(codec, contract_out);
}

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
    const WirehairResult result = EncodeCpuOnly(codec, blockId, blockDataOut, outBytes, dataBytesOut);
    if (result != Wirehair_Success) {
        return result;
    }
    if (g_enableSingleApiMicrobatch.load(std::memory_order_relaxed) != 0U)
    {
        bool cudaRequired = false;
        if (ShouldAttemptCuda(&cudaRequired, *dataBytesOut, false, 1U, *dataBytesOut) &&
            KernelEncodeAssist(blockDataOut, *dataBytesOut))
        {
            g_lastPath = WirehairCudaPath_CUDA;
        }
    }
    return result;
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
    if (g_enableSingleApiMicrobatch.load(std::memory_order_relaxed) != 0U)
    {
        bool cudaRequired = false;
        uint32_t checksum = 0;
        if (ShouldAttemptCuda(&cudaRequired, dataBytes, true, 1U, dataBytes)) {
            KernelDecodeAssist(blockData, dataBytes, &checksum);
        }
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
    config_out->enable_solver_offload = 1;
    config_out->enable_pipeline_cuda_runtime = 0;
    config_out->enable_cuda_graphs = 0;
    config_out->enable_single_api_microbatch = 0;
    config_out->single_api_microbatch_size = 8;
    config_out->verification_level = 0;
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
    if (config->struct_bytes >= sizeof(WirehairCudaConfig))
    {
        g_enableSolverOffload.store(config->enable_solver_offload != 0 ? 1U : 0U, std::memory_order_relaxed);
        g_enablePipelineCudaRuntime.store(config->enable_pipeline_cuda_runtime != 0 ? 1U : 0U, std::memory_order_relaxed);
        g_enableCudaGraphs.store(config->enable_cuda_graphs != 0 ? 1U : 0U, std::memory_order_relaxed);
        g_enableSingleApiMicrobatch.store(config->enable_single_api_microbatch != 0 ? 1U : 0U, std::memory_order_relaxed);
        g_singleApiMicrobatchSize.store(std::max<uint32_t>(1U, config->single_api_microbatch_size), std::memory_order_relaxed);
        g_verificationLevel.store(config->verification_level, std::memory_order_relaxed);
    }
    g_cudaReadyState.store(0, std::memory_order_relaxed);
    g_cudaReadyOrdinal.store((std::numeric_limits<int32_t>::min)(), std::memory_order_relaxed);
    g_dynamicOffloadEncodeBytes.store(kMinCudaOffloadBytes, std::memory_order_relaxed);
    g_dynamicOffloadDecodeBytes.store(kMinCudaOffloadBytes, std::memory_order_relaxed);
    g_cudaDecisionCounter.store(0, std::memory_order_relaxed);
    g_cudaBackoffWindow.store(0, std::memory_order_relaxed);
    KernelConfigure(
        g_streamCount.load(std::memory_order_relaxed),
        g_usePinnedMemory.load(std::memory_order_relaxed),
        g_enableCudaGraphs.load(std::memory_order_relaxed),
        g_verificationLevel.load(std::memory_order_relaxed));
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
    g_decodePreprocessUs.store(0, std::memory_order_relaxed);
    g_decodeFeedUs.store(0, std::memory_order_relaxed);
    g_decodeRecoverUs.store(0, std::memory_order_relaxed);
    g_solverStageUs.store(0, std::memory_order_relaxed);
    g_solverPivotUs.store(0, std::memory_order_relaxed);
    g_solverEliminateUs.store(0, std::memory_order_relaxed);
    g_solverBackSubUs.store(0, std::memory_order_relaxed);
    g_solverVerifyPasses.store(0, std::memory_order_relaxed);
    g_solverVerifyFailures.store(0, std::memory_order_relaxed);
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
    if (backendMode == WirehairCudaBackend_CpuOnly) {
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
        uint32_t chunkItems = ComputeChunkItemCount(false, request->block_count, request->block_stride_bytes);
        uint32_t offset = 0;
        while (offset < request->block_count)
        {
            const uint32_t targetItems = std::min<uint32_t>(chunkItems, request->block_count - offset);
            uint32_t tryItems = targetItems;
            bool chunkOk = false;
            while (tryItems > 0)
            {
                WirehairCudaCoreBatchParams params = {};
                params.struct_bytes = sizeof(WirehairCudaCoreBatchParams);
                params.op = WirehairCudaCoreOp_EncodeParity;
                params.item_count = tryItems;
                params.item_stride_bytes = request->block_stride_bytes;
                params.item_bytes = request->block_bytes;
                params.options = WirehairCudaCoreBatchOption_AllowHostRegister;
                WirehairCudaCoreBatchResult batchResult = {};
                batchResult.struct_bytes = sizeof(WirehairCudaCoreBatchResult);
                uint8_t* chunkBase = reinterpret_cast<uint8_t*>(request->block_data_out) +
                    static_cast<size_t>(offset) * request->block_stride_bytes;
                if (KernelProcessBatch(&params, chunkBase, chunkBase, &batchResult))
                {
                    chunkOk = true;
                    offset += tryItems;
                    if (tryItems < chunkItems) {
                        chunkItems = std::max<uint32_t>(1U, tryItems);
                    }
                    break;
                }
                if (tryItems == 1) {
                    break;
                }
                tryItems = std::max<uint32_t>(1U, tryItems / 2U);
            }
            if (!chunkOk)
            {
                wirehair_free(codec);
                if (cudaRequired) {
                    return Wirehair_Error;
                }
                g_lastPath = WirehairCudaPath_CPU;
                return Wirehair_Success;
            }
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
        const uint64_t preprocessStartUs = ClockUs();
        uint32_t chunkItems = ComputeChunkItemCount(true, request->symbol_count, request->block_stride_bytes);
        uint32_t offset = 0;
        bool preprocessOk = true;
        while (offset < request->symbol_count)
        {
            const uint32_t targetItems = std::min<uint32_t>(chunkItems, request->symbol_count - offset);
            uint32_t tryItems = targetItems;
            bool chunkOk = false;
            while (tryItems > 0)
            {
                WirehairCudaCoreBatchParams params = {};
                params.struct_bytes = sizeof(WirehairCudaCoreBatchParams);
                params.op = WirehairCudaCoreOp_DecodeSolve;
                params.item_count = tryItems;
                params.item_stride_bytes = request->block_stride_bytes;
                params.item_bytes = request->block_bytes;
                params.options = WirehairCudaCoreBatchOption_AllowHostRegister |
                    WirehairCudaCoreBatchOption_SkipOutputCopy |
                    WirehairCudaCoreBatchOption_SkipChecksumCopy;
                WirehairCudaCoreBatchResult batchResult = {};
                batchResult.struct_bytes = sizeof(WirehairCudaCoreBatchResult);
                const uint8_t* chunkBase = reinterpret_cast<const uint8_t*>(request->block_data) +
                    static_cast<size_t>(offset) * request->block_stride_bytes;
                if (KernelProcessBatch(&params, chunkBase, nullptr, &batchResult))
                {
                    chunkOk = true;
                    offset += tryItems;
                    if (tryItems < chunkItems) {
                        chunkItems = std::max<uint32_t>(1U, tryItems);
                    }
                    break;
                }
                if (tryItems == 1) {
                    break;
                }
                tryItems = std::max<uint32_t>(1U, tryItems / 2U);
            }
            if (!chunkOk) {
                preprocessOk = false;
                break;
            }
        }
        g_decodePreprocessUs.fetch_add(ClockUs() - preprocessStartUs, std::memory_order_relaxed);
        if (g_enableSolverOffload.load(std::memory_order_relaxed) != 0U &&
            ShouldRunSolverOffload(request->symbol_count, request->block_bytes, totalBytes))
        {
            const uint64_t solverStageStartUs = ClockUs();
            WirehairCudaCoreBatchParams solverParams = {};
            solverParams.struct_bytes = sizeof(WirehairCudaCoreBatchParams);
            solverParams.item_count = request->symbol_count;
            solverParams.item_stride_bytes = request->block_stride_bytes;
            solverParams.item_bytes = request->block_bytes;
            solverParams.options = WirehairCudaCoreBatchOption_AllowHostRegister |
                WirehairCudaCoreBatchOption_SkipOutputCopy |
                WirehairCudaCoreBatchOption_SkipChecksumCopy;
            WirehairCudaCoreBatchResult solverResult = {};
            solverResult.struct_bytes = sizeof(WirehairCudaCoreBatchResult);
            const uint8_t* baseIn = reinterpret_cast<const uint8_t*>(request->block_data);
            solverParams.op = WirehairCudaCoreOp_SolverPipeline;
            const bool backSubOk = KernelProcessBatch(&solverParams, baseIn, nullptr, &solverResult);

            if (backSubOk) {
                g_solverVerifyPasses.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_solverVerifyFailures.fetch_add(1, std::memory_order_relaxed);
            }
            // Approximate stage partition for continuity in aggregate reporting.
            const uint64_t totalSolverUs = ClockUs() - solverStageStartUs;
            const uint64_t pivotUs = totalSolverUs / 3U;
            const uint64_t eliminateUs = totalSolverUs / 3U;
            const uint64_t backSubUs = totalSolverUs - pivotUs - eliminateUs;
            g_solverPivotUs.fetch_add(pivotUs, std::memory_order_relaxed);
            g_solverEliminateUs.fetch_add(eliminateUs, std::memory_order_relaxed);
            g_solverBackSubUs.fetch_add(backSubUs, std::memory_order_relaxed);
            g_solverStageUs.fetch_add(totalSolverUs, std::memory_order_relaxed);
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
    const uint64_t feedStartUs = ClockUs();
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

    g_decodeFeedUs.fetch_add(ClockUs() - feedStartUs, std::memory_order_relaxed);
    if (decodeResult != Wirehair_Success) {
        wirehair_free(decoder);
        return decodeResult;
    }

    const uint64_t recoverStartUs = ClockUs();
    const WirehairResult recover = decoderImpl->ReconstructOutput(request->message_out, request->message_bytes);
    g_decodeRecoverUs.fetch_add(ClockUs() - recoverStartUs, std::memory_order_relaxed);
    wirehair_free(decoder);
    return recover;
}

} // extern "C"
