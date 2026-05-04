#ifndef WIREHAIR_CUDA_KERNELS_CUH
#define WIREHAIR_CUDA_KERNELS_CUH

#include <stdint.h>

struct WirehairCudaKernelStats
{
    uint64_t encode_calls;
    uint64_t decode_calls;
    uint64_t setup_us;
    uint64_t h2d_us;
    uint64_t kernel_us;
    uint64_t d2h_us;
    uint64_t sync_us;
    uint64_t h2d_event_us;
    uint64_t kernel_event_us;
    uint64_t d2h_event_us;
    uint64_t e2e_event_us;
    uint64_t enqueue_us;
    uint64_t queue_stall_us;
    uint64_t queue_depth_samples;
    uint64_t queue_depth_total;
    uint64_t device_idle_us;
    uint64_t submit_batches;
    uint64_t submit_items;
    uint64_t producer_wait_us;
    uint64_t transfer_wait_us;
    uint64_t compute_wait_us;
    uint64_t completion_wait_us;
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
};

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

struct WirehairCudaCoreBatchParams
{
    uint32_t struct_bytes;
    uint32_t op;
    uint32_t item_count;
    uint32_t item_stride_bytes;
    uint32_t item_bytes;
    uint32_t options;
};

enum WirehairCudaCoreBatchOption : uint32_t
{
    WirehairCudaCoreBatchOption_None = 0,
    WirehairCudaCoreBatchOption_AllowHostRegister = 1 << 0,
    WirehairCudaCoreBatchOption_SkipOutputCopy = 1 << 1,
    WirehairCudaCoreBatchOption_SkipChecksumCopy = 1 << 2,
};

struct WirehairCudaCoreBatchResult
{
    uint32_t struct_bytes;
    uint32_t processed_count;
    uint32_t checksum;
};

bool WirehairCudaKernelProbe(uint32_t* deviceCountOut);
bool WirehairCudaKernelEnableDevice(int32_t deviceOrdinal);
bool WirehairCudaKernelConfigure(
    uint32_t streamCount,
    uint32_t usePinnedMemory,
    uint32_t enableCudaGraphs,
    uint32_t verificationLevel);
bool WirehairCudaKernelEncodeAssist(void* data, uint32_t bytes);
bool WirehairCudaKernelDecodeAssist(const void* data, uint32_t bytes, uint32_t* checksumOut);
bool WirehairCudaKernelXorInPlace(void* destData, const void* srcData, uint32_t bytes);
bool WirehairCudaKernelProcessBatch(
    const WirehairCudaCoreBatchParams* params,
    const void* inputData,
    void* outputData,
    WirehairCudaCoreBatchResult* resultOut);
void WirehairCudaKernelResetStats();
bool WirehairCudaKernelGetStats(WirehairCudaKernelStats* statsOut);

#endif // WIREHAIR_CUDA_KERNELS_CUH
