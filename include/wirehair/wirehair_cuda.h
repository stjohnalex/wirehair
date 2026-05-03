#ifndef WIREHAIR_CUDA_H
#define WIREHAIR_CUDA_H

#include <wirehair/wirehair.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum WirehairCudaBackendMode_t
{
    WirehairCudaBackend_Auto = 0,
    WirehairCudaBackend_CpuOnly = 1,
    WirehairCudaBackend_CudaPrefer = 2,
    WirehairCudaBackend_CudaOnly = 3,
    WirehairCudaBackend_Count,
    WirehairCudaBackend_Padding = 0x7fffffff
} WirehairCudaBackendMode;

typedef enum WirehairCudaPath_t
{
    WirehairCudaPath_CPU = 0,
    WirehairCudaPath_CUDA = 1,
    WirehairCudaPath_Padding = 0x7fffffff
} WirehairCudaPath;

typedef struct WirehairCudaConfig_t
{
    uint32_t struct_bytes;
    uint32_t backend_mode;
    int32_t device_ordinal;
    uint32_t stream_count;
    uint32_t use_pinned_memory;
} WirehairCudaConfig;

typedef struct WirehairCudaPerfStats_t
{
    uint32_t struct_bytes;
    uint64_t encode_calls;
    uint64_t decode_calls;
    uint64_t setup_us;
    uint64_t h2d_us;
    uint64_t kernel_us;
    uint64_t d2h_us;
    uint64_t sync_us;
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
} WirehairCudaPerfStats;

WIREHAIR_EXPORT void wirehair_cuda_get_default_config(
    WirehairCudaConfig* config_out
);

WIREHAIR_EXPORT WirehairResult wirehair_cuda_set_config(
    const WirehairCudaConfig* config
);

WIREHAIR_EXPORT WirehairResult wirehair_cuda_get_config(
    WirehairCudaConfig* config_out
);

WIREHAIR_EXPORT WirehairResult wirehair_cuda_is_available(
    uint32_t* device_count_out
);

WIREHAIR_EXPORT WirehairCudaPath wirehair_cuda_get_last_path(
    void
);

WIREHAIR_EXPORT WirehairResult wirehair_cuda_encode_batch(
    const WirehairEncodeBatchRequest* request
);

WIREHAIR_EXPORT WirehairResult wirehair_cuda_decode_batch(
    const WirehairDecodeBatchRequest* request
);

WIREHAIR_EXPORT WirehairResult wirehair_cuda_get_perf_stats(
    WirehairCudaPerfStats* stats_out
);

WIREHAIR_EXPORT void wirehair_cuda_reset_perf_stats(
    void
);

#ifdef __cplusplus
}
#endif

#endif // WIREHAIR_CUDA_H
