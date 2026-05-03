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
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
};

bool WirehairCudaKernelProbe(uint32_t* deviceCountOut);
bool WirehairCudaKernelEnableDevice(int32_t deviceOrdinal);
bool WirehairCudaKernelConfigure(uint32_t streamCount, uint32_t usePinnedMemory);
bool WirehairCudaKernelEncodeAssist(void* data, uint32_t bytes);
bool WirehairCudaKernelDecodeAssist(const void* data, uint32_t bytes, uint32_t* checksumOut);
void WirehairCudaKernelResetStats();
bool WirehairCudaKernelGetStats(WirehairCudaKernelStats* statsOut);

#endif // WIREHAIR_CUDA_KERNELS_CUH
