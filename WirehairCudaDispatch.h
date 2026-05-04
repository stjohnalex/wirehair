#ifndef WIREHAIR_CUDA_DISPATCH_H
#define WIREHAIR_CUDA_DISPATCH_H

#include <wirehair/wirehair.h>
#include <wirehair/wirehair_cuda.h>

namespace wirehair {
class Codec;
}

struct WirehairCudaSolverStageContract
{
    uint32_t stage = 0;
    uint32_t input_rows = 0;
    uint32_t dense_count = 0;
    uint32_t mix_count = 0;
    uint64_t epoch = 0;
};

bool WirehairCudaDispatchGetSolverStageContract(
    wirehair::Codec* codec,
    WirehairCudaSolverStageContract* contract_out
);

WirehairResult WirehairCudaDispatchEncode(
    wirehair::Codec* codec,
    unsigned blockId,
    void* blockDataOut,
    uint32_t outBytes,
    uint32_t* dataBytesOut
);

WirehairResult WirehairCudaDispatchDecode(
    wirehair::Codec* codec,
    unsigned blockId,
    const void* blockData,
    uint32_t dataBytes
);

bool WirehairCudaDispatchXorInPlace(
    void* destData,
    const void* srcData,
    uint32_t bytes
);

#endif // WIREHAIR_CUDA_DISPATCH_H
