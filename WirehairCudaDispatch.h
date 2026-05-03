#ifndef WIREHAIR_CUDA_DISPATCH_H
#define WIREHAIR_CUDA_DISPATCH_H

#include <wirehair/wirehair.h>
#include <wirehair/wirehair_cuda.h>

namespace wirehair {
class Codec;
}

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
