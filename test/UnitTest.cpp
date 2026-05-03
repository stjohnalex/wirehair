#include <wirehair/wirehair.h>
#include <wirehair/wirehair_cuda.h>

#include "SiameseTools.h"

#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
using namespace std;

#define ENABLE_OMP
#define ENABLE_PARTIAL_FINAL
#define ENABLE_ALL_RECOVERY_FINAL_BLOCK
#define BENCHMARK_SHORT_LIST

static const unsigned kMinN = 2;
static const unsigned kMaxN = 64000;
static const unsigned kMaxBlockBytes = 65;
static const unsigned kExtraWarnThreshold = 4;

static const uint8_t kPadChar = (uint8_t)0xee;

static void FillMessage(uint8_t* message, unsigned bytes, siamese::PCGRandom& prng)
{
    while (bytes >= 4)
    {
        uint32_t* words = reinterpret_cast<uint32_t*>(message);
        *words = prng.Next();
        message += 4;
        bytes -= 4;
    }

    if (bytes > 0)
    {
        uint32_t word = prng.Next();
        for (unsigned j = 0; j < bytes; ++j)
        {
            message[j] = (uint8_t)word;
            word >>= 8;
        }
    }
}

static bool Test_ConcurrentInitAndCodecCreate()
{
    static const int kThreadCount = 8;
    static const int kIterationsPerThread = 200;
    static const uint32_t kMessageBytes = 1023;
    static const uint32_t kBlockBytes = 64;

    atomic<bool> failed(false);
    vector<thread> workers;
    workers.reserve(kThreadCount);

    for (int threadIndex = 0; threadIndex < kThreadCount; ++threadIndex)
    {
        workers.emplace_back([threadIndex, &failed]() {
            siamese::PCGRandom prng;
            prng.Seed(0x8f3d7211ULL + threadIndex, 0x27b9ca34ULL + threadIndex);
            vector<uint8_t> message(kMessageBytes);
            FillMessage(message.data(), kMessageBytes, prng);

            for (int i = 0; i < kIterationsPerThread; ++i)
            {
                if (wirehair_init() != Wirehair_Success) {
                    failed.store(true);
                    return;
                }

                WirehairCodec encoder = wirehair_encoder_create(nullptr, message.data(), kMessageBytes, kBlockBytes);
                WirehairCodec decoder = wirehair_decoder_create(nullptr, kMessageBytes, kBlockBytes);
                if (!encoder || !decoder)
                {
                    if (encoder) {
                        wirehair_free(encoder);
                    }
                    if (decoder) {
                        wirehair_free(decoder);
                    }
                    failed.store(true);
                    return;
                }

                wirehair_free(decoder);
                wirehair_free(encoder);
            }
        });
    }

    for (thread& worker : workers) {
        worker.join();
    }
    return !failed.load();
}

static bool Test_AsyncBatchDeterminism()
{
    static const uint32_t kMessageBytes = 128 * 1024 + 13;
    static const uint32_t kBlockBytes = 1024;
    static const uint32_t kStartBlockId = 300;
    static const uint32_t kBlockCount = 96;

    siamese::PCGRandom prng;
    prng.Seed(0x71ULL, 0x42ULL);
    vector<uint8_t> message(kMessageBytes);
    FillMessage(message.data(), kMessageBytes, prng);
    vector<uint8_t> encodedSingle(kBlockCount * kBlockBytes, 0);
    vector<uint8_t> encodedMulti(kBlockCount * kBlockBytes, 0);
    vector<uint32_t> bytesSingle(kBlockCount, 0);
    vector<uint32_t> bytesMulti(kBlockCount, 0);

    WirehairPipelineConfig oneCfg = {1, 1, 32};
    WirehairPipelineConfig manyCfg = {4, 4, 32};
    WirehairPipeline one = wirehair_pipeline_create(&oneCfg);
    WirehairPipeline many = wirehair_pipeline_create(&manyCfg);
    if (!one || !many) {
        if (one) wirehair_pipeline_free(one);
        if (many) wirehair_pipeline_free(many);
        return false;
    }

    WirehairEncodeBatchRequest reqSingle = {};
    reqSingle.request_id = 1;
    reqSingle.message = message.data();
    reqSingle.message_bytes = kMessageBytes;
    reqSingle.block_bytes = kBlockBytes;
    reqSingle.start_block_id = kStartBlockId;
    reqSingle.block_count = kBlockCount;
    reqSingle.block_data_out = encodedSingle.data();
    reqSingle.block_stride_bytes = kBlockBytes;
    reqSingle.bytes_out = bytesSingle.data();

    WirehairEncodeBatchRequest reqMulti = reqSingle;
    reqMulti.request_id = 2;
    reqMulti.block_data_out = encodedMulti.data();
    reqMulti.bytes_out = bytesMulti.data();

    WirehairPipelineEvent event = {};
    if (wirehair_encode_batch_async(one, &reqSingle) != Wirehair_Success ||
        wirehair_encode_batch_async(many, &reqMulti) != Wirehair_Success)
    {
        wirehair_pipeline_free(one);
        wirehair_pipeline_free(many);
        return false;
    }
    while (wirehair_pipeline_poll(one, 1000, &event) == Wirehair_NeedMore) {}
    if (event.result != Wirehair_Success) {
        wirehair_pipeline_free(one);
        wirehair_pipeline_free(many);
        return false;
    }
    while (wirehair_pipeline_poll(many, 1000, &event) == Wirehair_NeedMore) {}
    if (event.result != Wirehair_Success) {
        wirehair_pipeline_free(one);
        wirehair_pipeline_free(many);
        return false;
    }

    wirehair_pipeline_free(one);
    wirehair_pipeline_free(many);

    if (bytesSingle != bytesMulti) {
        return false;
    }
    return memcmp(encodedSingle.data(), encodedMulti.data(), encodedSingle.size()) == 0;
}

static bool Test_AsyncPipelineStress()
{
    const uint32_t kMessageBytes = 64 * 1024 + 17;
    const uint32_t kBlockBytes = 512;
    const uint32_t kSymbols = 144;
    const int kRequests = 48;
    siamese::PCGRandom prng;
    prng.Seed(0x9981ULL, 0x5512ULL);

    WirehairPipelineConfig cfg = {8, 8, 64};
    WirehairPipeline pipeline = wirehair_pipeline_create(&cfg);
    if (!pipeline) {
        return false;
    }

    vector<vector<uint8_t>> messages(kRequests);
    vector<vector<uint8_t>> encoded(kRequests);
    vector<vector<uint8_t>> recovered(kRequests);
    vector<vector<uint32_t>> lens(kRequests);
    vector<vector<uint32_t>> ids(kRequests);
    for (int i = 0; i < kRequests; ++i)
    {
        messages[i].resize(kMessageBytes);
        encoded[i].resize(kSymbols * kBlockBytes);
        recovered[i].resize(kMessageBytes);
        lens[i].assign(kSymbols, 0);
        ids[i].assign(kSymbols, 0);
        FillMessage(messages[i].data(), kMessageBytes, prng);
        for (uint32_t j = 0; j < kSymbols; ++j) {
            ids[i][j] = 256 + j;
        }
        WirehairEncodeBatchRequest enc = {};
        enc.request_id = static_cast<uint64_t>(i + 1);
        enc.message = messages[i].data();
        enc.message_bytes = kMessageBytes;
        enc.block_bytes = kBlockBytes;
        enc.start_block_id = 256;
        enc.block_count = kSymbols;
        enc.block_data_out = encoded[i].data();
        enc.block_stride_bytes = kBlockBytes;
        enc.bytes_out = lens[i].data();
        if (wirehair_encode_batch_async(pipeline, &enc) != Wirehair_Success) {
            wirehair_pipeline_free(pipeline);
            return false;
        }
    }

    int encodeDone = 0;
    int decodeSubmitted = 0;
    int decodeDone = 0;
    vector<uint8_t> seenEncode(kRequests, 0);
    vector<uint8_t> seenDecode(kRequests, 0);
    while (decodeDone < kRequests)
    {
        WirehairPipelineEvent event = {};
        if (wirehair_pipeline_poll(pipeline, 2000, &event) != Wirehair_Success) {
            wirehair_pipeline_free(pipeline);
            return false;
        }
        if (event.job_type == WirehairPipelineJob_EncodeBatch)
        {
            const int index = static_cast<int>(event.request_id - 1);
            if (index < 0 || index >= kRequests || seenEncode[index]) {
                wirehair_pipeline_free(pipeline);
                return false;
            }
            seenEncode[index] = 1;
            ++encodeDone;
            WirehairDecodeBatchRequest dec = {};
            dec.request_id = static_cast<uint64_t>(1000 + index);
            dec.message_bytes = kMessageBytes;
            dec.block_bytes = kBlockBytes;
            dec.block_ids = ids[index].data();
            dec.block_data = encoded[index].data();
            dec.block_data_bytes = lens[index].data();
            dec.symbol_count = kSymbols;
            dec.block_stride_bytes = kBlockBytes;
            dec.message_out = recovered[index].data();
            if (wirehair_decode_batch_async(pipeline, &dec) != Wirehair_Success) {
                wirehair_pipeline_free(pipeline);
                return false;
            }
            ++decodeSubmitted;
        }
        else if (event.job_type == WirehairPipelineJob_DecodeBatch)
        {
            const int index = static_cast<int>(event.request_id - 1000);
            if (index < 0 || index >= kRequests || seenDecode[index] || event.result != Wirehair_Success) {
                wirehair_pipeline_free(pipeline);
                return false;
            }
            seenDecode[index] = 1;
            ++decodeDone;
        }
    }
    wirehair_pipeline_free(pipeline);

    if (encodeDone != kRequests || decodeSubmitted != kRequests) {
        return false;
    }
    for (int i = 0; i < kRequests; ++i)
    {
        if (memcmp(messages[i].data(), recovered[i].data(), kMessageBytes) != 0) {
            return false;
        }
    }
    return true;
}

static bool Test_ThreadingPerfGates()
{
    const uint32_t kMessageBytes = 2 * 1024 * 1024 + 31;
    const uint32_t kBlockBytes = 2048;
    const uint32_t kBlocks = 1024;
    siamese::PCGRandom prng;
    prng.Seed(0x1abULL, 0x2cdULL);
    vector<uint8_t> message(kMessageBytes);
    FillMessage(message.data(), kMessageBytes, prng);
    vector<uint8_t> out(kBlocks * kBlockBytes, 0);
    vector<uint32_t> lens(kBlocks, 0);

    auto runPipeline = [&](uint32_t threads) -> double {
        WirehairPipelineConfig cfg = {threads, threads, 16};
        WirehairPipeline pipeline = wirehair_pipeline_create(&cfg);
        if (!pipeline) {
            return 0.0;
        }
        WirehairEncodeBatchRequest req = {};
        req.request_id = 7 + threads;
        req.message = message.data();
        req.message_bytes = kMessageBytes;
        req.block_bytes = kBlockBytes;
        req.start_block_id = 300;
        req.block_count = kBlocks;
        req.block_data_out = out.data();
        req.block_stride_bytes = kBlockBytes;
        req.bytes_out = lens.data();
        const auto t0 = chrono::high_resolution_clock::now();
        if (wirehair_encode_batch_async(pipeline, &req) != Wirehair_Success) {
            wirehair_pipeline_free(pipeline);
            return 0.0;
        }
        WirehairPipelineEvent event = {};
        while (wirehair_pipeline_poll(pipeline, 2000, &event) == Wirehair_NeedMore) {}
        const auto t1 = chrono::high_resolution_clock::now();
        wirehair_pipeline_free(pipeline);
        if (event.result != Wirehair_Success) {
            return 0.0;
        }
        const double usec = static_cast<double>(chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
        return usec > 0.0 ? (static_cast<double>(kBlocks * kBlockBytes) / usec) : 0.0;
    };

    auto runLegacy = [&]() -> double {
        WirehairCodec encoder = wirehair_encoder_create(nullptr, message.data(), kMessageBytes, kBlockBytes);
        if (!encoder) {
            return 0.0;
        }
        const auto t0 = chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < kBlocks; ++i)
        {
            uint32_t written = 0;
            if (wirehair_encode(encoder, 300 + i, out.data() + i * kBlockBytes, kBlockBytes, &written) != Wirehair_Success)
            {
                wirehair_free(encoder);
                return 0.0;
            }
        }
        const auto t1 = chrono::high_resolution_clock::now();
        wirehair_free(encoder);
        const double usec = static_cast<double>(chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
        return usec > 0.0 ? (static_cast<double>(kBlocks * kBlockBytes) / usec) : 0.0;
    };

    const double legacy1 = runLegacy();
    const double pipeline1 = runPipeline(1);
    const double pipeline8 = runPipeline(8);
    if (legacy1 <= 0.0 || pipeline1 <= 0.0 || pipeline8 <= 0.0) {
        return false;
    }

    // Gate 1: threaded path must not regress single-thread performance by more than 5%.
    if (pipeline1 < legacy1 * 0.95) {
        return false;
    }

    // Gate 2: 8-thread run should have at least 1.5x throughput over pipeline 1-thread.
    if (pipeline8 < pipeline1 * 1.5) {
        return false;
    }

    return true;
}

static bool Test_CudaConfigRoundTrip()
{
    WirehairCudaConfig cfg = {};
    wirehair_cuda_get_default_config(&cfg);
    cfg.backend_mode = WirehairCudaBackend_CpuOnly;
    cfg.stream_count = 2;
    cfg.use_pinned_memory = 0;
    if (wirehair_cuda_set_config(&cfg) != Wirehair_Success) {
        return false;
    }

    WirehairCudaConfig readBack = {};
    if (wirehair_cuda_get_config(&readBack) != Wirehair_Success) {
        return false;
    }
    return readBack.backend_mode == cfg.backend_mode &&
        readBack.stream_count == cfg.stream_count &&
        readBack.use_pinned_memory == cfg.use_pinned_memory;
}

static bool Test_CudaBatchParityWithCpuFallback()
{
    static const uint32_t kMessageBytes = 32768 + 29;
    static const uint32_t kBlockBytes = 512;
    static const uint32_t kStartBlockId = 128;
    static const uint32_t kBlockCount = 48;

    siamese::PCGRandom prng;
    prng.Seed(0x5312ULL, 0x9123ULL);

    vector<uint8_t> message(kMessageBytes);
    FillMessage(message.data(), kMessageBytes, prng);
    vector<uint8_t> cpuEncoded(kBlockCount * kBlockBytes, 0);
    vector<uint8_t> cudaEncoded(kBlockCount * kBlockBytes, 0);
    vector<uint32_t> cpuLens(kBlockCount, 0);
    vector<uint32_t> cudaLens(kBlockCount, 0);
    vector<uint8_t> cpuRecovered(kMessageBytes, 0);
    vector<uint8_t> cudaRecovered(kMessageBytes, 0);
    vector<uint32_t> ids(kBlockCount, 0);

    WirehairCudaConfig cfg = {};
    wirehair_cuda_get_default_config(&cfg);
    cfg.backend_mode = WirehairCudaBackend_CpuOnly;
    if (wirehair_cuda_set_config(&cfg) != Wirehair_Success) {
        return false;
    }

    WirehairCodec cpu = wirehair_encoder_create(nullptr, message.data(), kMessageBytes, kBlockBytes);
    if (!cpu) {
        return false;
    }
    for (uint32_t i = 0; i < kBlockCount; ++i)
    {
        const unsigned blockId = kStartBlockId + i;
        ids[i] = blockId;
        if (wirehair_encode(cpu, blockId, &cpuEncoded[i * kBlockBytes], kBlockBytes, &cpuLens[i]) != Wirehair_Success) {
            wirehair_free(cpu);
            return false;
        }
    }
    wirehair_free(cpu);

    cfg.backend_mode = WirehairCudaBackend_CudaPrefer;
    if (wirehair_cuda_set_config(&cfg) != Wirehair_Success) {
        return false;
    }

    WirehairEncodeBatchRequest encodeReq = {};
    encodeReq.request_id = 1;
    encodeReq.message = message.data();
    encodeReq.message_bytes = kMessageBytes;
    encodeReq.block_bytes = kBlockBytes;
    encodeReq.start_block_id = kStartBlockId;
    encodeReq.block_count = kBlockCount;
    encodeReq.block_data_out = cudaEncoded.data();
    encodeReq.block_stride_bytes = kBlockBytes;
    encodeReq.bytes_out = cudaLens.data();

    if (wirehair_cuda_encode_batch(&encodeReq) != Wirehair_Success) {
        return false;
    }
    if (cudaLens != cpuLens) {
        return false;
    }
    if (memcmp(cudaEncoded.data(), cpuEncoded.data(), cpuEncoded.size()) != 0) {
        return false;
    }

    WirehairDecodeBatchRequest decodeReq = {};
    decodeReq.request_id = 2;
    decodeReq.message_bytes = kMessageBytes;
    decodeReq.block_bytes = kBlockBytes;
    decodeReq.block_ids = ids.data();
    decodeReq.block_data = cudaEncoded.data();
    decodeReq.block_data_bytes = cudaLens.data();
    decodeReq.symbol_count = kBlockCount;
    decodeReq.block_stride_bytes = kBlockBytes;
    decodeReq.message_out = cudaRecovered.data();
    if (wirehair_cuda_decode_batch(&decodeReq) != Wirehair_Success) {
        return false;
    }

    WirehairCodec cpuDecoder = wirehair_decoder_create(nullptr, kMessageBytes, kBlockBytes);
    if (!cpuDecoder) {
        return false;
    }
    for (uint32_t i = 0; i < kBlockCount; ++i)
    {
        if (wirehair_decode(cpuDecoder, ids[i], &cpuEncoded[i * kBlockBytes], cpuLens[i]) == Wirehair_Success) {
            break;
        }
    }
    const WirehairResult recover = wirehair_recover(cpuDecoder, cpuRecovered.data(), kMessageBytes);
    wirehair_free(cpuDecoder);
    if (recover != Wirehair_Success) {
        return false;
    }
    return memcmp(cpuRecovered.data(), cudaRecovered.data(), kMessageBytes) == 0;
}

static bool Test_CudaPerfStatsTelemetry()
{
    WirehairCudaPerfStats stats = {};
    stats.struct_bytes = sizeof(WirehairCudaPerfStats);
    wirehair_cuda_reset_perf_stats();
    const WirehairResult before = wirehair_cuda_get_perf_stats(&stats);
    if (before == Wirehair_UnsupportedPlatform) {
        return true;
    }
    if (before != Wirehair_Success) {
        return false;
    }

    static const uint32_t kMessageBytes = 65536 + 13;
    static const uint32_t kBlockBytes = 1024;
    static const uint32_t kBlockCount = 96;
    static const uint32_t kStartBlockId = 200;

    siamese::PCGRandom prng;
    prng.Seed(0x7711ULL, 0x8122ULL);
    vector<uint8_t> message(kMessageBytes);
    FillMessage(message.data(), kMessageBytes, prng);
    vector<uint8_t> encoded(kBlockCount * kBlockBytes, 0);
    vector<uint32_t> lens(kBlockCount, 0);
    vector<uint32_t> ids(kBlockCount, 0);
    vector<uint8_t> recovered(kMessageBytes, 0);
    for (uint32_t i = 0; i < kBlockCount; ++i) {
        ids[i] = kStartBlockId + i;
    }

    WirehairCudaConfig cfg = {};
    wirehair_cuda_get_default_config(&cfg);
    cfg.backend_mode = WirehairCudaBackend_CudaPrefer;
    cfg.stream_count = 2;
    cfg.use_pinned_memory = 1;
    if (wirehair_cuda_set_config(&cfg) != Wirehair_Success) {
        return false;
    }

    WirehairEncodeBatchRequest encodeReq = {};
    encodeReq.request_id = 11;
    encodeReq.message = message.data();
    encodeReq.message_bytes = kMessageBytes;
    encodeReq.block_bytes = kBlockBytes;
    encodeReq.start_block_id = kStartBlockId;
    encodeReq.block_count = kBlockCount;
    encodeReq.block_data_out = encoded.data();
    encodeReq.block_stride_bytes = kBlockBytes;
    encodeReq.bytes_out = lens.data();
    if (wirehair_cuda_encode_batch(&encodeReq) != Wirehair_Success) {
        return false;
    }

    WirehairDecodeBatchRequest decodeReq = {};
    decodeReq.request_id = 12;
    decodeReq.message_bytes = kMessageBytes;
    decodeReq.block_bytes = kBlockBytes;
    decodeReq.block_ids = ids.data();
    decodeReq.block_data = encoded.data();
    decodeReq.block_data_bytes = lens.data();
    decodeReq.symbol_count = kBlockCount;
    decodeReq.block_stride_bytes = kBlockBytes;
    decodeReq.message_out = recovered.data();
    if (wirehair_cuda_decode_batch(&decodeReq) != Wirehair_Success) {
        return false;
    }

    WirehairCudaPerfStats afterStats = {};
    afterStats.struct_bytes = sizeof(WirehairCudaPerfStats);
    const WirehairResult after = wirehair_cuda_get_perf_stats(&afterStats);
    if (after == Wirehair_UnsupportedPlatform) {
        return true;
    }
    if (after != Wirehair_Success) {
        return false;
    }
    if (afterStats.encode_calls == 0 || afterStats.decode_calls == 0) {
        return false;
    }
    if (afterStats.h2d_us == 0 || afterStats.kernel_us == 0) {
        return false;
    }
    return true;
}

#pragma warning(disable: 4505)

static bool Test_EncoderProducesOriginals(
    WirehairCodec encoder,
    unsigned N,
    uint8_t* block,
    uint8_t* message,
    unsigned blockBytes,
    unsigned finalBytes)
{
    for (unsigned originalBlockId = 0; originalBlockId < N; ++originalBlockId)
    {
        if (originalBlockId == N - 1) {
            memset(block, kPadChar, blockBytes);
        }

        uint32_t bytesOut = 0;
        WirehairResult encodeResult = wirehair_encode(encoder, originalBlockId, &block[0], blockBytes, &bytesOut);
        if (encodeResult != Wirehair_Success)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "!!! Encode original failed for N = " << N << ", originalBlockId = " << originalBlockId << endl;
            return false;
        }

        if (originalBlockId == N - 1)
        {
            if (bytesOut != finalBytes)
            {
                SIAMESE_DEBUG_BREAK();
                cout << "!!! Encode original wrong length for N = " << N
                    << ", originalBlockId = " << originalBlockId
                    << ", bytesOut = " << bytesOut << ", finalBytes = " << finalBytes << endl;
                return false;
            }
            if (0 != memcmp(&message[0] + originalBlockId * blockBytes, &block[0], bytesOut))
            {
                SIAMESE_DEBUG_BREAK();
                cout << "!!! Encode original wrong data for N = " << N
                    << ", originalBlockId = " << originalBlockId
                    << ", bytesOut = " << bytesOut << ", finalBytes = " << finalBytes << endl;
                return false;
            }
            for (unsigned i = finalBytes; i < blockBytes; ++i)
            {
                if (block[i] != kPadChar)
                {
                    SIAMESE_DEBUG_BREAK();
                    cout << "!!! Encode original buffer overrun for N = " << N
                        << ", originalBlockId = " << originalBlockId
                        << ", bytesOut = " << bytesOut << ", finalBytes = " << finalBytes << endl;
                    return false;
                }
            }
        }
        else
        {
            if (bytesOut != blockBytes)
            {
                SIAMESE_DEBUG_BREAK();
                cout << "!!! Encode original wrong length for N = " << N
                    << ", originalBlockId = " << originalBlockId
                    << ", bytesOut = " << bytesOut << ", blockBytes = " << blockBytes << endl;
                return false;
            }
            if (0 != memcmp(&message[0] + originalBlockId * blockBytes, &block[0], bytesOut))
            {
                SIAMESE_DEBUG_BREAK();
                cout << "!!! Encode original wrong data for N = " << N
                    << ", originalBlockId = " << originalBlockId
                    << ", bytesOut = " << bytesOut << ", finalBytes = " << finalBytes << endl;
                return false;
            }
        }
    }

    return true;
}

static bool DoesCodecProduceOriginals(
    WirehairCodec decoder,
    unsigned N,
    unsigned blockBytes,
    unsigned finalBytes,
    const uint8_t* message,
    uint8_t* decodedMessage,
    unsigned messageBytes)
{
    // Check buffer overflow
    decodedMessage[messageBytes] = kPadChar;

    WirehairResult recoverResult = wirehair_recover(decoder, decodedMessage, messageBytes);

    if (recoverResult != Wirehair_Success)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "Failed to recover with original pieces for N = " << N << endl;
        return false;
    }

    if (0 != memcmp(message, decodedMessage, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "Failed to recover the data with original pieces for N = " << N << endl;
        return false;
    }

    if (decodedMessage[messageBytes] != kPadChar)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "Failed to recover buffer overrun with original pieces for N = " << N << endl;
        return false;
    }

    for (unsigned i = 0; i < N; ++i)
    {
        decodedMessage[finalBytes] = kPadChar;

        uint32_t recoveredBytes = 0;
        WirehairResult blockResult = wirehair_recover_block(decoder, i, decodedMessage, &recoveredBytes);

        if (blockResult != Wirehair_Success)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "Failed to wirehair_recover_block with original pieces for N = " << N << ", i = " << i << endl;
            return false;
        }

        if ((i == N - 1) && decodedMessage[finalBytes] != kPadChar)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "Failed to wirehair_recover_block memory corruption with original pieces for N = " << N << endl;
            return false;
        }

        const unsigned expectedBytes = (i == N - 1) ? finalBytes : blockBytes;

        if (0 != memcmp(message + i * blockBytes, decodedMessage, expectedBytes))
        {
            SIAMESE_DEBUG_BREAK();
            cout << "Failed to wirehair_recover_block the data with original pieces for N = " << N << endl;
            return false;
        }
    }

    return true;
}

static bool TestAllRecoveryData(
    WirehairCodec encoder,
    WirehairCodec decoder,
    unsigned N,
    unsigned blockBytes,
    unsigned finalBytes,
    const uint8_t* message,
    uint8_t* decodedMessage,
    unsigned messageBytes)
{
    unsigned needed = 0;

#ifdef ENABLE_ALL_RECOVERY_FINAL_BLOCK
    // Check the special case final block:
    {
        uint32_t Nm1Len = 0;
        WirehairResult encodeResult = wirehair_encode(encoder, N - 1, decodedMessage, blockBytes, &Nm1Len);

        if (encodeResult != Wirehair_Success)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_encode failed for N-1 and N = " << N << endl;
            return false;
        }

        if (Nm1Len != finalBytes)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_encode failed wrong len for N-1 and N = " << N << endl;
            return false;
        }

        WirehairResult decodeResult = wirehair_decode(decoder, N - 1, decodedMessage, Nm1Len);

        if (decodeResult != Wirehair_NeedMore)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_decode failed for N-1 and N = " << N << endl;
            return false;
        }

        ++needed;
    }
#endif

    for (unsigned blockId = N;; ++blockId)
    {
        ++needed;

        uint32_t writeLen = 0;
        WirehairResult encodeResult = wirehair_encode(encoder, blockId, decodedMessage, blockBytes, &writeLen);

        if (encodeResult != Wirehair_Success)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_encode failed for N = " << N << endl;
            return false;
        }

        if (writeLen != blockBytes)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_encode failed wrong len for N-1 and N = " << N << endl;
            return false;
        }

        WirehairResult decodeResult = wirehair_decode(decoder, blockId, decodedMessage, writeLen);

        if (decodeResult != Wirehair_NeedMore)
        {
            if (decodeResult == Wirehair_Success) {
                break;
            }

            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_decode failed for " << blockId << " and N = " << N << endl;
            return false;
        }
    }

    if (needed >= N + kExtraWarnThreshold) {
        //SIAMESE_DEBUG_BREAK();
        cout << "TestAllRecoveryData: Too much overhead: " << (needed - N) << " extra for N=" << N << " Seed=n/a BlockBytes=" << blockBytes << endl;
        //return false;
    }

    if (!DoesCodecProduceOriginals(decoder, N, blockBytes, finalBytes, message, decodedMessage, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!! Decoder2 failed to produce originals" << endl;
        return false;
    }

    return true;
}

// Verify that decoder can recover when all input data is available
// ..and that it can turn into an encoder that reproduces all the original data
// ..and that a decoder fed with recovery data produced by the re-encoder will decode
static bool Test_DecodeAllOriginal(
    unsigned N,
    unsigned blockBytes,
    unsigned finalBytes,
    const uint8_t* message,
    uint8_t* decodedMessage,
    unsigned messageBytes)
{
    WirehairCodec decoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
    if (!decoder)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Failed to create decoder for N = " << N << ", blockBytes = " << blockBytes << endl;
        return false;
    }

    // Feed it all original data (out of order)

    for (int originalBlockId = N - 1; originalBlockId >= 0; --originalBlockId)
    {
        const uint8_t* originalBlock = &message[0] + originalBlockId * blockBytes;
        const unsigned originalBlockBytes = ((unsigned)originalBlockId == N - 1) ? finalBytes : blockBytes;

        WirehairResult decodeResult = wirehair_decode(decoder, originalBlockId, originalBlock, originalBlockBytes);

        // If this is the last one:
        if (originalBlockId == 0)
        {
            if (decodeResult != Wirehair_Success)
            {
                SIAMESE_DEBUG_BREAK();
                cout << "Failed to decode with original pieces for N = " << N << ", originalBlockId = " << originalBlockId << endl;
                return false;
            }
        }
        else if (decodeResult != Wirehair_NeedMore)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "Failed to decode midway with original pieces for N = " << N << ", originalBlockId = " << originalBlockId << endl;
            return false;
        }
    }

    if (!DoesCodecProduceOriginals(decoder, N, blockBytes, finalBytes, message, decodedMessage, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!! Decoder failed to produce originals" << endl;
        return false;
    }

    // ..and that it can turn into an encoder that reproduces all the original data
    WirehairResult becomeResult = wirehair_decoder_becomes_encoder(decoder);

    if (becomeResult != Wirehair_Success)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "Failed to wirehair_decoder_becomes_encoder with original pieces for N = " << N << endl;
        return false;
    }

    // ..and that a decoder fed with recovery data produced by the re-encoder will decode

    WirehairCodec decoder2 = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
    if (!decoder2)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Failed to create decoder2 for N = " << N << ", blockBytes = " << blockBytes << endl;
        return false;
    }

    if (!TestAllRecoveryData(decoder, decoder2, N, blockBytes, finalBytes, message, decodedMessage, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "TestAllRecoveryData failed for N = " << N << endl;
        return false;
    }

    wirehair_free(decoder2);
    wirehair_free(decoder);

    return true;
}

// Verify that decoder can recover when no input data is available
// ..and that it can turn into an encoder that reproduces all the original data
// ..and that a decoder fed with recovery data produced by the re-encoder will decode
static bool Test_DecodeAllRecovery(
    unsigned N,
    unsigned blockBytes,
    unsigned finalBytes,
    const uint8_t* message,
    uint8_t* decodedMessage,
    unsigned messageBytes)
{
    WirehairCodec encoder = wirehair_encoder_create(nullptr, message, messageBytes, blockBytes);
    if (!encoder)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Failed to create encoder for N = " << N << ", blockBytes = " << blockBytes << endl;
        return false;
    }

    WirehairCodec decoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
    if (!decoder)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Failed to create decoder for N = " << N << ", blockBytes = " << blockBytes << endl;
        return false;
    }

    if (!TestAllRecoveryData(encoder, decoder, N, blockBytes, finalBytes, message, decodedMessage, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "TestAllRecoveryData failed for N = " << N << endl;
        return false;
    }

    // ..and that it can turn into an encoder that reproduces all the original data
    WirehairResult becomeResult = wirehair_decoder_becomes_encoder(decoder);

    if (becomeResult != Wirehair_Success)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "Failed to wirehair_decoder_becomes_encoder with original pieces for N = " << N << endl;
        return false;
    }

    if (!DoesCodecProduceOriginals(decoder, N, blockBytes, finalBytes, message, decodedMessage, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!! Converted Decoder failed to produce originals" << endl;
        return false;
    }

    wirehair_free(decoder);
    wirehair_free(encoder);

    return true;
}

// Verify that decoder can recover when no input data is available
// Do not try to do decoder_becomes_encoder
static bool Test_DecodeRandomLosses(
    siamese::PCGRandom& prng,
    uint64_t seed,
    unsigned N,
    unsigned blockBytes,
    unsigned finalBytes,
    const uint8_t* message,
    uint8_t* decodedMessage,
    unsigned messageBytes)
{
    WirehairCodec encoder = wirehair_encoder_create(nullptr, message, messageBytes, blockBytes);
    if (!encoder)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Failed to create encoder for N = " << N << ", blockBytes = " << blockBytes << endl;
        return false;
    }

    WirehairCodec decoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
    if (!decoder)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Failed to create decoder for N = " << N << ", blockBytes = " << blockBytes << endl;
        return false;
    }

    unsigned blockId = 0;
    unsigned needed = 0;
    for (;;)
    {
        blockId++;
        if (prng.Next() % 100 < 10) {
            continue;
        }

        ++needed;

        uint32_t writeLen = 0;
        WirehairResult encodeResult = wirehair_encode(encoder, blockId, decodedMessage, blockBytes, &writeLen);

        if (encodeResult != Wirehair_Success)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_encode failed for N = " << N << endl;
            return false;
        }

        if (writeLen != blockBytes)
        {
            if (blockId != N - 1 || writeLen != finalBytes)
            {
                SIAMESE_DEBUG_BREAK();
                cout << "wirehair_encode failed wrong len for " << blockId << " and N = " << N << endl;
                return false;
            }
        }

        WirehairResult decodeResult = wirehair_decode(decoder, blockId, decodedMessage, writeLen);

        if (decodeResult != Wirehair_NeedMore)
        {
            if (decodeResult == Wirehair_Success) {
                break;
            }

            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_decode failed for " << blockId << " and N = " << N << endl;
            return false;
        }
    }

    if (needed >= N + kExtraWarnThreshold) {
        //SIAMESE_DEBUG_BREAK();
        cout << "Test_DecodeRandomLosses: Too much overhead: " << (needed - N) << " extra for N=" << N << " Seed=" << seed <<" BlockBytes=" << blockBytes << endl;
        //return false;
    }

    if (!DoesCodecProduceOriginals(decoder, N, blockBytes, finalBytes, message, decodedMessage, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!! Decoder2 failed to produce originals" << endl;
        return false;
    }

    wirehair_free(decoder);
    wirehair_free(encoder);

    return true;
}

static atomic<bool> TestFailed(false);

static void TestN(uint64_t seed, int N, unsigned blockBytes)
{
    siamese::PCGRandom prng;
    prng.Seed(seed + N, blockBytes);

    vector<uint8_t> message(N * blockBytes + 1);
    vector<uint8_t> block(blockBytes);
    vector<uint8_t> decodedMessageVector(N * blockBytes + 1);

    // Final block is partial
#ifdef ENABLE_PARTIAL_FINAL
    const unsigned finalBytes = blockBytes == 1 ? 1 : (blockBytes - 1);
#else
    const unsigned finalBytes = blockBytes;
#endif
    const unsigned messageBytes = blockBytes * (N - 1) + finalBytes;

    uint8_t* decodedMessagePtr = &decodedMessageVector[0];

    // Set up a random message
    FillMessage(&message[0], messageBytes, prng);

    WirehairCodec encoder = wirehair_encoder_create(nullptr, &message[0], messageBytes, blockBytes);
    if (!encoder)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Failed to create encoder for N = " << N << ", blockBytes = " << blockBytes << endl;
        TestFailed = true;
        return;
    }

    // Verify that encoder produces original blocks

    if (!Test_EncoderProducesOriginals(encoder, N, &block[0], &message[0], blockBytes, finalBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Test_EncoderProducesOriginals failed" << endl;
        TestFailed = true;
        return;
    }

    // Decoder checks:

    if (!Test_DecodeAllOriginal(N, blockBytes, finalBytes, &message[0], decodedMessagePtr, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Test_DecodeAllOriginal failed" << endl;
        TestFailed = true;
        return;
    }

    if (!Test_DecodeAllRecovery(N, blockBytes, finalBytes, &message[0], decodedMessagePtr, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Test_DecodeAllRecovery failed" << endl;
        TestFailed = true;
        return;
    }

    if (!Test_DecodeRandomLosses(prng, seed, N, blockBytes, finalBytes, &message[0], decodedMessagePtr, messageBytes))
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Test_DecodeAllRecovery failed" << endl;
        TestFailed = true;
        return;
    }

    wirehair_free(encoder);
}


static bool ReadmeExample()
{
    // Size of packets to produce
    static const int kPacketSize = 1400;

    // Note: Does not need to be an even multiple of packet size or 16 etc
    static const int kMessageBytes = 1000 * 1000 + 333;

    vector<uint8_t> message(kMessageBytes);

    // Fill message contents
    memset(&message[0], 1, message.size());

    // Create encoder
    WirehairCodec encoder = wirehair_encoder_create(nullptr, &message[0], kMessageBytes, kPacketSize);
    if (!encoder)
    {
        cout << "!!! Failed to create encoder" << endl;
        return false;
    }

    // Create decoder
    WirehairCodec decoder = wirehair_decoder_create(nullptr, kMessageBytes, kPacketSize);
    if (!decoder)
    {
        // Free memory for encoder
        wirehair_free(encoder);

        cout << "!!! Failed to create decoder" << endl;
        return false;
    }

    unsigned blockId = 0, needed = 0;

    for (;;)
    {
        // Select which block to encode.
        // Note: First N blocks are the original data, so it's possible to start
        // sending data while wirehair_encoder_create() is getting started.
        blockId++;

        // Simulate 10% packetloss
        if (blockId % 10 == 0) {
            continue;
        }

        // Keep track of how many pieces were needed
        ++needed;

        vector<uint8_t> block(kPacketSize);

        // Encode a packet
        uint32_t writeLen = 0;
        WirehairResult encodeResult = wirehair_encode(
            encoder, // Encoder object
            blockId, // ID of block to generate
            &block[0], // Output buffer
            kPacketSize, // Output buffer size
            &writeLen); // Returned block length

        if (encodeResult != Wirehair_Success)
        {
            cout << "wirehair_encode failed: " << encodeResult << endl;
            return false;
        }

        // Attempt decode
        WirehairResult decodeResult = wirehair_decode(
            decoder, // Decoder object
            blockId, // ID of block that was encoded
            &block[0], // Input block
            writeLen); // Block length

        // If decoder returns success:
        if (decodeResult == Wirehair_Success) {
            // Decoder has enough data to recover now
            break;
        }

        if (decodeResult != Wirehair_NeedMore)
        {
            cout << "wirehair_decode failed: " << decodeResult << endl;
            return false;
        }
    }

    vector<uint8_t> decoded(kMessageBytes);

    // Recover original data on decoder side
    WirehairResult decodeResult = wirehair_recover(
        decoder,
        &decoded[0],
        kMessageBytes);

    if (decodeResult != Wirehair_Success)
    {
        cout << "wirehair_recover failed: " << decodeResult << endl;
        return false;
    }

    // Free memory for encoder and decoder
    wirehair_free(encoder);
    wirehair_free(decoder);

    return true;
}

static bool Benchmark(unsigned N, unsigned packetBytes, unsigned trials)
{
    siamese::PCGRandom prng;
    prng.Seed(N, packetBytes);

    const unsigned kBlockBytes = packetBytes;
    const unsigned kMessageBytes = N * kBlockBytes;

    vector<uint8_t> message(kMessageBytes);
    vector<uint8_t> block(kBlockBytes);
    vector<uint8_t> decoded(kMessageBytes);

    memset(&message[0], 6, kMessageBytes);

    uint64_t encode_create_sum = 0;
    uint64_t runs = 0;
    uint64_t extra_sum = 0;
    uint64_t encode_sum = 0;
    uint64_t decode_sum = 0;
    uint64_t packets = 0;
    uint64_t recover_sum = 0;

    for (unsigned trial = 0; trial < trials; ++trial)
    {
        ++runs;

        uint64_t t0 = siamese::GetTimeUsec();

        WirehairCodec encoder = wirehair_encoder_create(nullptr, &message[0], kMessageBytes, kBlockBytes);
        if (!encoder)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "!!! Failed to create encoder" << endl;
            return false;
        }

        uint64_t t1 = siamese::GetTimeUsec();

        WirehairCodec decoder = wirehair_decoder_create(nullptr, kMessageBytes, kBlockBytes);
        if (!decoder)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "!!! Failed to create decoder" << endl;
            return false;
        }

        encode_create_sum += t1 - t0;

        unsigned blockId = 0;
        unsigned needed = 0;
        for (;;)
        {
            blockId++;

            // Introduce about 30% loss to the data
            if (prng.Next() % 100 < 30) {
                continue;
            }

            ++needed;

            uint64_t t3 = siamese::GetTimeUsec();

            uint32_t writeLen = 0;
            WirehairResult encodeResult = wirehair_encode(
                encoder,
                blockId,
                &block[0],
                kBlockBytes,
                &writeLen);
            if (encodeResult != Wirehair_Success)
            {
                SIAMESE_DEBUG_BREAK();
                cout << "wirehair_encode failed" << endl;
                return false;
            }

            uint64_t t4 = siamese::GetTimeUsec();

            WirehairResult decodeResult = wirehair_decode(decoder, blockId, &block[0], writeLen);

            uint64_t t5 = siamese::GetTimeUsec();

            encode_sum += t4 - t3;
            decode_sum += t5 - t4;
            ++packets;

            if (decodeResult != Wirehair_NeedMore)
            {
                if (decodeResult == Wirehair_Success) {
                    break;
                }

                SIAMESE_DEBUG_BREAK();
                cout << "wirehair_decode failed for " << blockId << " and N = " << N << endl;
                return false;
            }
        }

        SIAMESE_DEBUG_ASSERT(needed >= N);
        extra_sum += needed - N;

        uint64_t t7 = siamese::GetTimeUsec();

        WirehairResult recoverResult = wirehair_recover(decoder, &decoded[0], kMessageBytes);

        uint64_t t8 = siamese::GetTimeUsec();

        if (recoverResult != Wirehair_Success)
        {
            SIAMESE_DEBUG_BREAK();
            cout << "wirehair_recover failed" << endl;
            return false;
        }

        recover_sum += t8 - t7;

        wirehair_free(decoder);
        wirehair_free(encoder);
    }

    cout << "For N = " << N << " packets of " << packetBytes << " bytes:" << endl;

    uint64_t avg_encode_create_usec = encode_create_sum / runs;

    float encode_create_MBPS = 0.f;
    if (avg_encode_create_usec > 0) {
        encode_create_MBPS = kMessageBytes / (float)avg_encode_create_usec;
    }

    cout << "+ Average wirehair_encoder_create() time: " << avg_encode_create_usec << " usec (" << encode_create_MBPS << " MBPS)" << endl;
    // wirehair_decoder_create() is super fast.  So is the decoder_becomes_encoder() most of the time

    uint64_t avg_encode_usec = encode_sum / packets;
    uint64_t avg_decode_usec = decode_sum / packets;

    float encode_MBPS = 0.f, decode_MBPS = 0.f;
    if (encode_sum > 0) {
        encode_MBPS = (packets * kBlockBytes) / (float)encode_sum;
    }
    if (decode_sum > 0) {
        decode_MBPS = (packets * kBlockBytes) / (float)decode_sum;
    }

    cout << "+ Average wirehair_encode() time: " << avg_encode_usec << " usec (" << encode_MBPS << " MBPS)" << endl;
    cout << "+ Average wirehair_decode() time: " << avg_decode_usec << " usec (" << decode_MBPS << " MBPS)" << endl;

    float avg_overhead_pieces = extra_sum / (float)runs;

    cout << "+ Average overhead piece count beyond N = " << avg_overhead_pieces << endl;

    uint64_t avg_recover_usec = recover_sum / runs;

    float recover_MBPS = 0.f;
    if (recover_sum > 0) {
        recover_MBPS = (runs * kMessageBytes) / (float)recover_sum;
    }

    cout << "+ Average wirehair_recover() time: " << avg_recover_usec << " usec (" << recover_MBPS << " MBPS)" << endl;

    return true;
}

static const unsigned kBenchmarkNList[] = {
    12,
    32,
    102,
    134,
    169,
    201,
    294,
    359,
    413,
    770,
    1000
};

int main()
{
    const WirehairResult initResult = wirehair_init();

    if (initResult != Wirehair_Success)
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Wirehair initialization failed: " << initResult << endl;
        return -1;
    }

    if (!Test_ConcurrentInitAndCodecCreate())
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Concurrent init/create test failed" << endl;
        return -5;
    }

    if (!Test_AsyncBatchDeterminism())
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Async batch determinism test failed" << endl;
        return -6;
    }

    if (!Test_AsyncPipelineStress())
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Async pipeline stress test failed" << endl;
        return -7;
    }

    if (!Test_ThreadingPerfGates())
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Threading performance gates failed" << endl;
        return -8;
    }

    if (!Test_CudaConfigRoundTrip())
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! CUDA config round-trip test failed" << endl;
        return -9;
    }

    if (!Test_CudaBatchParityWithCpuFallback())
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! CUDA batch parity test failed" << endl;
        return -10;
    }

    if (!Test_CudaPerfStatsTelemetry())
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! CUDA perf stats telemetry test failed" << endl;
        return -11;
    }

    if (!ReadmeExample())
    {
        SIAMESE_DEBUG_BREAK();
        cout << "!!! Example usage failed" << endl;
        return -2;
    }

#ifdef BENCHMARK_SHORT_LIST
    for (unsigned i = 0; i < sizeof(kBenchmarkNList) / sizeof(kBenchmarkNList[0]); ++i)
    {
        const unsigned N = kBenchmarkNList[i];
#else
    for (unsigned N = 2; N < 64000; N *= 2)
    {
#endif
        const unsigned kPacketBytes = 1300;
        const unsigned kBenchTrials = 2000;

        if (!Benchmark(N, kPacketBytes, kBenchTrials))
        {
            SIAMESE_DEBUG_BREAK();
            cout << "!!! Benchmark failed" << endl;
            return -3;
        }
    }

    cout << "Wirehair Unit Test" << endl;

    uint64_t seed = siamese::GetTimeUsec();

    cout << "Start seed = " << seed << endl;

    for (unsigned N = kMinN; N <= kMaxN; ++N)
    {
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
        for (int blockBytes = 1; blockBytes <= kMaxBlockBytes; ++blockBytes) {
            TestN(seed, N, blockBytes);
        }

        if (TestFailed) {
            cout << "A test failed for N = " << N << endl;
            return -4;
        }

        //cout << "Test passed for N = " << N << endl;
    }

    return 0;
}
