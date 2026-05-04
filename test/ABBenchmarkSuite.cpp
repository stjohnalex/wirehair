#include <wirehair/wirehair.h>
#include <wirehair/wirehair_cuda.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#if defined(_MSC_VER)
#include <ppl.h>
#endif
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifndef AB_VARIANT_NAME
#define AB_VARIANT_NAME "unknown"
#endif

#ifndef AB_IS_OPTIMIZED
#define AB_IS_OPTIMIZED 0
#endif

namespace {

struct Config
{
    uint32_t Trials = 12;
    uint64_t Seed = 0x1234abcd9876ULL;
    std::string OutputPath = "test-output/ab-bench/ab-benchmark.json";
    std::string IoRoot = "test-output/ab-bench/io";
    bool Quiet = false;
    bool Stress = false;
    uint32_t CudaBackendMode = WirehairCudaBackend_CudaPrefer;
    bool CoreUseCudaBatch = false;
    bool CudaHighMemory = true;
};

class XorShift64
{
public:
    explicit XorShift64(uint64_t seed) : State(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    uint64_t Next()
    {
        uint64_t x = State;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        State = x;
        return x;
    }
    uint32_t Next32() { return static_cast<uint32_t>(Next() >> 32); }
    uint32_t NextBounded(uint32_t bound)
    {
        if (bound == 0) {
            return 0;
        }
        return static_cast<uint32_t>(Next() % bound);
    }

private:
    uint64_t State;
};

struct CoreCaseResult
{
    uint32_t N = 0;
    uint32_t BlockBytes = 0;
    uint32_t LossPercent = 0;
    uint32_t Trials = 0;
    uint32_t Successes = 0;
    double AvgExtra = 0.0;
    double EncodeMBps = 0.0;
    double DecodeMBps = 0.0;
    double RecoverMBps = 0.0;
    double AvgCreateUs = 0.0;
    double HostPackMB = 0.0;
    double HostPackUs = 0.0;
    double CudaPreprocessUs = 0.0;
    double CudaDecodeFeedUs = 0.0;
    double CudaRecoverOnlyUs = 0.0;
    uint32_t DifferentialChecks = 0;
    uint32_t DifferentialMismatches = 0;
};

struct ParityCaseResult
{
    uint32_t DriveCount = 0;
    uint32_t Trials = 0;
    uint32_t Successes = 0;
    double AvgNeeded = 0.0;
};

struct ChurnCaseResult
{
    uint32_t TargetCount = 0;
    uint32_t Trials = 0;
    uint32_t Successes = 0;
    double AvgNeeded = 0.0;
};

struct StorageCaseResult
{
    uint32_t Trials = 0;
    uint32_t Successes = 0;
    double WriteMBps = 0.0;
    double ReadMBps = 0.0;
    double AvgNeeded = 0.0;
    uint64_t WriteBatchCount = 0;
    uint64_t ReadBatchCount = 0;
    double AvgWriteBatchBytes = 0.0;
    double AvgReadBatchBytes = 0.0;
    double RandomWriteP50Ms = 0.0;
    double RandomWriteP95Ms = 0.0;
    double RandomReadP50Ms = 0.0;
    double RandomReadP95Ms = 0.0;
    double EffectiveReadBytesBeforeSuccess = 0.0;
    double WorkCompletionRatio = 0.0;
    std::string ModeProfile = "balanced";
};

struct StorageProfilesResult
{
    StorageCaseResult Isolated;
    StorageCaseResult E2eCuda;
};

struct ThreadScalingResult
{
    uint32_t Threads = 1;
    uint32_t Trials = 0;
    double EncodeMBps = 0.0;
    double DecodeMBps = 0.0;
    double P50LatencyMs = 0.0;
    double P95LatencyMs = 0.0;
    double Speedup = 0.0;
    double Efficiency = 0.0;
};

struct CudaPerfStatsResult
{
    bool Available = false;
    uint64_t EncodeCalls = 0;
    uint64_t DecodeCalls = 0;
    uint64_t SetupUs = 0;
    uint64_t H2dUs = 0;
    uint64_t KernelUs = 0;
    uint64_t D2hUs = 0;
    uint64_t SyncUs = 0;
    uint64_t H2dEventUs = 0;
    uint64_t KernelEventUs = 0;
    uint64_t D2hEventUs = 0;
    uint64_t E2eEventUs = 0;
    uint64_t EnqueueUs = 0;
    uint64_t QueueStallUs = 0;
    uint64_t QueueDepthSamples = 0;
    uint64_t QueueDepthTotal = 0;
    uint64_t DeviceIdleUs = 0;
    uint64_t SubmitBatches = 0;
    uint64_t SubmitItems = 0;
    uint64_t ProducerWaitUs = 0;
    uint64_t TransferWaitUs = 0;
    uint64_t ComputeWaitUs = 0;
    uint64_t CompletionWaitUs = 0;
    uint64_t BytesH2d = 0;
    uint64_t BytesD2h = 0;
    uint64_t CoreOffloadCalls = 0;
    double KernelSharePct = 0.0;
    double TransferSyncSharePct = 0.0;
    double AvgBytesPerCall = 0.0;
    double AvgQueueDepth = 0.0;
    double QueueStallPct = 0.0;
    double DeviceIdlePct = 0.0;
    double SymbolsPerSubmit = 0.0;
    double OverlapRatio = 0.0;
    uint64_t SolverStageUs = 0;
    uint64_t SolverPivotUs = 0;
    uint64_t SolverEliminateUs = 0;
    uint64_t SolverBackSubUs = 0;
    uint64_t SolverVerifyPasses = 0;
    uint64_t SolverVerifyFailures = 0;
    double SolverKernelSharePct = 0.0;
};

static const char* CudaBackendName(uint32_t mode)
{
    switch (mode)
    {
    case WirehairCudaBackend_Auto:
        return "auto";
    case WirehairCudaBackend_CpuOnly:
        return "cpuonly";
    case WirehairCudaBackend_CudaPrefer:
        return "cudaprefer";
    case WirehairCudaBackend_CudaOnly:
        return "cudaonly";
    default:
        return "unknown";
    }
}

static uint64_t NowUs()
{
    const auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
}

static std::string JoinPath(const std::string& a, const std::string& b)
{
    if (a.empty()) {
        return b;
    }
    if (a[a.size() - 1] == '/' || a[a.size() - 1] == '\\') {
        return a + b;
    }
    return a + "/" + b;
}

static bool EnsureDirectory(const std::string& inputPath)
{
    if (inputPath.empty()) {
        return false;
    }
    std::string path = inputPath;
    std::replace(path.begin(), path.end(), '\\', '/');

    std::string current;
    if (path.size() >= 2 && path[1] == ':') {
        current = path.substr(0, 2);
    }

    size_t start = current.empty() ? 0 : 2;
    if (path.size() > start && path[start] == '/') {
        current += "/";
        ++start;
    }

    for (size_t i = start; i <= path.size(); ++i)
    {
        if (i != path.size() && path[i] != '/') {
            continue;
        }
        const std::string part = path.substr(start, i - start);
        start = i + 1;
        if (part.empty()) {
            continue;
        }
        if (!current.empty() && current[current.size() - 1] != '/') {
            current += "/";
        }
        current += part;
#if defined(_WIN32)
        if (_mkdir(current.c_str()) != 0 && errno != EEXIST) {
            return false;
        }
#else
        if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
#endif
    }
    return true;
}

static bool ParseArgs(int argc, char** argv, Config& cfg)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << std::endl;
            return false;
        }
        const std::string val = argv[++i];
        char* end = nullptr;
        if (arg == "--output") {
            cfg.OutputPath = val;
        } else if (arg == "--io-root") {
            cfg.IoRoot = val;
        } else if (arg == "--quiet") {
            cfg.Quiet = (val == "1" || val == "true" || val == "TRUE" || val == "yes");
        } else if (arg == "--stress") {
            cfg.Stress = (val == "1" || val == "true" || val == "TRUE" || val == "yes");
        } else if (arg == "--core-use-cuda-batch") {
            cfg.CoreUseCudaBatch = (val == "1" || val == "true" || val == "TRUE" || val == "yes");
        } else if (arg == "--cuda-high-memory") {
            cfg.CudaHighMemory = (val == "1" || val == "true" || val == "TRUE" || val == "yes");
        } else if (arg == "--cuda-backend") {
            if (val == "auto") {
                cfg.CudaBackendMode = WirehairCudaBackend_Auto;
            } else if (val == "cpuonly") {
                cfg.CudaBackendMode = WirehairCudaBackend_CpuOnly;
            } else if (val == "cudaprefer") {
                cfg.CudaBackendMode = WirehairCudaBackend_CudaPrefer;
            } else if (val == "cudaonly") {
                cfg.CudaBackendMode = WirehairCudaBackend_CudaOnly;
            } else {
                std::cerr << "Unknown --cuda-backend value: " << val << std::endl;
                return false;
            }
        } else if (arg == "--trials") {
            const unsigned long long parsed = std::strtoull(val.c_str(), &end, 10);
            if (!end || *end != '\0') {
                return false;
            }
            cfg.Trials = static_cast<uint32_t>(parsed);
        } else if (arg == "--seed") {
            const unsigned long long parsed = std::strtoull(val.c_str(), &end, 10);
            if (!end || *end != '\0') {
                return false;
            }
            cfg.Seed = static_cast<uint64_t>(parsed);
        } else {
            std::cerr << "Unknown arg: " << arg << std::endl;
            return false;
        }
    }
    return cfg.Trials > 0;
}

static void FillRandom(std::vector<uint8_t>& data, XorShift64& prng)
{
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(prng.Next32() & 0xff);
    }
}

static uint64_t HashBytes(const uint8_t* data, size_t bytes)
{
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static double PercentileMs(std::vector<double> valuesMs, double pct)
{
    if (valuesMs.empty()) {
        return 0.0;
    }
    std::sort(valuesMs.begin(), valuesMs.end());
    const double idx = (pct / 100.0) * static_cast<double>(valuesMs.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(valuesMs.size() - 1, lo + 1);
    const double t = idx - static_cast<double>(lo);
    return valuesMs[lo] * (1.0 - t) + valuesMs[hi] * t;
}

static CoreCaseResult RunCoreCase(
    uint32_t N,
    uint32_t blockBytes,
    uint32_t lossPercent,
    uint32_t trials,
    uint64_t seed,
    bool useCudaBatchCore)
{
    CoreCaseResult r;
    r.N = N;
    r.BlockBytes = blockBytes;
    r.LossPercent = lossPercent;
    r.Trials = trials;

    uint64_t createUs = 0;
    uint64_t encodeUs = 0;
    uint64_t decodeUs = 0;
    uint32_t completedTrials = 0;
    uint64_t recoverUs = 0;
    uint64_t encodeBytes = 0;
    uint64_t decodeBytes = 0;
    uint64_t recoverBytes = 0;
    uint64_t extraSum = 0;
    uint64_t hostPackUs = 0;
    uint64_t hostPackBytes = 0;

    XorShift64 prng(seed ^ (N * 1315423911ULL) ^ (blockBytes << 7) ^ lossPercent);
    for (uint32_t t = 0; t < trials; ++t)
    {
        const uint32_t finalBytes = blockBytes > 1 ? blockBytes - 1 : 1;
        const uint64_t messageBytes = static_cast<uint64_t>(blockBytes) * (N - 1) + finalBytes;
        std::vector<uint8_t> message(static_cast<size_t>(messageBytes));
        FillRandom(message, prng);
        std::vector<uint8_t> recovered(static_cast<size_t>(messageBytes));
        std::vector<uint32_t> candidateIds;
        candidateIds.reserve(static_cast<size_t>(N) * 8);
        for (uint32_t blockId = 1; blockId <= N * 8; ++blockId)
        {
            if (prng.NextBounded(100) < lossPercent) {
                continue;
            }
            candidateIds.push_back(blockId);
        }
        if (candidateIds.empty()) {
            continue;
        }

        std::vector<uint8_t> encodedBlocks(candidateIds.size() * blockBytes);
        std::vector<uint32_t> encodedLens(candidateIds.size(), 0);
#if defined(AB_USE_CUDA)
        if (useCudaBatchCore)
        {
            const uint32_t startBlockId = 1;
            const uint32_t symbolCount = N * 8;
            std::vector<uint8_t> batchEncoded(static_cast<size_t>(symbolCount) * blockBytes);
            std::vector<uint32_t> batchLens(symbolCount, 0);
            WirehairEncodeBatchRequest encReq = {};
            encReq.request_id = 1;
            encReq.message = &message[0];
            encReq.message_bytes = messageBytes;
            encReq.block_bytes = blockBytes;
            encReq.start_block_id = startBlockId;
            encReq.block_count = symbolCount;
            encReq.block_data_out = &batchEncoded[0];
            encReq.block_stride_bytes = blockBytes;
            encReq.bytes_out = &batchLens[0];

            const uint64_t e0 = NowUs();
            const WirehairResult encResult = wirehair_cuda_encode_batch(&encReq);
            const uint64_t e1 = NowUs();
            encodeUs += (e1 - e0);
            if (encResult != Wirehair_Success) {
                continue;
            }

            std::vector<uint8_t> selectedMask(symbolCount, 0);
            uint32_t selectedCount = 0;
            for (uint32_t i = 0; i < symbolCount; ++i) {
                if (prng.NextBounded(100) >= lossPercent) {
                    selectedMask[i] = 1;
                    ++selectedCount;
                }
            }
            if (selectedCount == 0) {
                continue;
            }
            std::vector<uint32_t> selectedIds(selectedCount, 0);
            std::vector<uint8_t> decodeInput(static_cast<size_t>(selectedCount) * blockBytes, 0);
            std::vector<uint32_t> decodeLens(selectedCount, 0);
            uint32_t selectedWrite = 0;
            const uint64_t packStart = NowUs();
            for (uint32_t i = 0; i < symbolCount; ++i)
            {
                if (selectedMask[i] == 0) {
                    continue;
                }
                if (selectedWrite >= selectedCount) {
                    break;
                }
                selectedIds[selectedWrite] = startBlockId + i;
                decodeLens[selectedWrite] = batchLens[i];
                const uint8_t* src = &batchEncoded[static_cast<size_t>(i) * blockBytes];
                uint8_t* dst = &decodeInput[static_cast<size_t>(selectedWrite) * blockBytes];
                std::memcpy(dst, src, blockBytes);
                ++selectedWrite;
            }
            hostPackUs += (NowUs() - packStart);
            hostPackBytes += static_cast<uint64_t>(selectedCount) * blockBytes;

            const uint64_t selectedCount64 = static_cast<uint64_t>(selectedIds.size());
            encodeBytes += static_cast<uint64_t>(symbolCount) * blockBytes;
            decodeBytes += selectedCount64 * blockBytes;
#if defined(AB_USE_CUDA)
            WirehairCudaPerfStats statsBefore = {};
            WirehairCudaPerfStats statsAfter = {};
            statsBefore.struct_bytes = sizeof(WirehairCudaPerfStats);
            statsAfter.struct_bytes = sizeof(WirehairCudaPerfStats);
            const bool haveStatsBefore = (wirehair_cuda_get_perf_stats(&statsBefore) == Wirehair_Success);
#endif
            const uint64_t d0 = NowUs();
            WirehairDecodeBatchRequest decReq = {};
            decReq.request_id = 2;
            decReq.message_bytes = messageBytes;
            decReq.block_bytes = blockBytes;
            decReq.block_ids = &selectedIds[0];
            decReq.block_data = &decodeInput[0];
            decReq.block_data_bytes = &decodeLens[0];
            decReq.symbol_count = static_cast<uint32_t>(selectedIds.size());
            decReq.block_stride_bytes = blockBytes;
            decReq.message_out = &recovered[0];
            const WirehairResult decResult = wirehair_cuda_decode_batch(&decReq);
            const uint64_t d1 = NowUs();
            decodeUs += (d1 - d0);
#if defined(AB_USE_CUDA)
            const bool haveStatsAfter = (wirehair_cuda_get_perf_stats(&statsAfter) == Wirehair_Success);
            uint64_t preprocessUs = 0;
            uint64_t feedUs = 0;
            uint64_t recoverOnlyUs = 0;
            if (haveStatsBefore && haveStatsAfter)
            {
                if (statsAfter.decode_preprocess_us >= statsBefore.decode_preprocess_us) {
                    preprocessUs = statsAfter.decode_preprocess_us - statsBefore.decode_preprocess_us;
                }
                if (statsAfter.decode_feed_us >= statsBefore.decode_feed_us) {
                    feedUs = statsAfter.decode_feed_us - statsBefore.decode_feed_us;
                }
                if (statsAfter.decode_recover_us >= statsBefore.decode_recover_us) {
                    recoverOnlyUs = statsAfter.decode_recover_us - statsBefore.decode_recover_us;
                }
            }
            r.CudaPreprocessUs += static_cast<double>(preprocessUs);
            r.CudaDecodeFeedUs += static_cast<double>(feedUs);
            r.CudaRecoverOnlyUs += static_cast<double>(recoverOnlyUs);
#endif
            if (decResult == Wirehair_Success &&
                HashBytes(&recovered[0], recovered.size()) == HashBytes(&message[0], message.size()))
            {
#if defined(AB_USE_CUDA)
                if (haveStatsBefore && haveStatsAfter && recoverOnlyUs > 0) {
                    recoverUs += recoverOnlyUs;
                } else {
                    recoverUs += (d1 - d0);
                }
#else
                recoverUs += (d1 - d0);
#endif
                ++r.Successes;
                const uint32_t needed = static_cast<uint32_t>(selectedIds.size());
                if (needed > N) {
                    extraSum += (needed - N);
                }
                recoverBytes += messageBytes;

                // Differential harness: compare CUDA decode output against CPU decode path
                // on deterministic first-trial replay per case.
                if (t == 0)
                {
                    std::vector<uint8_t> cpuRecovered(static_cast<size_t>(messageBytes), 0);
                    WirehairCodec cpuDecoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
                    bool cpuOk = false;
                    if (cpuDecoder)
                    {
                        for (uint32_t di = 0; di < static_cast<uint32_t>(selectedIds.size()); ++di)
                        {
                            const uint8_t* symbol = &decodeInput[static_cast<size_t>(di) * blockBytes];
                            const WirehairResult cpuDec = wirehair_decode(cpuDecoder, selectedIds[di], symbol, decodeLens[di]);
                            if (cpuDec == Wirehair_Success) {
                                cpuOk = (wirehair_recover(cpuDecoder, &cpuRecovered[0], messageBytes) == Wirehair_Success);
                                break;
                            }
                            if (cpuDec != Wirehair_NeedMore) {
                                break;
                            }
                        }
                        wirehair_free(cpuDecoder);
                    }
                    ++r.DifferentialChecks;
                    if (!cpuOk || HashBytes(&cpuRecovered[0], cpuRecovered.size()) != HashBytes(&recovered[0], recovered.size())) {
                        ++r.DifferentialMismatches;
                    }
                }
            }
            continue;
        }
#endif

        const uint64_t t0 = NowUs();
        WirehairCodec encoder = wirehair_encoder_create(nullptr, &message[0], messageBytes, blockBytes);
        WirehairCodec decoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
        const uint64_t t1 = NowUs();
        createUs += (t1 - t0);
        if (!encoder || !decoder) {
            if (encoder) {
                wirehair_free(encoder);
            }
            if (decoder) {
                wirehair_free(decoder);
            }
            continue;
        }

        size_t encodedCount = 0;
        uint64_t trialEncodeBytes = 0;
        const uint64_t e0 = NowUs();
        for (size_t i = 0; i < candidateIds.size(); ++i)
        {
            const size_t offset = i * blockBytes;
            uint32_t writeLen = 0;
            const WirehairResult encResult = wirehair_encode(
                encoder,
                candidateIds[i],
                &encodedBlocks[offset],
                blockBytes,
                &writeLen);
            if (encResult != Wirehair_Success) {
                break;
            }
            encodedLens[i] = writeLen;
            trialEncodeBytes += writeLen;
            ++encodedCount;
        }
        const uint64_t e1 = NowUs();
        encodeUs += (e1 - e0);
        encodeBytes += trialEncodeBytes;

        uint32_t needed = 0;
        bool decoded = false;
        uint64_t trialDecodeBytes = 0;
        const uint64_t d0 = NowUs();
        for (size_t i = 0; i < encodedCount; ++i)
        {
            ++needed;
            trialDecodeBytes += encodedLens[i];
            const WirehairResult decResult = wirehair_decode(
                decoder,
                candidateIds[i],
                &encodedBlocks[i * blockBytes],
                encodedLens[i]);
            if (decResult == Wirehair_Success) {
                decoded = true;
                break;
            }
            if (decResult != Wirehair_NeedMore) {
                break;
            }
        }
        const uint64_t d1 = NowUs();
        decodeUs += (d1 - d0);
        decodeBytes += trialDecodeBytes;

        if (decoded) {
            const uint64_t r0 = NowUs();
            const WirehairResult recResult = wirehair_recover(decoder, &recovered[0], messageBytes);
            const uint64_t r1 = NowUs();
            recoverUs += (r1 - r0);
            recoverBytes += messageBytes;
            if (recResult == Wirehair_Success &&
                HashBytes(&recovered[0], recovered.size()) == HashBytes(&message[0], message.size()))
            {
                ++r.Successes;
                if (needed > N) {
                    extraSum += (needed - N);
                }
            }
        }

        wirehair_free(decoder);
        wirehair_free(encoder);
    }

    r.AvgCreateUs = trials > 0 ? static_cast<double>(createUs) / trials : 0.0;
    r.AvgExtra = r.Successes > 0 ? static_cast<double>(extraSum) / r.Successes : 0.0;
    r.EncodeMBps = encodeUs > 0 ? static_cast<double>(encodeBytes) / encodeUs : 0.0;
    r.DecodeMBps = decodeUs > 0 ? static_cast<double>(decodeBytes) / decodeUs : 0.0;
    r.RecoverMBps = recoverUs > 0 ? static_cast<double>(recoverBytes) / recoverUs : 0.0;
    r.HostPackMB = static_cast<double>(hostPackBytes) / (1024.0 * 1024.0);
    r.HostPackUs = static_cast<double>(hostPackUs);
    return r;
}

static bool DecodeWithSymbols(
    const std::vector<uint32_t>& symbolIds,
    WirehairCodec encoder,
    uint64_t messageBytes,
    uint32_t blockBytes,
    const std::vector<uint8_t>& source,
    uint32_t& neededOut)
{
    neededOut = 0;
    WirehairCodec decoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
    if (!decoder) {
        return false;
    }
    std::vector<uint8_t> symbol(blockBytes);
    std::vector<uint8_t> recovered(source.size());
    for (size_t i = 0; i < symbolIds.size(); ++i)
    {
        uint32_t writeLen = 0;
        if (wirehair_encode(encoder, symbolIds[i], &symbol[0], blockBytes, &writeLen) != Wirehair_Success) {
            wirehair_free(decoder);
            return false;
        }
        ++neededOut;
        const WirehairResult decodeResult = wirehair_decode(decoder, symbolIds[i], &symbol[0], writeLen);
        if (decodeResult == Wirehair_Success) {
            const WirehairResult rec = wirehair_recover(decoder, &recovered[0], messageBytes);
            wirehair_free(decoder);
            if (rec != Wirehair_Success) {
                return false;
            }
            return HashBytes(&recovered[0], recovered.size()) == HashBytes(&source[0], source.size());
        }
        if (decodeResult != Wirehair_NeedMore) {
            wirehair_free(decoder);
            return false;
        }
    }
    wirehair_free(decoder);
    return false;
}

static ParityCaseResult RunParityCase(uint32_t driveCount, uint32_t trials, uint64_t seed)
{
    ParityCaseResult r;
    r.DriveCount = driveCount;
    r.Trials = trials;
    XorShift64 prng(seed ^ (driveCount * 0x9e37ULL));
    const uint32_t blockBytes = 64 * 1024;
    const uint32_t N = 256;
    const uint32_t finalBytes = blockBytes - 1;
    const uint64_t messageBytes = static_cast<uint64_t>(blockBytes) * (N - 1) + finalBytes;
    std::vector<uint8_t> message(static_cast<size_t>(messageBytes));
    FillRandom(message, prng);
    WirehairCodec encoder = wirehair_encoder_create(nullptr, &message[0], messageBytes, blockBytes);
    if (!encoder) {
        return r;
    }

    const uint32_t symbolsPerDrive = 64;
    std::vector<std::vector<uint32_t> > symbols(driveCount);
    for (uint32_t i = 0; i < symbolsPerDrive * driveCount; ++i) {
        symbols[i % driveCount].push_back(N + i);
    }

    uint64_t neededSum = 0;
    for (uint32_t t = 0; t < trials; ++t)
    {
        const uint32_t subset = std::max<uint32_t>(1, prng.NextBounded(driveCount) + 1);
        std::set<uint32_t> selected;
        while (selected.size() < subset) {
            selected.insert(prng.NextBounded(driveCount));
        }
        std::vector<uint32_t> symbolIds;
        for (std::set<uint32_t>::const_iterator it = selected.begin(); it != selected.end(); ++it) {
            symbolIds.insert(symbolIds.end(), symbols[*it].begin(), symbols[*it].end());
        }
        std::sort(symbolIds.begin(), symbolIds.end());
        uint32_t needed = 0;
        if (DecodeWithSymbols(symbolIds, encoder, messageBytes, blockBytes, message, needed)) {
            ++r.Successes;
            neededSum += needed;
        }
    }
    wirehair_free(encoder);
    r.AvgNeeded = r.Successes > 0 ? static_cast<double>(neededSum) / r.Successes : 0.0;
    return r;
}

static ChurnCaseResult RunChurnCase(uint32_t targets, uint32_t trials, uint64_t seed)
{
    ChurnCaseResult r;
    r.TargetCount = targets;
    r.Trials = trials;
    XorShift64 prng(seed ^ (targets * 73ULL));
    const uint32_t blockBytes = 32 * 1024;
    const uint32_t N = 512;
    const uint64_t messageBytes = static_cast<uint64_t>(N) * blockBytes - 1;
    std::vector<uint8_t> message(static_cast<size_t>(messageBytes));
    FillRandom(message, prng);

    uint64_t neededSum = 0;
    for (uint32_t t = 0; t < trials; ++t)
    {
        WirehairCodec encoder = wirehair_encoder_create(nullptr, &message[0], messageBytes, blockBytes);
        WirehairCodec decoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
        if (!encoder || !decoder) {
            if (encoder) {
                wirehair_free(encoder);
            }
            if (decoder) {
                wirehair_free(decoder);
            }
            continue;
        }

        std::vector<uint8_t> symbol(blockBytes);
        std::vector<uint8_t> recovered(message.size());
        std::vector<uint8_t> active(targets, 1);
        uint32_t needed = 0;
        bool ok = false;
        for (uint32_t emitted = 0; emitted < N * 8; ++emitted)
        {
            if (prng.NextBounded(100) < 8) {
                active[prng.NextBounded(targets)] ^= 1;
            }
            const uint32_t blockId = N + emitted;
            const uint32_t target = blockId % targets;
            if (!active[target]) {
                continue;
            }
            uint32_t writeLen = 0;
            if (wirehair_encode(encoder, blockId, &symbol[0], blockBytes, &writeLen) != Wirehair_Success) {
                break;
            }
            ++needed;
            const WirehairResult dec = wirehair_decode(decoder, blockId, &symbol[0], writeLen);
            if (dec == Wirehair_Success) {
                if (wirehair_recover(decoder, &recovered[0], messageBytes) == Wirehair_Success &&
                    HashBytes(&recovered[0], recovered.size()) == HashBytes(&message[0], message.size()))
                {
                    ok = true;
                }
                break;
            }
            if (dec != Wirehair_NeedMore) {
                break;
            }
        }
        if (ok) {
            ++r.Successes;
            neededSum += needed;
        }
        wirehair_free(decoder);
        wirehair_free(encoder);
    }

    r.AvgNeeded = r.Successes > 0 ? static_cast<double>(neededSum) / r.Successes : 0.0;
    return r;
}

static StorageCaseResult RunStorageCase(uint32_t trials, const std::string& root, uint64_t seed, bool highMemoryMode)
{
    StorageCaseResult r;
    r.ModeProfile = highMemoryMode ? "high-memory" : "balanced";
    r.Trials = trials;
    XorShift64 prng(seed ^ 0xa5a5a5ULL);
    const uint32_t blockBytes = 64 * 1024;
    const uint32_t N = 128;
    const uint64_t messageBytes = static_cast<uint64_t>(N) * blockBytes - 1;
    const uint32_t totalSymbols = N + 32;
    std::vector<uint8_t> message(static_cast<size_t>(messageBytes));
    FillRandom(message, prng);
#if defined(AB_USE_CUDA)
    bool useCudaStoragePath = true;
    WirehairCudaConfig activeCfg = {};
    activeCfg.struct_bytes = sizeof(WirehairCudaConfig);
    if (wirehair_cuda_get_config(&activeCfg) == Wirehair_Success) {
        useCudaStoragePath = (activeCfg.backend_mode != WirehairCudaBackend_CpuOnly);
    }
#else
    const bool useCudaStoragePath = false;
#endif

    uint64_t writeUs = 0;
    uint64_t readUs = 0;
    uint64_t bytesW = 0;
    uint64_t bytesR = 0;
    uint64_t neededSum = 0;
    uint64_t writeBatchCount = 0;
    uint64_t readBatchCount = 0;
    uint32_t writeBatchSymbols = highMemoryMode ? 384U : 128U;
    const char* writeBatchEnv = std::getenv(highMemoryMode
        ? "WIREHAIR_STORAGE_WRITE_BATCH_HIGH"
        : "WIREHAIR_STORAGE_WRITE_BATCH");
    if (writeBatchEnv && writeBatchEnv[0] != '\0')
    {
        const unsigned long parsed = std::strtoul(writeBatchEnv, nullptr, 10);
        if (parsed > 0UL) {
            writeBatchSymbols = static_cast<uint32_t>(std::min<unsigned long>(4096UL, parsed));
        }
    }
    const uint32_t readBatchSymbols = highMemoryMode ? 384U : 128U;
    std::vector<double> writeLatencyMs;
    std::vector<double> readLatencyMs;
    writeLatencyMs.reserve(static_cast<size_t>(trials) * totalSymbols);
    readLatencyMs.reserve(static_cast<size_t>(trials) * totalSymbols);
    uint64_t successfulReadBytesTotal = 0;
    uint64_t attemptedReadBytesTotal = 0;

    EnsureDirectory(root);
    for (uint32_t t = 0; t < trials; ++t)
    {
        const std::string trialDir = JoinPath(root, "trial-" + std::to_string(t));
        EnsureDirectory(trialDir);

        std::vector<uint32_t> symbolIds(totalSymbols, 0);
        std::vector<uint32_t> symbolLens(totalSymbols, 0);
        std::vector<uint8_t> encodedSlab(static_cast<size_t>(totalSymbols) * blockBytes, 0);

        if (useCudaStoragePath)
        {
            WirehairEncodeBatchRequest encReq = {};
            encReq.request_id = static_cast<uint64_t>(t + 1);
            encReq.message = &message[0];
            encReq.message_bytes = messageBytes;
            encReq.block_bytes = blockBytes;
            encReq.start_block_id = N;
            encReq.block_count = totalSymbols;
            encReq.block_data_out = &encodedSlab[0];
            encReq.block_stride_bytes = blockBytes;
            encReq.bytes_out = &symbolLens[0];
            if (wirehair_cuda_encode_batch(&encReq) != Wirehair_Success) {
                continue;
            }
        }
        else
        {
            WirehairCodec codec = wirehair_encoder_create(nullptr, &message[0], messageBytes, blockBytes);
            if (!codec) {
                continue;
            }
            bool encodeOk = true;
            for (uint32_t i = 0; i < totalSymbols; ++i)
            {
                uint8_t* output = &encodedSlab[static_cast<size_t>(i) * blockBytes];
                uint32_t written = 0;
                if (wirehair_encode(codec, N + i, output, blockBytes, &written) != Wirehair_Success || written == 0) {
                    encodeOk = false;
                    break;
                }
                symbolLens[i] = written;
            }
            wirehair_free(codec);
            if (!encodeOk) {
                continue;
            }
        }
        for (uint32_t i = 0; i < totalSymbols; ++i) {
            symbolIds[i] = N + i;
        }

        std::vector<uint32_t> shuffledWrite(static_cast<uint32_t>(symbolIds.size()), 0);
        for (uint32_t i = 0; i < shuffledWrite.size(); ++i) {
            shuffledWrite[i] = i;
        }
        for (size_t i = shuffledWrite.size(); i > 1; --i) {
            const size_t j = prng.NextBounded(static_cast<uint32_t>(i));
            std::swap(shuffledWrite[i - 1], shuffledWrite[j]);
        }

        const std::string packPath = JoinPath(trialDir, "symbols.pack");
        std::fstream rw(packPath.c_str(), std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!rw) {
            continue;
        }
        std::vector<char> ioBuffer(highMemoryMode ? (8U * 1024U * 1024U) : (2U * 1024U * 1024U), 0);
        rw.rdbuf()->pubsetbuf(ioBuffer.data(), static_cast<std::streamsize>(ioBuffer.size()));
        const size_t fileBytes = symbolIds.size() * static_cast<size_t>(blockBytes);
        if (fileBytes > 0) {
            rw.seekp(static_cast<std::streamoff>(fileBytes - 1), std::ios::beg);
            char zero = 0;
            rw.write(&zero, 1);
            rw.flush();
        }

        for (size_t start = 0; start < shuffledWrite.size(); start += writeBatchSymbols)
        {
            const size_t count = std::min<size_t>(writeBatchSymbols, shuffledWrite.size() - start);
            for (size_t i = 0; i < count; ++i)
            {
                const uint32_t sourceIndex = shuffledWrite[start + i];
                const uint64_t w0 = NowUs();
                const std::streamoff offset = static_cast<std::streamoff>(sourceIndex) * blockBytes;
                rw.seekp(offset, std::ios::beg);
                rw.write(reinterpret_cast<const char*>(&encodedSlab[static_cast<size_t>(sourceIndex) * blockBytes]), blockBytes);
                const uint64_t w1 = NowUs();
                if (!rw) {
                    break;
                }
                writeLatencyMs.push_back(static_cast<double>(w1 - w0) / 1000.0);
                ++writeBatchCount;
                writeUs += (w1 - w0);
                bytesW += blockBytes;
            }
        }
        rw.flush();
        rw.clear();

        uint32_t needed = 0;
        std::vector<uint8_t> recovered(message.size());
        bool ok = false;
        bool decodeHardFail = false;
        uint64_t trialReadBytes = 0;
        uint64_t readBytesAtSuccess = 0;
        bool capturedReadBytesAtSuccess = false;
        std::vector<uint32_t> shuffledRead = shuffledWrite;
        for (size_t i = shuffledRead.size(); i > 1; --i) {
            const size_t j = prng.NextBounded(static_cast<uint32_t>(i));
            std::swap(shuffledRead[i - 1], shuffledRead[j]);
        }
        if (rw)
        {
            std::FILE* readFile = std::fopen(packPath.c_str(), "rb");
            std::vector<char> readIoBuffer;
            if (readFile) {
                readIoBuffer.assign(highMemoryMode ? (8U * 1024U * 1024U) : (2U * 1024U * 1024U), 0);
                std::setvbuf(readFile, readIoBuffer.data(), _IOFBF, readIoBuffer.size());
            }
            struct ReadBatch
            {
                bool Ok = true;
                std::vector<uint32_t> SourceIndices;
                std::vector<uint8_t> Data;
                std::vector<double> LatenciesMs;
                uint64_t ReadUs = 0;
                uint64_t Bytes = 0;
                uint64_t AttemptedBytes = 0;
            };
            std::vector<uint32_t> decodeIds;
            std::vector<uint32_t> decodeLens;
            std::vector<uint8_t> decodeSlab(static_cast<size_t>(totalSymbols) * blockBytes, 0);
            decodeIds.reserve(totalSymbols);
            decodeLens.reserve(totalSymbols);
            uint32_t decodeCount = 0;
            auto ReadBatchAsync = [&](size_t start) -> ReadBatch
            {
                const size_t count = std::min<size_t>(readBatchSymbols, shuffledRead.size() - start);
                ReadBatch batch = {};
                batch.SourceIndices.reserve(count);
                batch.Data.resize(count * static_cast<size_t>(blockBytes), 0);
                batch.LatenciesMs.reserve(count);
                for (size_t i = 0; i < count; ++i)
                {
                    const uint32_t sourceIndex = shuffledRead[start + i];
                    const std::streamoff offset = static_cast<std::streamoff>(sourceIndex) * blockBytes;
                    uint8_t* symbol = &batch.Data[i * static_cast<size_t>(blockBytes)];
                    const uint64_t r0 = NowUs();
                    size_t got = 0;
                    if (readFile)
                    {
#if defined(_WIN32)
                        if (_fseeki64(readFile, static_cast<long long>(offset), SEEK_SET) == 0)
#else
                        if (std::fseeko(readFile, static_cast<off_t>(offset), SEEK_SET) == 0)
#endif
                        {
                            got = std::fread(symbol, 1, blockBytes, readFile);
                        }
                    }
                    const uint64_t r1 = NowUs();
                    if (got == 0) {
                        batch.Ok = false;
                        break;
                    }
                    batch.SourceIndices.push_back(sourceIndex);
                    batch.LatenciesMs.push_back(static_cast<double>(r1 - r0) / 1000.0);
                    batch.ReadUs += (r1 - r0);
                    batch.Bytes += static_cast<uint64_t>(got);
                    batch.AttemptedBytes += static_cast<uint64_t>(got);
                }
                batch.Data.resize(batch.SourceIndices.size() * static_cast<size_t>(blockBytes));
                return batch;
            };

            size_t nextStart = 0;
            std::future<ReadBatch> pendingRead = std::async(std::launch::async, ReadBatchAsync, nextStart);
            nextStart += readBatchSymbols;

            while (pendingRead.valid() && !ok && !decodeHardFail)
            {
                ReadBatch batch = pendingRead.get();
                if (nextStart < shuffledRead.size()) {
                    pendingRead = std::async(std::launch::async, ReadBatchAsync, nextStart);
                    nextStart += readBatchSymbols;
                }

                if (!batch.Ok) {
                    break;
                }

                for (double latency : batch.LatenciesMs) {
                    readLatencyMs.push_back(latency);
                }
                readBatchCount += batch.SourceIndices.size();
                readUs += batch.ReadUs;
                bytesR += batch.Bytes;
                attemptedReadBytesTotal += batch.AttemptedBytes;
                trialReadBytes += batch.Bytes;

                if (!ok)
                {
                    for (size_t i = 0; i < batch.SourceIndices.size(); ++i)
                    {
                        const uint32_t sourceIndex = batch.SourceIndices[i];
                        const uint32_t blockId = symbolIds[sourceIndex];
                        uint8_t* symbolDest = &decodeSlab[static_cast<size_t>(decodeCount) * blockBytes];
                        const uint8_t* symbolSrc = &batch.Data[i * static_cast<size_t>(blockBytes)];
                        std::memcpy(symbolDest, symbolSrc, blockBytes);
                        ++needed;
                        decodeIds.push_back(blockId);
                        decodeLens.push_back(symbolLens[sourceIndex]);
                        ++decodeCount;
                    }

                    if (decodeCount >= N)
                    {
                    WirehairResult dr = Wirehair_Error;
                    if (useCudaStoragePath)
                    {
                        WirehairDecodeBatchRequest decReq = {};
                        decReq.request_id = static_cast<uint64_t>(t + 1);
                        decReq.message_bytes = messageBytes;
                        decReq.block_bytes = blockBytes;
                        decReq.block_ids = &decodeIds[0];
                        decReq.block_data = &decodeSlab[0];
                        decReq.block_data_bytes = &decodeLens[0];
                        decReq.symbol_count = decodeCount;
                        decReq.block_stride_bytes = blockBytes;
                        decReq.message_out = &recovered[0];
                        dr = wirehair_cuda_decode_batch(&decReq);
                    }
                    else
                    {
                        WirehairCodec decoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
                        if (!decoder) {
                            dr = Wirehair_Error;
                        } else {
                            dr = Wirehair_NeedMore;
                            for (uint32_t si = 0; si < decodeCount; ++si)
                            {
                                const uint8_t* symbol = &decodeSlab[static_cast<size_t>(si) * blockBytes];
                                const WirehairResult feed = wirehair_decode(decoder, decodeIds[si], symbol, decodeLens[si]);
                                if (feed == Wirehair_Success) {
                                    dr = wirehair_recover(decoder, &recovered[0], messageBytes);
                                    break;
                                }
                                if (feed != Wirehair_NeedMore) {
                                    dr = feed;
                                    break;
                                }
                            }
                            wirehair_free(decoder);
                        }
                    }
                        if (dr == Wirehair_Success &&
                            HashBytes(&recovered[0], recovered.size()) == HashBytes(&message[0], message.size()))
                        {
                            ok = true;
                            if (!capturedReadBytesAtSuccess) {
                                readBytesAtSuccess = trialReadBytes;
                                capturedReadBytesAtSuccess = true;
                            }
                        }
                        else if (dr != Wirehair_NeedMore && dr != Wirehair_Success)
                        {
                            decodeHardFail = true;
                        }
                    }
                }
            }
            if (readFile) {
                std::fclose(readFile);
            }
        }
        if (ok) {
            ++r.Successes;
            neededSum += needed;
            successfulReadBytesTotal += capturedReadBytesAtSuccess ? readBytesAtSuccess : trialReadBytes;
        }

    }

    r.WriteMBps = writeUs > 0 ? static_cast<double>(bytesW) / writeUs : 0.0;
    r.ReadMBps = readUs > 0 ? static_cast<double>(bytesR) / readUs : 0.0;
    r.AvgNeeded = r.Successes > 0 ? static_cast<double>(neededSum) / r.Successes : 0.0;
    r.WriteBatchCount = writeBatchCount;
    r.ReadBatchCount = readBatchCount;
    r.AvgWriteBatchBytes = writeBatchCount > 0 ? static_cast<double>(bytesW) / writeBatchCount : 0.0;
    r.AvgReadBatchBytes = readBatchCount > 0 ? static_cast<double>(bytesR) / readBatchCount : 0.0;
    r.RandomWriteP50Ms = PercentileMs(writeLatencyMs, 50.0);
    r.RandomWriteP95Ms = PercentileMs(writeLatencyMs, 95.0);
    r.RandomReadP50Ms = PercentileMs(readLatencyMs, 50.0);
    r.RandomReadP95Ms = PercentileMs(readLatencyMs, 95.0);
    r.EffectiveReadBytesBeforeSuccess = r.Successes > 0
        ? static_cast<double>(successfulReadBytesTotal) / static_cast<double>(r.Successes)
        : 0.0;
    r.WorkCompletionRatio = attemptedReadBytesTotal > 0
        ? static_cast<double>(successfulReadBytesTotal) / static_cast<double>(attemptedReadBytesTotal)
        : 0.0;
    return r;
}

static bool PollUntilPipelineEvent(
    WirehairPipeline pipeline,
    uint64_t expectedRequestId,
    uint32_t expectedJobType,
    WirehairPipelineEvent* out,
    int totalTimeoutMs)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(totalTimeoutMs);
    while (std::chrono::steady_clock::now() < deadline)
    {
        WirehairPipelineEvent ev = {};
        const WirehairResult pr = wirehair_pipeline_poll(pipeline, 250, &ev);
        if (pr != Wirehair_Success) {
            continue;
        }
        if (ev.request_id == expectedRequestId && ev.job_type == expectedJobType) {
            *out = ev;
            return true;
        }
    }
    return false;
}

static ThreadScalingResult RunThreadScalingCase(uint32_t threads, uint32_t trials, uint64_t seed)
{
    ThreadScalingResult r;
    r.Threads = threads;
    r.Trials = trials;

    // Lightweight pipeline smoke: large enough for decode success, small enough for fast CI runs.
    const uint32_t blockBytes = 1024;
    const uint32_t N = 128;
    const uint64_t messageBytes = static_cast<uint64_t>(N) * blockBytes - 1;
    const uint32_t startId = N + 1;
    const uint32_t symbolCount = N + 32;
    const uint64_t totalBytes = static_cast<uint64_t>(symbolCount) * blockBytes;
    XorShift64 prng(seed ^ (threads * 1009ULL));
    std::vector<double> latenciesMs;
    latenciesMs.reserve(trials);
    uint64_t encodeUs = 0;
    uint64_t decodeUs = 0;
    uint32_t completedTrials = 0;
    uint32_t requestId = 1;
    const int kPhaseTimeoutMs = 180000;

    for (uint32_t t = 0; t < trials; ++t)
    {
        WirehairPipelineConfig cfg = {};
        cfg.encode_threads = threads;
        cfg.decode_threads = threads;
        cfg.queue_capacity = std::max<uint32_t>(32, threads * 8);
        WirehairPipeline pipeline = wirehair_pipeline_create(&cfg);
        if (!pipeline) {
            continue;
        }

        std::vector<uint8_t> message(static_cast<size_t>(messageBytes));
        std::vector<uint8_t> recovered(static_cast<size_t>(messageBytes));
        std::vector<uint8_t> encoded(static_cast<size_t>(symbolCount) * blockBytes, 0);
        std::vector<uint32_t> symbolLens(symbolCount, 0);
        std::vector<uint32_t> symbolIds(symbolCount, 0);
        FillRandom(message, prng);
        for (uint32_t i = 0; i < symbolCount; ++i) {
            symbolIds[i] = startId + i;
        }

        WirehairEncodeBatchRequest encReq = {};
        encReq.request_id = requestId++;
        encReq.message = &message[0];
        encReq.message_bytes = messageBytes;
        encReq.block_bytes = blockBytes;
        encReq.start_block_id = startId;
        encReq.block_count = symbolCount;
        encReq.block_data_out = &encoded[0];
        encReq.block_stride_bytes = blockBytes;
        encReq.bytes_out = &symbolLens[0];

        const uint64_t startUs = NowUs();
        if (wirehair_encode_batch_async(pipeline, &encReq) != Wirehair_Success) {
            wirehair_pipeline_free(pipeline);
            continue;
        }
        WirehairPipelineEvent event = {};
        if (!PollUntilPipelineEvent(pipeline, encReq.request_id, WirehairPipelineJob_EncodeBatch, &event, kPhaseTimeoutMs)) {
            wirehair_pipeline_free(pipeline);
            continue;
        }
        if (event.result != Wirehair_Success) {
            wirehair_pipeline_free(pipeline);
            continue;
        }
        const uint64_t afterEncodeUs = NowUs();
        encodeUs += (afterEncodeUs - startUs);

        WirehairDecodeBatchRequest decReq = {};
        decReq.request_id = requestId++;
        decReq.message_bytes = messageBytes;
        decReq.block_bytes = blockBytes;
        decReq.block_ids = &symbolIds[0];
        decReq.block_data = &encoded[0];
        decReq.block_data_bytes = &symbolLens[0];
        decReq.symbol_count = symbolCount;
        decReq.block_stride_bytes = blockBytes;
        decReq.message_out = &recovered[0];

        if (wirehair_decode_batch_async(pipeline, &decReq) != Wirehair_Success) {
            wirehair_pipeline_free(pipeline);
            continue;
        }
        if (!PollUntilPipelineEvent(pipeline, decReq.request_id, WirehairPipelineJob_DecodeBatch, &event, kPhaseTimeoutMs)) {
            wirehair_pipeline_free(pipeline);
            continue;
        }
        const uint64_t endUs = NowUs();
        decodeUs += (endUs - afterEncodeUs);

        if (event.result != Wirehair_Success) {
            wirehair_pipeline_free(pipeline);
            continue;
        }
        if (HashBytes(&recovered[0], recovered.size()) != HashBytes(&message[0], message.size())) {
            wirehair_pipeline_free(pipeline);
            continue;
        }
        latenciesMs.push_back(static_cast<double>(endUs - startUs) / 1000.0);
        ++completedTrials;
        wirehair_pipeline_free(pipeline);
    }

    r.EncodeMBps = encodeUs > 0 ? static_cast<double>(totalBytes * completedTrials) / encodeUs : 0.0;
    r.DecodeMBps = decodeUs > 0 ? static_cast<double>(totalBytes * completedTrials) / decodeUs : 0.0;
    r.P50LatencyMs = PercentileMs(latenciesMs, 50.0);
    r.P95LatencyMs = PercentileMs(latenciesMs, 95.0);
    return r;
}

static void WriteStorageJson(std::ostream& out, const StorageCaseResult& storage)
{
    out << "\"trials\":" << storage.Trials
        << ",\"successes\":" << storage.Successes
        << ",\"success_rate\":" << (storage.Trials > 0 ? static_cast<double>(storage.Successes) / storage.Trials : 0.0)
        << ",\"write_mbps\":" << storage.WriteMBps
        << ",\"read_mbps\":" << storage.ReadMBps
        << ",\"avg_needed\":" << storage.AvgNeeded
        << ",\"write_batch_count\":" << storage.WriteBatchCount
        << ",\"read_batch_count\":" << storage.ReadBatchCount
        << ",\"avg_write_batch_bytes\":" << storage.AvgWriteBatchBytes
        << ",\"avg_read_batch_bytes\":" << storage.AvgReadBatchBytes
        << ",\"random_write_p50_ms\":" << storage.RandomWriteP50Ms
        << ",\"random_write_p95_ms\":" << storage.RandomWriteP95Ms
        << ",\"random_read_p50_ms\":" << storage.RandomReadP50Ms
        << ",\"random_read_p95_ms\":" << storage.RandomReadP95Ms
        << ",\"effective_read_bytes_before_success\":" << storage.EffectiveReadBytesBeforeSuccess
        << ",\"work_completion_ratio\":" << storage.WorkCompletionRatio
        << ",\"mode_profile\":\"" << storage.ModeProfile << "\"";
}

static void WriteJson(
    const std::string& path,
    const Config& cfg,
    const std::vector<CoreCaseResult>& core,
    const std::vector<ParityCaseResult>& parity,
    const std::vector<ChurnCaseResult>& churn,
    const StorageProfilesResult& storageProfiles,
    const std::vector<ThreadScalingResult>& threadScaling,
    const CudaPerfStatsResult& cudaStats)
{
    std::ofstream out(path.c_str(), std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open output file: " << path << std::endl;
        return;
    }

    out << "{\n";
    out << "  \"variant\": \"" << AB_VARIANT_NAME << "\",\n";
    out << "  \"seed\": " << cfg.Seed << ",\n";
    out << "  \"trials\": " << cfg.Trials << ",\n";
    out << "  \"core\": [\n";
    for (size_t i = 0; i < core.size(); ++i)
    {
        const CoreCaseResult& c = core[i];
        out << "    {\"n\":" << c.N
            << ",\"block_bytes\":" << c.BlockBytes
            << ",\"loss_percent\":" << c.LossPercent
            << ",\"trials\":" << c.Trials
            << ",\"successes\":" << c.Successes
            << ",\"avg_extra\":" << std::fixed << std::setprecision(4) << c.AvgExtra
            << ",\"encode_mbps\":" << c.EncodeMBps
            << ",\"decode_mbps\":" << c.DecodeMBps
            << ",\"recover_mbps\":" << c.RecoverMBps
            << ",\"avg_create_us\":" << c.AvgCreateUs
            << ",\"host_pack_mb\":" << c.HostPackMB
            << ",\"host_pack_us\":" << c.HostPackUs
            << ",\"cuda_preprocess_us\":" << (c.Trials > 0 ? (c.CudaPreprocessUs / c.Trials) : 0.0)
            << ",\"cuda_decode_feed_us\":" << (c.Trials > 0 ? (c.CudaDecodeFeedUs / c.Trials) : 0.0)
            << ",\"cuda_recover_only_us\":" << (c.Trials > 0 ? (c.CudaRecoverOnlyUs / c.Trials) : 0.0)
            << ",\"differential_checks\":" << c.DifferentialChecks
            << ",\"differential_mismatches\":" << c.DifferentialMismatches
            << "}";
        if (i + 1 < core.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"parity\": [\n";
    for (size_t i = 0; i < parity.size(); ++i)
    {
        const ParityCaseResult& p = parity[i];
        out << "    {\"drive_count\":" << p.DriveCount
            << ",\"trials\":" << p.Trials
            << ",\"successes\":" << p.Successes
            << ",\"success_rate\":" << (p.Trials > 0 ? static_cast<double>(p.Successes) / p.Trials : 0.0)
            << ",\"avg_needed\":" << p.AvgNeeded
            << "}";
        if (i + 1 < parity.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"churn\": [\n";
    for (size_t i = 0; i < churn.size(); ++i)
    {
        const ChurnCaseResult& c = churn[i];
        out << "    {\"target_count\":" << c.TargetCount
            << ",\"trials\":" << c.Trials
            << ",\"successes\":" << c.Successes
            << ",\"success_rate\":" << (c.Trials > 0 ? static_cast<double>(c.Successes) / c.Trials : 0.0)
            << ",\"avg_needed\":" << c.AvgNeeded
            << "}";
        if (i + 1 < churn.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    // Backward-compatible primary storage payload remains storage_isolated.
    out << "  \"storage\": {";
    WriteStorageJson(out, storageProfiles.Isolated);
    out << "},\n";
    out << "  \"storage_profiles\": {";
    out << "\"storage_isolated\":{";
    WriteStorageJson(out, storageProfiles.Isolated);
    out << "},\"storage_e2e_cuda\":{";
    WriteStorageJson(out, storageProfiles.E2eCuda);
    out << "}},\n";

    out << "  \"thread_scaling\": [\n";
    for (size_t i = 0; i < threadScaling.size(); ++i)
    {
        const ThreadScalingResult& tr = threadScaling[i];
        out << "    {\"threads\":" << tr.Threads
            << ",\"trials\":" << tr.Trials
            << ",\"encode_mbps\":" << tr.EncodeMBps
            << ",\"decode_mbps\":" << tr.DecodeMBps
            << ",\"p50_latency_ms\":" << tr.P50LatencyMs
            << ",\"p95_latency_ms\":" << tr.P95LatencyMs
            << ",\"speedup\":" << tr.Speedup
            << ",\"efficiency\":" << tr.Efficiency
            << "}";
        if (i + 1 < threadScaling.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"cuda_perf\": {";
    out << "\"available\":" << (cudaStats.Available ? "true" : "false")
        << ",\"encode_calls\":" << cudaStats.EncodeCalls
        << ",\"decode_calls\":" << cudaStats.DecodeCalls
        << ",\"setup_us\":" << cudaStats.SetupUs
        << ",\"h2d_us\":" << cudaStats.H2dUs
        << ",\"kernel_us\":" << cudaStats.KernelUs
        << ",\"d2h_us\":" << cudaStats.D2hUs
        << ",\"sync_us\":" << cudaStats.SyncUs
        << ",\"h2d_event_us\":" << cudaStats.H2dEventUs
        << ",\"kernel_event_us\":" << cudaStats.KernelEventUs
        << ",\"d2h_event_us\":" << cudaStats.D2hEventUs
        << ",\"e2e_event_us\":" << cudaStats.E2eEventUs
        << ",\"enqueue_us\":" << cudaStats.EnqueueUs
        << ",\"queue_stall_us\":" << cudaStats.QueueStallUs
        << ",\"queue_depth_samples\":" << cudaStats.QueueDepthSamples
        << ",\"queue_depth_total\":" << cudaStats.QueueDepthTotal
        << ",\"device_idle_us\":" << cudaStats.DeviceIdleUs
        << ",\"submit_batches\":" << cudaStats.SubmitBatches
        << ",\"submit_items\":" << cudaStats.SubmitItems
        << ",\"producer_wait_us\":" << cudaStats.ProducerWaitUs
        << ",\"transfer_wait_us\":" << cudaStats.TransferWaitUs
        << ",\"compute_wait_us\":" << cudaStats.ComputeWaitUs
        << ",\"completion_wait_us\":" << cudaStats.CompletionWaitUs
        << ",\"bytes_h2d\":" << cudaStats.BytesH2d
        << ",\"bytes_d2h\":" << cudaStats.BytesD2h
        << ",\"core_offload_calls\":" << cudaStats.CoreOffloadCalls
        << ",\"kernel_share_pct\":" << cudaStats.KernelSharePct
        << ",\"transfer_sync_share_pct\":" << cudaStats.TransferSyncSharePct
        << ",\"avg_bytes_per_call\":" << cudaStats.AvgBytesPerCall
        << ",\"avg_queue_depth\":" << cudaStats.AvgQueueDepth
        << ",\"queue_stall_pct\":" << cudaStats.QueueStallPct
        << ",\"device_idle_pct\":" << cudaStats.DeviceIdlePct
        << ",\"symbols_per_submit\":" << cudaStats.SymbolsPerSubmit
        << ",\"overlap_ratio\":" << cudaStats.OverlapRatio
        << ",\"solver_stage_us\":" << cudaStats.SolverStageUs
        << ",\"solver_pivot_us\":" << cudaStats.SolverPivotUs
        << ",\"solver_eliminate_us\":" << cudaStats.SolverEliminateUs
        << ",\"solver_backsub_us\":" << cudaStats.SolverBackSubUs
        << ",\"solver_verify_passes\":" << cudaStats.SolverVerifyPasses
        << ",\"solver_verify_failures\":" << cudaStats.SolverVerifyFailures
        << ",\"solver_kernel_share_pct\":" << cudaStats.SolverKernelSharePct
        << "}\n";

    out << "}\n";
}

static void PrintSummary(
    const std::vector<CoreCaseResult>& core,
    const std::vector<ParityCaseResult>& parity,
    const std::vector<ChurnCaseResult>& churn,
    const StorageProfilesResult& storageProfiles,
    const std::vector<ThreadScalingResult>& threadScaling,
    const CudaPerfStatsResult& cudaStats)
{
    std::cout << "=== " << AB_VARIANT_NAME << " benchmark summary ===" << std::endl;

    std::cout << "-- Core cases --" << std::endl;
    for (size_t i = 0; i < core.size(); ++i)
    {
        const CoreCaseResult& c = core[i];
        std::cout
            << "N=" << c.N
            << " block=" << c.BlockBytes
            << " loss=" << c.LossPercent << "%"
            << " success=" << c.Successes << "/" << c.Trials
            << " extra=" << std::fixed << std::setprecision(3) << c.AvgExtra
            << " enc_MBps=" << std::setprecision(2) << c.EncodeMBps
            << " dec_MBps=" << c.DecodeMBps
            << " rec_MBps=" << c.RecoverMBps
            << std::endl;
    }

    std::cout << "-- Parity cases --" << std::endl;
    for (size_t i = 0; i < parity.size(); ++i)
    {
        const ParityCaseResult& p = parity[i];
        const double rate = p.Trials > 0 ? static_cast<double>(p.Successes) / p.Trials : 0.0;
        std::cout
            << "drives=" << p.DriveCount
            << " success_rate=" << std::fixed << std::setprecision(3) << rate
            << " avg_needed=" << p.AvgNeeded
            << std::endl;
    }

    std::cout << "-- Churn cases --" << std::endl;
    for (size_t i = 0; i < churn.size(); ++i)
    {
        const ChurnCaseResult& c = churn[i];
        const double rate = c.Trials > 0 ? static_cast<double>(c.Successes) / c.Trials : 0.0;
        std::cout
            << "targets=" << c.TargetCount
            << " success_rate=" << std::fixed << std::setprecision(3) << rate
            << " avg_needed=" << c.AvgNeeded
            << std::endl;
    }

    const StorageCaseResult& storage = storageProfiles.Isolated;
    const StorageCaseResult& storageE2e = storageProfiles.E2eCuda;
    const double storageRate = storage.Trials > 0 ? static_cast<double>(storage.Successes) / storage.Trials : 0.0;
    std::cout
        << "-- Storage case (isolated) -- success_rate=" << std::fixed << std::setprecision(3) << storageRate
        << " write_MBps=" << std::setprecision(2) << storage.WriteMBps
        << " read_MBps=" << storage.ReadMBps
        << " avg_needed=" << storage.AvgNeeded
        << " write_batches=" << storage.WriteBatchCount
        << " read_batches=" << storage.ReadBatchCount
        << " avg_write_batch_bytes=" << storage.AvgWriteBatchBytes
        << " avg_read_batch_bytes=" << storage.AvgReadBatchBytes
        << " rnd_w_p50_ms=" << storage.RandomWriteP50Ms
        << " rnd_w_p95_ms=" << storage.RandomWriteP95Ms
        << " rnd_r_p50_ms=" << storage.RandomReadP50Ms
        << " rnd_r_p95_ms=" << storage.RandomReadP95Ms
        << " eff_read_bytes=" << storage.EffectiveReadBytesBeforeSuccess
        << " completion_ratio=" << storage.WorkCompletionRatio
        << " mode_profile=" << storage.ModeProfile
        << std::endl;
    const double storageE2eRate = storageE2e.Trials > 0 ? static_cast<double>(storageE2e.Successes) / storageE2e.Trials : 0.0;
    std::cout
        << "-- Storage case (e2e_cuda) -- success_rate=" << std::fixed << std::setprecision(3) << storageE2eRate
        << " write_MBps=" << std::setprecision(2) << storageE2e.WriteMBps
        << " read_MBps=" << storageE2e.ReadMBps
        << " avg_needed=" << storageE2e.AvgNeeded
        << " write_batches=" << storageE2e.WriteBatchCount
        << " read_batches=" << storageE2e.ReadBatchCount
        << " avg_write_batch_bytes=" << storageE2e.AvgWriteBatchBytes
        << " avg_read_batch_bytes=" << storageE2e.AvgReadBatchBytes
        << " rnd_w_p50_ms=" << storageE2e.RandomWriteP50Ms
        << " rnd_w_p95_ms=" << storageE2e.RandomWriteP95Ms
        << " rnd_r_p50_ms=" << storageE2e.RandomReadP50Ms
        << " rnd_r_p95_ms=" << storageE2e.RandomReadP95Ms
        << " eff_read_bytes=" << storageE2e.EffectiveReadBytesBeforeSuccess
        << " completion_ratio=" << storageE2e.WorkCompletionRatio
        << " mode_profile=" << storageE2e.ModeProfile
        << std::endl;

    std::cout << "-- Thread scaling --" << std::endl;
    for (size_t i = 0; i < threadScaling.size(); ++i)
    {
        const ThreadScalingResult& tr = threadScaling[i];
        std::cout
            << "threads=" << tr.Threads
            << " enc_MBps=" << std::fixed << std::setprecision(2) << tr.EncodeMBps
            << " dec_MBps=" << tr.DecodeMBps
            << " p50_ms=" << std::setprecision(3) << tr.P50LatencyMs
            << " p95_ms=" << tr.P95LatencyMs
            << " speedup=" << std::setprecision(3) << tr.Speedup
            << " efficiency=" << tr.Efficiency
            << std::endl;
    }

    if (cudaStats.Available) {
        std::cout << "-- CUDA stage timing (usec totals) -- "
                  << "setup=" << cudaStats.SetupUs
                  << " h2d=" << cudaStats.H2dUs
                  << " kernel=" << cudaStats.KernelUs
                  << " d2h=" << cudaStats.D2hUs
                  << " sync=" << cudaStats.SyncUs
                  << " h2d_evt=" << cudaStats.H2dEventUs
                  << " kernel_evt=" << cudaStats.KernelEventUs
                  << " d2h_evt=" << cudaStats.D2hEventUs
                  << " e2e_evt=" << cudaStats.E2eEventUs
                  << " enqueue_us=" << cudaStats.EnqueueUs
                  << " queue_stall_us=" << cudaStats.QueueStallUs
                  << " avg_queue_depth=" << cudaStats.AvgQueueDepth
                  << " symbols_per_submit=" << cudaStats.SymbolsPerSubmit
                  << " overlap_ratio=" << cudaStats.OverlapRatio
                  << " device_idle_us=" << cudaStats.DeviceIdleUs
                  << " kernel_share_pct=" << std::fixed << std::setprecision(2) << cudaStats.KernelSharePct
                  << " transfer_sync_share_pct=" << cudaStats.TransferSyncSharePct
                  << " queue_stall_pct=" << cudaStats.QueueStallPct
                  << " device_idle_pct=" << cudaStats.DeviceIdlePct
                  << " avg_bytes_per_call=" << cudaStats.AvgBytesPerCall
                  << " bytes_h2d=" << cudaStats.BytesH2d
                  << " bytes_d2h=" << cudaStats.BytesD2h
                  << " solver_stage_us=" << cudaStats.SolverStageUs
                  << " solver_pivot_us=" << cudaStats.SolverPivotUs
                  << " solver_eliminate_us=" << cudaStats.SolverEliminateUs
                  << " solver_backsub_us=" << cudaStats.SolverBackSubUs
                  << " solver_verify_passes=" << cudaStats.SolverVerifyPasses
                  << " solver_verify_failures=" << cudaStats.SolverVerifyFailures
                  << " solver_kernel_share_pct=" << cudaStats.SolverKernelSharePct
                  << std::endl;
    }
}

#if defined(_MSC_VER)
template <typename Fn>
static void RunParallelRange(size_t count, Fn&& fn)
{
    concurrency::parallel_for<size_t>(0, count, [&](size_t i) {
        fn(i);
    });
}
#else
template <typename Fn>
static void RunParallelRange(size_t count, Fn&& fn)
{
    for (size_t i = 0; i < count; ++i) {
        fn(i);
    }
}
#endif

} // namespace

int main(int argc, char** argv)
{
    Config cfg;
    if (!ParseArgs(argc, argv, cfg)) {
        std::cerr << "Usage: --output <json> --io-root <dir> --trials <n> --seed <n>" << std::endl;
        return 2;
    }

    const WirehairResult init = wirehair_init();
    if (init != Wirehair_Success) {
        std::cerr << "wirehair_init failed: " << wirehair_result_string(init) << std::endl;
        return 3;
    }

#if defined(AB_USE_CUDA)
    std::cout
        << "[ab_benchmark_cuda] startup "
        << "core_use_cuda_batch=" << (cfg.CoreUseCudaBatch ? 1 : 0)
        << " backend=" << CudaBackendName(cfg.CudaBackendMode)
        << " stress=" << (cfg.Stress ? 1 : 0)
        << " high_memory=" << (cfg.CudaHighMemory ? 1 : 0)
        << " trials=" << cfg.Trials
        << std::endl;

    uint32_t cudaDevices = 0;
    const WirehairResult availability = wirehair_cuda_is_available(&cudaDevices);
    std::cout
        << "[ab_benchmark_cuda] runtime "
        << "cuda_available=" << (availability == Wirehair_Success ? 1 : 0)
        << " device_count=" << cudaDevices
        << std::endl;

    WirehairCudaConfig cudaConfig = {};
    wirehair_cuda_get_default_config(&cudaConfig);
    cudaConfig.backend_mode = cfg.CudaBackendMode;
    cudaConfig.enable_solver_offload = cfg.CoreUseCudaBatch ? 1U : 0U;
    cudaConfig.enable_pipeline_cuda_runtime = 1U;
    cudaConfig.enable_cuda_graphs = cfg.CoreUseCudaBatch ? 1U : 0U;
    cudaConfig.enable_single_api_microbatch = 1U;
    cudaConfig.single_api_microbatch_size = cfg.CudaHighMemory ? 32U : 8U;
    cudaConfig.verification_level = 1U;
    if (availability == Wirehair_Success && cudaDevices > 0) {
        if (cfg.CudaHighMemory) {
            cudaConfig.stream_count = std::min<uint32_t>(16, std::max<uint32_t>(8, cudaDevices * 4));
        } else {
            cudaConfig.stream_count = std::min<uint32_t>(8, std::max<uint32_t>(4, cudaDevices * 3));
        }
    }
    std::cout
        << "[ab_benchmark_cuda] config "
        << "stream_count=" << cudaConfig.stream_count
        << " pinned=" << cudaConfig.use_pinned_memory
        << " solver_offload=" << cudaConfig.enable_solver_offload
        << " pipeline_cuda_runtime=" << cudaConfig.enable_pipeline_cuda_runtime
        << " cuda_graphs=" << cudaConfig.enable_cuda_graphs
        << " single_api_microbatch=" << cudaConfig.enable_single_api_microbatch
        << std::endl;
    if (wirehair_cuda_set_config(&cudaConfig) != Wirehair_Success) {
        std::cerr << "wirehair_cuda_set_config failed" << std::endl;
        return 4;
    }
    wirehair_cuda_reset_perf_stats();
#endif

    const std::string outDir = cfg.OutputPath.substr(0, cfg.OutputPath.find_last_of("/\\"));
    if (!outDir.empty()) {
        EnsureDirectory(outDir);
    }
    EnsureDirectory(cfg.IoRoot);

    std::vector<CoreCaseResult> core;
    static const uint32_t kNsDefault[] = {64, 256, 1024};
    static const uint32_t kBlocksDefault[] = {1300, 4096};
    static const uint32_t kLossesDefault[] = {10, 30};
    static const uint32_t kNsStress[] = {64, 256, 1024, 2048, 4096};
    static const uint32_t kBlocksStress[] = {1300, 4096, 8192};
    static const uint32_t kLossesStress[] = {10, 30, 45};
    const uint32_t* Ns = cfg.Stress ? kNsStress : kNsDefault;
    const uint32_t* blocks = cfg.Stress ? kBlocksStress : kBlocksDefault;
    const uint32_t* losses = cfg.Stress ? kLossesStress : kLossesDefault;
    const size_t nCount = cfg.Stress ? (sizeof(kNsStress) / sizeof(kNsStress[0])) : (sizeof(kNsDefault) / sizeof(kNsDefault[0]));
    const size_t bCount = cfg.Stress ? (sizeof(kBlocksStress) / sizeof(kBlocksStress[0])) : (sizeof(kBlocksDefault) / sizeof(kBlocksDefault[0]));
    const size_t lCount = cfg.Stress ? (sizeof(kLossesStress) / sizeof(kLossesStress[0])) : (sizeof(kLossesDefault) / sizeof(kLossesDefault[0]));
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t> > coreInputs;
    for (size_t ni = 0; ni < nCount; ++ni) {
        for (size_t bi = 0; bi < bCount; ++bi) {
            for (size_t li = 0; li < lCount; ++li) {
                coreInputs.push_back(std::make_tuple(Ns[ni], blocks[bi], losses[li]));
            }
        }
    }
    core.resize(coreInputs.size());
    const bool runCoreSequentialWithProgress = cfg.CoreUseCudaBatch;
    if (runCoreSequentialWithProgress)
    {
        const uint64_t coreStartUs = NowUs();
        std::cout << "[ab_benchmark_cuda] core_cases total=" << coreInputs.size() << " mode=sequential_with_progress" << std::endl;
        for (size_t i = 0; i < coreInputs.size(); ++i)
        {
            const uint32_t n = std::get<0>(coreInputs[i]);
            const uint32_t block = std::get<1>(coreInputs[i]);
            const uint32_t loss = std::get<2>(coreInputs[i]);
            core[i] = RunCoreCase(n, block, loss, cfg.Trials, cfg.Seed + static_cast<uint64_t>(i), cfg.CoreUseCudaBatch);
            const uint64_t elapsedUs = NowUs() - coreStartUs;
            const double avgPerCaseUs = static_cast<double>(elapsedUs) / static_cast<double>(i + 1);
            const double remainingUs = avgPerCaseUs * static_cast<double>(coreInputs.size() - (i + 1));
            std::cout
                << "[ab_benchmark_cuda] core_progress "
                << (i + 1) << "/" << coreInputs.size()
                << " n=" << n
                << " block=" << block
                << " loss=" << loss
                << " elapsed_s=" << std::fixed << std::setprecision(2) << (static_cast<double>(elapsedUs) / 1000000.0)
                << " eta_s=" << std::fixed << std::setprecision(2) << (remainingUs / 1000000.0)
                << std::endl;
        }
    }
    else
    {
        RunParallelRange(coreInputs.size(), [&](size_t i) {
            const uint32_t n = std::get<0>(coreInputs[i]);
            const uint32_t block = std::get<1>(coreInputs[i]);
            const uint32_t loss = std::get<2>(coreInputs[i]);
            core[i] = RunCoreCase(n, block, loss, cfg.Trials, cfg.Seed + static_cast<uint64_t>(i), cfg.CoreUseCudaBatch);
        });
    }

    std::vector<ParityCaseResult> parity;
    const uint32_t driveCounts[] = {4, 8, 16};
    parity.resize(sizeof(driveCounts) / sizeof(driveCounts[0]));
    RunParallelRange(parity.size(), [&](size_t i) {
        parity[i] = RunParityCase(driveCounts[i], cfg.Trials, cfg.Seed + (i * 37ULL));
    });

    std::vector<ChurnCaseResult> churn;
    const uint32_t churnTargets[] = {8, 16, 32};
    churn.resize(sizeof(churnTargets) / sizeof(churnTargets[0]));
    RunParallelRange(churn.size(), [&](size_t i) {
        churn[i] = RunChurnCase(churnTargets[i], cfg.Trials, cfg.Seed + (i * 71ULL));
    });

    StorageProfilesResult storageProfiles = {};
#if defined(AB_USE_CUDA)
    WirehairCudaConfig storageSavedConfig = {};
    storageSavedConfig.struct_bytes = sizeof(WirehairCudaConfig);
    const bool hasCudaConfig = (wirehair_cuda_get_config(&storageSavedConfig) == Wirehair_Success);
    if (hasCudaConfig)
    {
        WirehairCudaConfig storageCpuConfig = storageSavedConfig;
        storageCpuConfig.backend_mode = WirehairCudaBackend_CpuOnly;
        storageCpuConfig.struct_bytes = sizeof(WirehairCudaConfig);
        wirehair_cuda_set_config(&storageCpuConfig);
    }
    storageProfiles.Isolated = RunStorageCase(
        cfg.Trials,
        JoinPath(JoinPath(cfg.IoRoot, AB_VARIANT_NAME), "storage_isolated"),
        cfg.Seed,
        cfg.CudaHighMemory);
    if (hasCudaConfig) {
        wirehair_cuda_set_config(&storageSavedConfig);
        storageProfiles.E2eCuda = RunStorageCase(
            cfg.Trials,
            JoinPath(JoinPath(cfg.IoRoot, AB_VARIANT_NAME), "storage_e2e_cuda"),
            cfg.Seed + 0x9e37ULL,
            cfg.CudaHighMemory);
    } else {
        storageProfiles.E2eCuda = storageProfiles.Isolated;
    }
#else
    storageProfiles.Isolated = RunStorageCase(
        cfg.Trials,
        JoinPath(JoinPath(cfg.IoRoot, AB_VARIANT_NAME), "storage_isolated"),
        cfg.Seed,
        cfg.CudaHighMemory);
    storageProfiles.E2eCuda = storageProfiles.Isolated;
#endif
    std::vector<ThreadScalingResult> scaling;
    const uint32_t requestedThreads[] = {1, 2, 4, 8, 16};
    const uint32_t hw = std::max<uint32_t>(1, static_cast<uint32_t>(std::thread::hardware_concurrency()));
    for (size_t i = 0; i < sizeof(requestedThreads) / sizeof(requestedThreads[0]); ++i)
    {
        const uint32_t tc = std::min(requestedThreads[i], hw);
        if (!scaling.empty() && scaling.back().Threads == tc) {
            continue;
        }
        scaling.push_back(RunThreadScalingCase(tc, cfg.Trials, cfg.Seed + (9000ULL + i)));
    }
    const double baseline = scaling.empty() ? 0.0 : scaling[0].EncodeMBps;
    for (size_t i = 0; i < scaling.size(); ++i)
    {
        scaling[i].Speedup = baseline > 0.0 ? scaling[i].EncodeMBps / baseline : 0.0;
        scaling[i].Efficiency = scaling[i].Threads > 0 ? scaling[i].Speedup / scaling[i].Threads : 0.0;
    }

    CudaPerfStatsResult cudaStats = {};
#if defined(AB_USE_CUDA)
    WirehairCudaPerfStats stats = {};
    stats.struct_bytes = sizeof(WirehairCudaPerfStats);
    if (wirehair_cuda_get_perf_stats(&stats) == Wirehair_Success)
    {
        cudaStats.Available = true;
        cudaStats.EncodeCalls = stats.encode_calls;
        cudaStats.DecodeCalls = stats.decode_calls;
        cudaStats.SetupUs = stats.setup_us;
        cudaStats.H2dUs = stats.h2d_us;
        cudaStats.KernelUs = stats.kernel_us;
        cudaStats.D2hUs = stats.d2h_us;
        cudaStats.SyncUs = stats.sync_us;
        cudaStats.H2dEventUs = stats.h2d_event_us;
        cudaStats.KernelEventUs = stats.kernel_event_us;
        cudaStats.D2hEventUs = stats.d2h_event_us;
        cudaStats.E2eEventUs = stats.e2e_event_us;
        cudaStats.EnqueueUs = stats.enqueue_us;
        cudaStats.QueueStallUs = stats.queue_stall_us;
        cudaStats.QueueDepthSamples = stats.queue_depth_samples;
        cudaStats.QueueDepthTotal = stats.queue_depth_total;
        cudaStats.DeviceIdleUs = stats.device_idle_us;
        cudaStats.SubmitBatches = stats.submit_batches;
        cudaStats.SubmitItems = stats.submit_items;
        cudaStats.ProducerWaitUs = stats.producer_wait_us;
        cudaStats.TransferWaitUs = stats.transfer_wait_us;
        cudaStats.ComputeWaitUs = stats.compute_wait_us;
        cudaStats.CompletionWaitUs = stats.completion_wait_us;
        cudaStats.BytesH2d = stats.bytes_h2d;
        cudaStats.BytesD2h = stats.bytes_d2h;
        cudaStats.SolverStageUs = stats.solver_stage_us;
        cudaStats.SolverPivotUs = stats.solver_pivot_us;
        cudaStats.SolverEliminateUs = stats.solver_eliminate_us;
        cudaStats.SolverBackSubUs = stats.solver_backsub_us;
        cudaStats.SolverVerifyPasses = stats.solver_verify_passes;
        cudaStats.SolverVerifyFailures = stats.solver_verify_failures;
        cudaStats.CoreOffloadCalls = stats.encode_calls + stats.decode_calls;
        const uint64_t transferSyncUs = stats.h2d_event_us + stats.d2h_event_us;
        const uint64_t totalObservedUs = transferSyncUs + stats.kernel_event_us;
        if (totalObservedUs > 0) {
            cudaStats.KernelSharePct = (100.0 * static_cast<double>(stats.kernel_event_us)) / static_cast<double>(totalObservedUs);
            cudaStats.TransferSyncSharePct =
                (100.0 * static_cast<double>(transferSyncUs)) / static_cast<double>(totalObservedUs);
        }
        if (cudaStats.CoreOffloadCalls > 0) {
            cudaStats.AvgBytesPerCall =
                static_cast<double>(stats.bytes_h2d + stats.bytes_d2h) / static_cast<double>(cudaStats.CoreOffloadCalls);
        }
        if (stats.queue_depth_samples > 0) {
            cudaStats.AvgQueueDepth = static_cast<double>(stats.queue_depth_total) / static_cast<double>(stats.queue_depth_samples);
        }
        if (stats.submit_batches > 0) {
            cudaStats.SymbolsPerSubmit = static_cast<double>(stats.submit_items) / static_cast<double>(stats.submit_batches);
        }
        const double transferKernelUs = static_cast<double>(stats.h2d_event_us + stats.kernel_event_us + stats.d2h_event_us);
        const double queueObservedUs = transferKernelUs + static_cast<double>(stats.queue_stall_us);
        if (queueObservedUs > 0.0) {
            cudaStats.QueueStallPct = (100.0 * static_cast<double>(stats.queue_stall_us)) / queueObservedUs;
        }
        const double deviceTimelineUs = static_cast<double>(stats.e2e_event_us) + static_cast<double>(stats.device_idle_us);
        if (deviceTimelineUs > 0.0) {
            cudaStats.DeviceIdlePct = (100.0 * static_cast<double>(stats.device_idle_us)) / deviceTimelineUs;
        }
        const double overlapByTimeline = stats.e2e_event_us > 0
            ? (transferKernelUs / static_cast<double>(stats.e2e_event_us))
            : 0.0;
        const double overlapByQueue = queueObservedUs > 0.0
            ? (transferKernelUs / queueObservedUs)
            : 0.0;
        cudaStats.OverlapRatio = std::max(overlapByTimeline, overlapByQueue);
        if (cudaStats.OverlapRatio > 1.0) {
            cudaStats.OverlapRatio = 1.0;
        }
        if (cudaStats.KernelUs > 0) {
            cudaStats.SolverKernelSharePct =
                (100.0 * static_cast<double>(stats.solver_stage_us)) / static_cast<double>(cudaStats.KernelUs);
        }
    }
#endif

    if (!cfg.Quiet) {
        PrintSummary(core, parity, churn, storageProfiles, scaling, cudaStats);
    }
    WriteJson(cfg.OutputPath, cfg, core, parity, churn, storageProfiles, scaling, cudaStats);
    std::cout << "Wrote benchmark results: " << cfg.OutputPath << std::endl;
    return 0;
}

