#include <wirehair/wirehair.h>
#include <wirehair/wirehair_cuda.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
    uint64_t BytesH2d = 0;
    uint64_t BytesD2h = 0;
    uint64_t CoreOffloadCalls = 0;
    double KernelSharePct = 0.0;
    double TransferSyncSharePct = 0.0;
    double AvgBytesPerCall = 0.0;
};

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
            const uint32_t startBlockId = N + 1;
            const uint32_t symbolCount = std::max<uint32_t>(N + 64, N + (N / 2));
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

            std::vector<uint32_t> selectedIds;
            selectedIds.reserve(symbolCount);
            std::vector<uint8_t> decodeInput;
            decodeInput.reserve(static_cast<size_t>(symbolCount) * blockBytes);
            std::vector<uint32_t> decodeLens;
            decodeLens.reserve(symbolCount);
            for (uint32_t i = 0; i < symbolCount; ++i)
            {
                if (prng.NextBounded(100) < lossPercent) {
                    continue;
                }
                selectedIds.push_back(startBlockId + i);
                const uint8_t* src = &batchEncoded[static_cast<size_t>(i) * blockBytes];
                decodeInput.insert(decodeInput.end(), src, src + blockBytes);
                decodeLens.push_back(batchLens[i]);
            }
            if (selectedIds.empty()) {
                continue;
            }

            const uint64_t selectedCount = static_cast<uint64_t>(selectedIds.size());
            encodeBytes += static_cast<uint64_t>(symbolCount) * blockBytes;
            decodeBytes += selectedCount * blockBytes;
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
            if (decResult == Wirehair_Success &&
                HashBytes(&recovered[0], recovered.size()) == HashBytes(&message[0], message.size()))
            {
                ++r.Successes;
                const uint32_t needed = static_cast<uint32_t>(selectedIds.size());
                if (needed > N) {
                    extraSum += (needed - N);
                }
                recoverBytes += messageBytes;
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

static StorageCaseResult RunStorageCase(uint32_t trials, const std::string& root, uint64_t seed)
{
    StorageCaseResult r;
    r.Trials = trials;
    XorShift64 prng(seed ^ 0xa5a5a5ULL);
    const uint32_t blockBytes = 64 * 1024;
    const uint32_t N = 128;
    const uint64_t messageBytes = static_cast<uint64_t>(N) * blockBytes - 1;
    const uint32_t totalSymbols = N + 32;
    std::vector<uint8_t> message(static_cast<size_t>(messageBytes));
    FillRandom(message, prng);

    uint64_t writeUs = 0;
    uint64_t readUs = 0;
    uint64_t bytesW = 0;
    uint64_t bytesR = 0;
    uint64_t neededSum = 0;

    EnsureDirectory(root);
    for (uint32_t t = 0; t < trials; ++t)
    {
        const std::string trialDir = JoinPath(root, "trial-" + std::to_string(t));
        EnsureDirectory(trialDir);

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

        std::vector<std::string> symbolPaths;
        std::vector<uint32_t> symbolIds;
        std::vector<uint8_t> symbol(blockBytes);

        for (uint32_t i = 0; i < totalSymbols; ++i)
        {
            const uint32_t blockId = N + i;
            uint32_t writeLen = 0;
            if (wirehair_encode(encoder, blockId, &symbol[0], blockBytes, &writeLen) != Wirehair_Success) {
                continue;
            }
            const std::string path = JoinPath(trialDir, "sym-" + std::to_string(blockId) + ".bin");
            const uint64_t w0 = NowUs();
            std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(&symbol[0]), writeLen);
            const uint64_t w1 = NowUs();
            if (out) {
                symbolPaths.push_back(path);
                symbolIds.push_back(blockId);
                writeUs += (w1 - w0);
                bytesW += writeLen;
            }
        }

        uint32_t needed = 0;
        std::vector<uint8_t> recovered(message.size());
        bool ok = false;
        for (size_t i = 0; i < symbolPaths.size(); ++i)
        {
            std::ifstream in(symbolPaths[i].c_str(), std::ios::binary);
            if (!in) {
                continue;
            }
            const uint64_t r0 = NowUs();
            in.read(reinterpret_cast<char*>(&symbol[0]), blockBytes);
            const std::streamsize got = in.gcount();
            const uint64_t r1 = NowUs();
            if (got <= 0) {
                continue;
            }
            readUs += (r1 - r0);
            bytesR += static_cast<uint64_t>(got);
            ++needed;

#if AB_IS_OPTIMIZED
            const uint32_t blockId = symbolIds[i];
#else
            const std::string file = symbolPaths[i];
            const size_t p0 = file.find("sym-");
            const size_t p1 = file.find(".bin");
            if (p0 == std::string::npos || p1 == std::string::npos || p1 <= p0 + 4) {
                continue;
            }
            const uint32_t blockId = static_cast<uint32_t>(std::strtoul(file.substr(p0 + 4, p1 - (p0 + 4)).c_str(), nullptr, 10));
#endif
            const WirehairResult dr = wirehair_decode(decoder, blockId, &symbol[0], static_cast<uint32_t>(got));
            if (dr == Wirehair_Success) {
                if (wirehair_recover(decoder, &recovered[0], messageBytes) == Wirehair_Success &&
                    HashBytes(&recovered[0], recovered.size()) == HashBytes(&message[0], message.size()))
                {
                    ok = true;
                }
                break;
            }
            if (dr != Wirehair_NeedMore) {
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

    r.WriteMBps = writeUs > 0 ? static_cast<double>(bytesW) / writeUs : 0.0;
    r.ReadMBps = readUs > 0 ? static_cast<double>(bytesR) / readUs : 0.0;
    r.AvgNeeded = r.Successes > 0 ? static_cast<double>(neededSum) / r.Successes : 0.0;
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

static void WriteJson(
    const std::string& path,
    const Config& cfg,
    const std::vector<CoreCaseResult>& core,
    const std::vector<ParityCaseResult>& parity,
    const std::vector<ChurnCaseResult>& churn,
    const StorageCaseResult& storage,
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

    out << "  \"storage\": {";
    out << "\"trials\":" << storage.Trials
        << ",\"successes\":" << storage.Successes
        << ",\"success_rate\":" << (storage.Trials > 0 ? static_cast<double>(storage.Successes) / storage.Trials : 0.0)
        << ",\"write_mbps\":" << storage.WriteMBps
        << ",\"read_mbps\":" << storage.ReadMBps
        << ",\"avg_needed\":" << storage.AvgNeeded
        << "},\n";

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
        << ",\"bytes_h2d\":" << cudaStats.BytesH2d
        << ",\"bytes_d2h\":" << cudaStats.BytesD2h
        << ",\"core_offload_calls\":" << cudaStats.CoreOffloadCalls
        << ",\"kernel_share_pct\":" << cudaStats.KernelSharePct
        << ",\"transfer_sync_share_pct\":" << cudaStats.TransferSyncSharePct
        << ",\"avg_bytes_per_call\":" << cudaStats.AvgBytesPerCall
        << "}\n";

    out << "}\n";
}

static void PrintSummary(
    const std::vector<CoreCaseResult>& core,
    const std::vector<ParityCaseResult>& parity,
    const std::vector<ChurnCaseResult>& churn,
    const StorageCaseResult& storage,
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

    const double storageRate = storage.Trials > 0 ? static_cast<double>(storage.Successes) / storage.Trials : 0.0;
    std::cout
        << "-- Storage case -- success_rate=" << std::fixed << std::setprecision(3) << storageRate
        << " write_MBps=" << std::setprecision(2) << storage.WriteMBps
        << " read_MBps=" << storage.ReadMBps
        << " avg_needed=" << storage.AvgNeeded
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
                  << " kernel_share_pct=" << std::fixed << std::setprecision(2) << cudaStats.KernelSharePct
                  << " transfer_sync_share_pct=" << cudaStats.TransferSyncSharePct
                  << " avg_bytes_per_call=" << cudaStats.AvgBytesPerCall
                  << " bytes_h2d=" << cudaStats.BytesH2d
                  << " bytes_d2h=" << cudaStats.BytesD2h
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
    WirehairCudaConfig cudaConfig = {};
    wirehair_cuda_get_default_config(&cudaConfig);
    cudaConfig.backend_mode = cfg.CudaBackendMode;
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
    RunParallelRange(coreInputs.size(), [&](size_t i) {
        const uint32_t n = std::get<0>(coreInputs[i]);
        const uint32_t block = std::get<1>(coreInputs[i]);
        const uint32_t loss = std::get<2>(coreInputs[i]);
        core[i] = RunCoreCase(n, block, loss, cfg.Trials, cfg.Seed + static_cast<uint64_t>(i), cfg.CoreUseCudaBatch);
    });

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

    const StorageCaseResult storage = RunStorageCase(cfg.Trials, JoinPath(cfg.IoRoot, AB_VARIANT_NAME), cfg.Seed);
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
        cudaStats.BytesH2d = stats.bytes_h2d;
        cudaStats.BytesD2h = stats.bytes_d2h;
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
    }
#endif

    if (!cfg.Quiet) {
        PrintSummary(core, parity, churn, storage, scaling, cudaStats);
    }
    WriteJson(cfg.OutputPath, cfg, core, parity, churn, storage, scaling, cudaStats);
    std::cout << "Wrote benchmark results: " << cfg.OutputPath << std::endl;
    return 0;
}

