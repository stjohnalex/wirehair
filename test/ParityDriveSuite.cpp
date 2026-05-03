#include <wirehair/wirehair.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#include <sys/stat.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

struct SuiteConfig
{
    std::string OutputRoot = "test-output/parity-drive-suite";
    uint32_t MinDrives = 1;
    uint32_t MaxDrives = 32;
    uint64_t SourceBytes = 1024ULL * 1024ULL * 1024ULL; // 1 GiB
    uint32_t BlockBytes = 1024U * 1024U;                // 1 MiB
    uint32_t ExtraSymbols = 8;
    uint32_t SymbolsPerDrive = 0; // 0 = derive from message blocks + extra
    uint32_t SamplesPerSubsetSize = 8;
    uint32_t MaxDecodeSymbolsPerTrial = 0; // 0 = unlimited
    uint32_t MaxTrialsPerN = 128;
    uint64_t MaxEncodedBytesPerN = 2ULL * 1024ULL * 1024ULL * 1024ULL; // 2 GiB guardrail
    uint64_t Seed = 0x6d0f27ad4cc0ffeeULL;
    bool ForceLarge = false;
    bool ReuseExisting = true;
};

struct NResult
{
    uint32_t DriveCount = 0;
    uint32_t MessageBlocks = 0;
    uint32_t TotalSymbols = 0;
    uint32_t Trials = 0;
    uint32_t Successes = 0;
    uint32_t Failures = 0;
    uint32_t MinNeeded = std::numeric_limits<uint32_t>::max();
    uint32_t MaxNeeded = 0;
    double AvgNeeded = 0.0;
    bool SkippedByGuardrail = false;
    std::string SkipReason;
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

    uint32_t Next32()
    {
        return static_cast<uint32_t>(Next() >> 32);
    }

private:
    uint64_t State;
};

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

static bool DirectoryExists(const std::string& path)
{
    struct stat s;
    if (stat(path.c_str(), &s) != 0) {
        return false;
    }
    return (s.st_mode & S_IFDIR) != 0;
}

static bool FileExists(const std::string& path)
{
    std::ifstream probe(path.c_str(), std::ios::binary);
    return probe.good();
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

static uint64_t HashBytes(const uint8_t* data, size_t bytes)
{
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t HashFile(const std::string& path)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        return 0;
    }
    const size_t kChunk = 1 << 20;
    std::vector<uint8_t> buffer(kChunk);
    uint64_t h = 1469598103934665603ULL;
    while (in)
    {
        in.read(reinterpret_cast<char*>(&buffer[0]), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = in.gcount();
        for (std::streamsize i = 0; i < got; ++i) {
            h ^= buffer[static_cast<size_t>(i)];
            h *= 1099511628211ULL;
        }
    }
    return h;
}

static bool ParseUnsigned(const std::string& value, uint64_t& out)
{
    char* endPtr = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &endPtr, 10);
    if (!endPtr || *endPtr != '\0') {
        return false;
    }
    out = static_cast<uint64_t>(parsed);
    return true;
}

static bool ParseArgs(int argc, char** argv, SuiteConfig& config)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--force-large") {
            config.ForceLarge = true;
            continue;
        }
        if (arg == "--no-reuse") {
            config.ReuseExisting = false;
            continue;
        }

        if (arg.rfind("--", 0) != 0) {
            std::cerr << "Unknown positional argument: " << arg << std::endl;
            return false;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << std::endl;
            return false;
        }

        const std::string value = argv[++i];
        uint64_t parsed = 0;
        if (arg == "--output-root") {
            config.OutputRoot = value;
            continue;
        }
        if (!ParseUnsigned(value, parsed)) {
            std::cerr << "Invalid numeric value for " << arg << ": " << value << std::endl;
            return false;
        }

        if (arg == "--min-drives") {
            config.MinDrives = static_cast<uint32_t>(parsed);
        } else if (arg == "--max-drives") {
            config.MaxDrives = static_cast<uint32_t>(parsed);
        } else if (arg == "--source-bytes") {
            config.SourceBytes = parsed;
        } else if (arg == "--block-bytes") {
            config.BlockBytes = static_cast<uint32_t>(parsed);
        } else if (arg == "--extra-symbols") {
            config.ExtraSymbols = static_cast<uint32_t>(parsed);
        } else if (arg == "--symbols-per-drive") {
            config.SymbolsPerDrive = static_cast<uint32_t>(parsed);
        } else if (arg == "--samples-per-subset-size") {
            config.SamplesPerSubsetSize = static_cast<uint32_t>(parsed);
        } else if (arg == "--max-decode-symbols-per-trial") {
            config.MaxDecodeSymbolsPerTrial = static_cast<uint32_t>(parsed);
        } else if (arg == "--max-trials-per-n") {
            config.MaxTrialsPerN = static_cast<uint32_t>(parsed);
        } else if (arg == "--max-encoded-bytes-per-n") {
            config.MaxEncodedBytesPerN = parsed;
        } else if (arg == "--seed") {
            config.Seed = parsed;
        } else {
            std::cerr << "Unknown flag: " << arg << std::endl;
            return false;
        }
    }

    if (config.MinDrives < 1 || config.MaxDrives < config.MinDrives) {
        std::cerr << "Invalid drive range" << std::endl;
        return false;
    }
    if (config.BlockBytes < 1 || config.SourceBytes < 2) {
        std::cerr << "Invalid source/block configuration" << std::endl;
        return false;
    }
    return true;
}

static bool GenerateSourceFile(const std::string& path, uint64_t bytes, uint64_t seed)
{
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to create source file: " << path << std::endl;
        return false;
    }

    XorShift64 prng(seed ^ 0x41544f4d5f534f55ULL);
    const size_t chunkSize = 1 << 20;
    std::vector<uint8_t> chunk(chunkSize);

    uint64_t remaining = bytes;
    while (remaining > 0)
    {
        const size_t writeNow = static_cast<size_t>(std::min<uint64_t>(remaining, chunk.size()));
        for (size_t i = 0; i < writeNow; ++i) {
            chunk[i] = static_cast<uint8_t>(prng.Next32() & 0xff);
        }
        out.write(reinterpret_cast<const char*>(&chunk[0]), static_cast<std::streamsize>(writeNow));
        if (!out) {
            std::cerr << "Failed writing source chunk" << std::endl;
            return false;
        }
        remaining -= writeNow;
    }
    return true;
}

static bool ReadFileBytes(const std::string& path, std::vector<uint8_t>& out)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) {
        return false;
    }
    in.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (!out.empty()) {
        in.read(reinterpret_cast<char*>(&out[0]), static_cast<std::streamsize>(out.size()));
        if (!in) {
            return false;
        }
    }
    return true;
}

static bool WriteBytes(const std::string& path, const uint8_t* data, size_t bytes)
{
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(out);
}

static std::string ZeroPad(uint32_t x, uint32_t width)
{
    std::ostringstream oss;
    oss << std::setw(static_cast<int>(width)) << std::setfill('0') << x;
    return oss.str();
}

static uint32_t ComputeTotalSymbols(uint32_t messageBlocks, uint32_t driveCount, const SuiteConfig& config)
{
    if (config.SymbolsPerDrive > 0) {
        return config.SymbolsPerDrive * driveCount;
    }
    return messageBlocks + config.ExtraSymbols;
}

static std::vector<std::vector<uint32_t> > SampleCombinations(
    uint32_t driveCount,
    uint32_t subsetSize,
    uint32_t maxSamples,
    XorShift64& prng)
{
    std::vector<std::vector<uint32_t> > combos;
    if (subsetSize == 0 || subsetSize > driveCount || maxSamples == 0) {
        return combos;
    }

    std::set<std::string> seen;
    uint32_t attempts = 0;
    const uint32_t maxAttempts = std::max<uint32_t>(maxSamples * 25, 50);
    while (combos.size() < maxSamples && attempts++ < maxAttempts)
    {
        std::set<uint32_t> picked;
        while (picked.size() < subsetSize) {
            const uint32_t v = static_cast<uint32_t>(prng.Next() % driveCount);
            picked.insert(v);
        }
        std::vector<uint32_t> combo(picked.begin(), picked.end());
        std::ostringstream sig;
        for (size_t i = 0; i < combo.size(); ++i) {
            if (i > 0) {
                sig << ",";
            }
            sig << combo[i];
        }
        if (seen.insert(sig.str()).second) {
            combos.push_back(combo);
        }
    }

    return combos;
}

static bool DecodeCombination(
    const std::vector<uint32_t>& selectedDrives,
    const std::vector<std::vector<uint32_t> >& symbolsByDrive,
    const std::unordered_map<uint32_t, std::string>& pathBySymbol,
    const std::vector<uint8_t>& sourceData,
    uint64_t sourceHash,
    uint64_t messageBytes,
    uint32_t blockBytes,
    uint32_t maxDecodeSymbols,
    uint32_t& neededOut)
{
    neededOut = 0;
    WirehairCodec decoder = wirehair_decoder_create(nullptr, messageBytes, blockBytes);
    if (!decoder) {
        return false;
    }

    std::set<uint32_t> uniqueSymbols;
    for (size_t i = 0; i < selectedDrives.size(); ++i) {
        const uint32_t drive = selectedDrives[i];
        if (drive >= symbolsByDrive.size()) {
            continue;
        }
        for (size_t j = 0; j < symbolsByDrive[drive].size(); ++j) {
            uniqueSymbols.insert(symbolsByDrive[drive][j]);
        }
    }

    std::vector<uint8_t> symbolData;
    bool decoded = false;
    for (std::set<uint32_t>::const_iterator it = uniqueSymbols.begin();
         it != uniqueSymbols.end();
         ++it)
    {
        const uint32_t symbolId = *it;
        std::unordered_map<uint32_t, std::string>::const_iterator pathIt = pathBySymbol.find(symbolId);
        if (pathIt == pathBySymbol.end()) {
            continue;
        }
        if (!ReadFileBytes(pathIt->second, symbolData)) {
            wirehair_free(decoder);
            return false;
        }
        ++neededOut;

        const WirehairResult decodeResult = wirehair_decode(
            decoder,
            symbolId,
            &symbolData[0],
            static_cast<uint32_t>(symbolData.size()));

        if (decodeResult == Wirehair_Success) {
            decoded = true;
            break;
        }
        if (decodeResult != Wirehair_NeedMore) {
            wirehair_free(decoder);
            return false;
        }
        if (maxDecodeSymbols > 0 && neededOut >= maxDecodeSymbols) {
            break;
        }
    }

    if (!decoded) {
        wirehair_free(decoder);
        return false;
    }

    std::vector<uint8_t> recovered(sourceData.size());
    const WirehairResult recoverResult = wirehair_recover(decoder, &recovered[0], messageBytes);
    wirehair_free(decoder);
    if (recoverResult != Wirehair_Success) {
        return false;
    }
    const uint64_t recoveredHash = HashBytes(&recovered[0], recovered.size());
    return recoveredHash == sourceHash;
}

static void WriteNReport(const std::string& reportPath, const NResult& result)
{
    std::ofstream out(reportPath.c_str(), std::ios::trunc);
    if (!out) {
        return;
    }
    out << "{\n";
    out << "  \"drive_count\": " << result.DriveCount << ",\n";
    out << "  \"message_blocks\": " << result.MessageBlocks << ",\n";
    out << "  \"total_symbols\": " << result.TotalSymbols << ",\n";
    out << "  \"trials\": " << result.Trials << ",\n";
    out << "  \"successes\": " << result.Successes << ",\n";
    out << "  \"failures\": " << result.Failures << ",\n";
    out << "  \"avg_needed\": " << std::fixed << std::setprecision(2) << result.AvgNeeded << ",\n";
    out << "  \"min_needed\": " << ((result.MinNeeded == std::numeric_limits<uint32_t>::max()) ? 0 : result.MinNeeded) << ",\n";
    out << "  \"max_needed\": " << result.MaxNeeded << ",\n";
    out << "  \"skipped_by_guardrail\": " << (result.SkippedByGuardrail ? "true" : "false") << ",\n";
    out << "  \"skip_reason\": \"" << result.SkipReason << "\"\n";
    out << "}\n";
}

static void WriteSummaryReport(const std::string& reportPath, const std::vector<NResult>& allResults)
{
    std::ofstream out(reportPath.c_str(), std::ios::trunc);
    if (!out) {
        return;
    }
    out << "{\n  \"results\": [\n";
    for (size_t i = 0; i < allResults.size(); ++i)
    {
        const NResult& r = allResults[i];
        out << "    {\n";
        out << "      \"drive_count\": " << r.DriveCount << ",\n";
        out << "      \"trials\": " << r.Trials << ",\n";
        out << "      \"successes\": " << r.Successes << ",\n";
        out << "      \"failures\": " << r.Failures << ",\n";
        out << "      \"skipped\": " << (r.SkippedByGuardrail ? "true" : "false") << "\n";
        out << "    }";
        if (i + 1 < allResults.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv)
{
    SuiteConfig config;
    if (!ParseArgs(argc, argv, config)) {
        return 2;
    }

    const WirehairResult initResult = wirehair_init();
    if (initResult != Wirehair_Success) {
        std::cerr << "wirehair_init failed: " << wirehair_result_string(initResult) << std::endl;
        return 3;
    }

    if (!EnsureDirectory(config.OutputRoot)) {
        std::cerr << "Failed to create output root: " << config.OutputRoot << std::endl;
        return 4;
    }

    const std::string sourceDir = JoinPath(config.OutputRoot, "source");
    if (!EnsureDirectory(sourceDir)) {
        std::cerr << "Failed to create source dir" << std::endl;
        return 5;
    }
    const std::string sourcePath = JoinPath(sourceDir, "source-1gb.bin");

    if (!config.ReuseExisting || !FileExists(sourcePath))
    {
        std::cout << "Generating source file: " << sourcePath << " (" << config.SourceBytes << " bytes)" << std::endl;
        if (!GenerateSourceFile(sourcePath, config.SourceBytes, config.Seed ^ 0x12345ULL)) {
            return 6;
        }
    }

    std::vector<uint8_t> sourceData;
    if (!ReadFileBytes(sourcePath, sourceData)) {
        std::cerr << "Failed to read source file: " << sourcePath << std::endl;
        return 7;
    }
    if (sourceData.size() != config.SourceBytes) {
        std::cerr << "Unexpected source file size: " << sourceData.size() << std::endl;
        return 8;
    }
    const uint64_t sourceHash = HashBytes(&sourceData[0], sourceData.size());
    std::cout << "Source hash (FNV64): " << sourceHash << std::endl;

    const uint32_t messageBlocks = static_cast<uint32_t>(
        (config.SourceBytes + config.BlockBytes - 1) / config.BlockBytes);
    if (messageBlocks < 2) {
        std::cerr << "Message block count must be at least 2; adjust block/source size." << std::endl;
        return 9;
    }

    std::vector<NResult> allResults;
    XorShift64 comboPrng(config.Seed ^ 0xabcdefULL);

    for (uint32_t driveCount = config.MinDrives; driveCount <= config.MaxDrives; ++driveCount)
    {
        NResult result;
        result.DriveCount = driveCount;
        result.MessageBlocks = messageBlocks;
        result.TotalSymbols = ComputeTotalSymbols(messageBlocks, driveCount, config);

        const uint64_t estimatedBytes = static_cast<uint64_t>(result.TotalSymbols) * config.BlockBytes;
        if (!config.ForceLarge && estimatedBytes > config.MaxEncodedBytesPerN)
        {
            result.SkippedByGuardrail = true;
            std::ostringstream reason;
            reason << "estimated encoded bytes " << estimatedBytes
                   << " exceeds guardrail " << config.MaxEncodedBytesPerN;
            result.SkipReason = reason.str();
            std::cout << "Skipping N=" << driveCount << ": " << result.SkipReason << std::endl;
            allResults.push_back(result);
            continue;
        }

        const std::string nRoot = JoinPath(config.OutputRoot, "N-" + ZeroPad(driveCount, 4));
        const std::string reportDir = JoinPath(nRoot, "reports");
        if (!EnsureDirectory(nRoot) || !EnsureDirectory(reportDir)) {
            std::cerr << "Failed to create directory tree for N=" << driveCount << std::endl;
            return 10;
        }

        std::vector<std::string> driveDirs(driveCount);
        for (uint32_t d = 0; d < driveCount; ++d)
        {
            driveDirs[d] = JoinPath(nRoot, "drive-" + ZeroPad(d + 1, 4));
            if (!EnsureDirectory(driveDirs[d])) {
                std::cerr << "Failed to create drive folder: " << driveDirs[d] << std::endl;
                return 11;
            }
        }

        WirehairCodec encoder = wirehair_encoder_create(
            nullptr,
            &sourceData[0],
            static_cast<uint64_t>(sourceData.size()),
            config.BlockBytes);
        if (!encoder) {
            std::cerr << "Failed to create encoder for drive count " << driveCount << std::endl;
            return 12;
        }

        std::vector<std::vector<uint32_t> > symbolsByDrive(driveCount);
        std::unordered_map<uint32_t, std::string> pathBySymbol;
        std::vector<uint8_t> symbolBuffer(config.BlockBytes);

        const uint32_t firstParityBlockId = messageBlocks;
        for (uint32_t i = 0; i < result.TotalSymbols; ++i)
        {
            const uint32_t symbolId = firstParityBlockId + i;
            const uint32_t driveIndex = i % driveCount;
            uint32_t written = 0;
            const WirehairResult encodeResult = wirehair_encode(
                encoder,
                symbolId,
                &symbolBuffer[0],
                config.BlockBytes,
                &written);
            if (encodeResult != Wirehair_Success || written != config.BlockBytes) {
                wirehair_free(encoder);
                std::cerr << "Encode failed for symbol " << symbolId
                          << " at N=" << driveCount << std::endl;
                return 13;
            }

            const std::string symbolFile = JoinPath(
                driveDirs[driveIndex],
                "symbol-" + ZeroPad(symbolId, 10) + ".bin");
            if (!WriteBytes(symbolFile, &symbolBuffer[0], written)) {
                wirehair_free(encoder);
                std::cerr << "Failed writing symbol file: " << symbolFile << std::endl;
                return 14;
            }

            symbolsByDrive[driveIndex].push_back(symbolId);
            pathBySymbol[symbolId] = symbolFile;
        }
        wirehair_free(encoder);

        for (uint32_t subsetSize = 1; subsetSize <= driveCount; ++subsetSize)
        {
            std::vector<std::vector<uint32_t> > combos = SampleCombinations(
                driveCount,
                subsetSize,
                config.SamplesPerSubsetSize,
                comboPrng);
            for (size_t i = 0; i < combos.size(); ++i)
            {
                if (result.Trials >= config.MaxTrialsPerN) {
                    break;
                }
                uint32_t needed = 0;
                const bool ok = DecodeCombination(
                    combos[i],
                    symbolsByDrive,
                    pathBySymbol,
                    sourceData,
                    sourceHash,
                    static_cast<uint64_t>(sourceData.size()),
                    config.BlockBytes,
                    config.MaxDecodeSymbolsPerTrial,
                    needed);

                ++result.Trials;
                if (ok) {
                    ++result.Successes;
                    result.MinNeeded = std::min(result.MinNeeded, needed);
                    result.MaxNeeded = std::max(result.MaxNeeded, needed);
                    result.AvgNeeded += needed;
                } else {
                    ++result.Failures;
                }
            }
            if (result.Trials >= config.MaxTrialsPerN) {
                break;
            }
        }

        if (result.Successes > 0) {
            result.AvgNeeded /= static_cast<double>(result.Successes);
        } else {
            result.MinNeeded = 0;
        }

        const std::string nReport = JoinPath(reportDir, "summary.json");
        WriteNReport(nReport, result);
        std::cout << "N=" << driveCount
                  << " symbols=" << result.TotalSymbols
                  << " trials=" << result.Trials
                  << " success=" << result.Successes
                  << " fail=" << result.Failures
                  << std::endl;

        allResults.push_back(result);
    }

    const std::string summaryPath = JoinPath(config.OutputRoot, "summary.json");
    WriteSummaryReport(summaryPath, allResults);
    std::cout << "Wrote summary: " << summaryPath << std::endl;
    return 0;
}
