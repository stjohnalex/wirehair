#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <process.h>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

struct Config
{
    std::string OutputDir = "ab-bench";
    uint32_t Trials = 0;
    uint32_t MinTrials = 8;
    uint32_t CampaignSeconds = 900;
    uint64_t Seed = 0x1234abcd9876ULL;
    /** Pass --require-cuda-* thresholds to BenchmarkCompareReport.py (exit 3 if CUDA loses). */
    bool EnforceCudaGates = false;
};

struct VariantSpec
{
    std::string Label;
    std::string ExePath;
    std::string OutputJson;
};

static std::string JoinPath(const std::string& a, const std::string& b)
{
    if (a.empty()) {
        return b;
    }
    if (a[a.size() - 1] == '/' || a[a.size() - 1] == '\\') {
        return a + b;
    }
    if (a.find('\\') != std::string::npos) {
        return a + "\\" + b;
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
    if (argc >= 2)
    {
        const std::string a = argv[1];
        if (a == "--help" || a == "-h") {
            return false;
        }
    }
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--enforce-cuda-gates") {
            cfg.EnforceCudaGates = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << std::endl;
            return false;
        }
        const std::string val = argv[++i];
        char* end = nullptr;
        if (arg == "--output-dir") {
            cfg.OutputDir = val;
        } else if (arg == "--trials") {
            const unsigned long long parsed = std::strtoull(val.c_str(), &end, 10);
            if (!end || *end != '\0') {
                return false;
            }
            cfg.Trials = static_cast<uint32_t>(parsed);
        } else if (arg == "--min-trials") {
            const unsigned long long parsed = std::strtoull(val.c_str(), &end, 10);
            if (!end || *end != '\0') {
                return false;
            }
            cfg.MinTrials = static_cast<uint32_t>(parsed);
        } else if (arg == "--campaign-seconds") {
            const unsigned long long parsed = std::strtoull(val.c_str(), &end, 10);
            if (!end || *end != '\0') {
                return false;
            }
            cfg.CampaignSeconds = static_cast<uint32_t>(parsed);
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
    return true;
}

static int RunProcess(const std::string& exe, const std::vector<std::string>& args, bool searchPath)
{
    std::cout << "[ab_benchmark_compare] " << exe;
    for (size_t i = 0; i < args.size(); ++i) {
        std::cout << " " << args[i];
    }
    std::cout << std::endl;

#if defined(_WIN32)
    auto quoteArg = [](const std::string& in) -> std::string
    {
        if (in.find_first_of(" \t\"") == std::string::npos) {
            return in;
        }
        std::string out = "\"";
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i] == '"') {
                out += "\\\"";
            } else {
                out += in[i];
            }
        }
        out += "\"";
        return out;
    };

    std::string commandLine = quoteArg(exe);
    for (size_t i = 0; i < args.size(); ++i) {
        commandLine += " ";
        commandLine += quoteArg(args[i]);
    }
    std::vector<char> commandBuffer(commandLine.begin(), commandLine.end());
    commandBuffer.push_back('\0');

    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    const char* applicationName = searchPath ? nullptr : exe.c_str();
    const BOOL created = CreateProcessA(
        applicationName,
        commandBuffer.data(),
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        nullptr,
        &si,
        &pi);
    if (!created) {
        return -1;
    }

    const auto start = std::chrono::steady_clock::now();
    auto heartbeatAt = start + std::chrono::seconds(5);
    for (;;)
    {
        const DWORD waitCode = WaitForSingleObject(pi.hProcess, 1000);
        if (waitCode == WAIT_OBJECT_0) {
            break;
        }
        if (waitCode != WAIT_TIMEOUT) {
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            return -1;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= heartbeatAt)
        {
            const auto elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            const int minutes = static_cast<int>(elapsedSec / 60);
            const int seconds = static_cast<int>(elapsedSec % 60);
            std::cout << "  ... still running (" << std::setw(2) << std::setfill('0') << minutes
                      << ":" << std::setw(2) << std::setfill('0') << seconds << " elapsed)" << std::endl;
            heartbeatAt = now + std::chrono::seconds(5);
        }
    }

    DWORD exitCode = 1;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(exitCode);
#else
    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char*>(exe.c_str()));
    for (size_t i = 0; i < args.size(); ++i) {
        argv.push_back(const_cast<char*>(args[i].c_str()));
    }
    argv.push_back(nullptr);
    const intptr_t exitCode = searchPath
        ? _spawnvp(_P_WAIT, exe.c_str(), &argv[0])
        : _spawnv(_P_WAIT, exe.c_str(), &argv[0]);
    return static_cast<int>(exitCode);
#endif
}

#if defined(_WIN32)
static bool FileExists(const std::string& path)
{
    return _access(path.c_str(), 0) == 0;
}
#else
static bool FileExists(const std::string& path)
{
    std::ifstream f(path.c_str());
    return f.good();
}
#endif

static std::string DirectoryOfExecutable(char** argv)
{
#if defined(_WIN32)
    char buf[MAX_PATH];
    const DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH)
    {
        std::string full(buf, buf + len);
        const size_t slash = full.find_last_of("\\/");
        if (slash != std::string::npos) {
            return full.substr(0, slash);
        }
    }
#endif
    const std::string exePath = argv[0];
    const size_t slash = exePath.find_last_of("/\\");
    return (slash == std::string::npos) ? std::string(".") : exePath.substr(0, slash);
}

static std::string ResolveCompareReportScript(const std::string& exeDir)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base(exeDir);
    const std::vector<fs::path> candidates = {
        base / ".." / ".." / "test" / "BenchmarkCompareReport.py",
        base / ".." / "test" / "BenchmarkCompareReport.py",
        base / "test" / "BenchmarkCompareReport.py",
        base / "BenchmarkCompareReport.py",
    };
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        const fs::path canon = fs::weakly_canonical(candidates[i], ec);
        if (ec) {
            continue;
        }
        if (fs::exists(canon)) {
            return canon.string();
        }
    }
    return std::string();
}

static std::string ResolveBenchmarkExe(const std::string& exeDir, const std::string& stem)
{
#if defined(_DEBUG)
    const char* primarySuffix = "_Debug";
    const char* secondarySuffix = "_Release";
#else
    const char* primarySuffix = "_Release";
    const char* secondarySuffix = "_Debug";
#endif
#if defined(_WIN32)
    const std::string ext = ".exe";
#else
    const std::string ext = "";
#endif
    const std::vector<std::string> candidates = {
        JoinPath(exeDir, stem + primarySuffix + ext),
        JoinPath(exeDir, stem + secondarySuffix + ext),
        JoinPath(exeDir, stem + ext),
    };
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        if (FileExists(candidates[i])) {
            return candidates[i];
        }
    }
    return std::string();
}

static double RunVariantAndMeasureSeconds(
    const VariantSpec& variant,
    const std::string& ioRoot,
    uint32_t trials,
    uint64_t seed)
{
    const std::vector<std::string> args = {
        "--output",
        variant.OutputJson,
        "--io-root",
        ioRoot,
        "--quiet",
        "1",
        "--stress",
        "1",
        "--trials",
        std::to_string(trials),
        "--seed",
        std::to_string(seed),
    };

    const auto begin = std::chrono::steady_clock::now();
    const int code = RunProcess(variant.ExePath, args, false);
    const auto end = std::chrono::steady_clock::now();
    if (code != 0) {
        return -1.0;
    }
    return std::chrono::duration_cast<std::chrono::duration<double> >(end - begin).count();
}

} // namespace

int main(int argc, char** argv)
{
    Config cfg;
    if (!ParseArgs(argc, argv, cfg)) {
        std::cerr
            << "Wirehair benchmark suite — runs all variant exes and prints a console comparison.\n"
            << "Usage:\n"
            << "  ab_benchmark_compare [--output-dir <dir>] [--trials <n>] [--min-trials <n>]\n"
            << "                      [--campaign-seconds <sec>] [--seed <u64>] [--enforce-cuda-gates]\n"
            << "  Runs variants in --quiet 1 --stress 1 mode by default.\n"
            << "  --trials 0 calibrates trial count from --campaign-seconds (default 900).\n"
            << "  --enforce-cuda-gates  optional CI mode: Python exits 3 if CUDA is below thresholds vs control.\n"
            << "Requires sibling exes: ab_benchmark_control[_Release|_Debug], ab_benchmark_simd_avx2*,\n"
            << "ab_benchmark_simd_avx2_lto*, optional ab_benchmark_cuda*.\n";
        return 2;
    }

    const std::string exeDir = DirectoryOfExecutable(argv);

    const std::string controlExe = ResolveBenchmarkExe(exeDir, "ab_benchmark_control");
    const std::string simdAvx2Exe = ResolveBenchmarkExe(exeDir, "ab_benchmark_simd_avx2");
    const std::string simdAvx2LtoExe = ResolveBenchmarkExe(exeDir, "ab_benchmark_simd_avx2_lto");
    const std::string cudaExe = ResolveBenchmarkExe(exeDir, "ab_benchmark_cuda");
    const std::string scriptPath = ResolveCompareReportScript(exeDir);

    std::cout << "=== Wirehair benchmark suite (all variants) ===" << std::endl;
    if (controlExe.empty() || simdAvx2Exe.empty() || simdAvx2LtoExe.empty())
    {
        std::cerr << "Missing required benchmark executable(s) next to:\n  " << exeDir << "\n";
        if (controlExe.empty()) {
            std::cerr << "  - ab_benchmark_control_Release/Debug (.exe)\n";
        }
        if (simdAvx2Exe.empty()) {
            std::cerr << "  - ab_benchmark_simd_avx2_Release/Debug (.exe)\n";
        }
        if (simdAvx2LtoExe.empty()) {
            std::cerr << "  - ab_benchmark_simd_avx2_lto_Release/Debug (.exe)\n";
        }
        return 11;
    }
    std::cout << "Using control: " << controlExe << std::endl;
    std::cout << "Using simd_avx2: " << simdAvx2Exe << std::endl;
    std::cout << "Using simd_avx2_lto: " << simdAvx2LtoExe << std::endl;

    if (!EnsureDirectory(cfg.OutputDir)) {
        std::cerr << "Failed to create output dir: " << cfg.OutputDir << std::endl;
        return 3;
    }
    const std::string ioRoot = JoinPath(cfg.OutputDir, "io");
    if (!EnsureDirectory(ioRoot)) {
        std::cerr << "Failed to create io dir: " << ioRoot << std::endl;
        return 4;
    }

    const std::string controlJson = JoinPath(cfg.OutputDir, "control.json");
    const std::string simdAvx2Json = JoinPath(cfg.OutputDir, "simd_avx2.json");
    const std::string simdAvx2LtoJson = JoinPath(cfg.OutputDir, "simd_avx2_lto.json");
    const std::string cudaJson = JoinPath(cfg.OutputDir, "cuda.json");
    const std::string compareJson = JoinPath(cfg.OutputDir, "compare.json");
    const std::string calibrationDir = JoinPath(cfg.OutputDir, "calibration");
    EnsureDirectory(calibrationDir);
    const std::string calibrationIoRoot = JoinPath(ioRoot, "calibration");
    EnsureDirectory(calibrationIoRoot);

    std::vector<VariantSpec> variants;
    variants.push_back({"control", controlExe, controlJson});
    variants.push_back({"simd_avx2", simdAvx2Exe, simdAvx2Json});
    variants.push_back({"simd_avx2_lto", simdAvx2LtoExe, simdAvx2LtoJson});
    const bool cudaAvailable = !cudaExe.empty();
    if (cudaAvailable) {
        variants.push_back({"cuda", cudaExe, cudaJson});
        std::cout << "Using cuda: " << cudaExe << std::endl;
    } else {
        std::cout << "Skipping cuda (executable not found next to suite)." << std::endl;
    }

    uint32_t chosenTrials = cfg.Trials;
    if (chosenTrials == 0)
    {
        std::cout << "Calibrating trials for ~" << cfg.CampaignSeconds << " seconds campaign..." << std::endl;
        double calibrationTotalSeconds = 0.0;
        for (size_t i = 0; i < variants.size(); ++i)
        {
            VariantSpec calibrationVariant = variants[i];
            calibrationVariant.OutputJson = JoinPath(calibrationDir, variants[i].Label + ".json");
            const double seconds = RunVariantAndMeasureSeconds(
                calibrationVariant,
                calibrationIoRoot,
                1,
                cfg.Seed + static_cast<uint64_t>(i));
            if (seconds <= 0.0) {
                return 5;
            }
            calibrationTotalSeconds += seconds;
            std::cout << "  " << variants[i].Label << ": " << std::fixed << std::setprecision(2)
                      << seconds << "s for 1 trial" << std::endl;
        }
        if (calibrationTotalSeconds <= 0.0) {
            chosenTrials = cfg.MinTrials;
        } else {
            chosenTrials = static_cast<uint32_t>(std::floor(static_cast<double>(cfg.CampaignSeconds) / calibrationTotalSeconds));
            if (chosenTrials < cfg.MinTrials) {
                chosenTrials = cfg.MinTrials;
            }
        }
        std::cout << "Selected trials per variant: " << chosenTrials << std::endl;
    } else {
        std::cout << "Using explicit trials per variant: " << chosenTrials << std::endl;
    }

    for (size_t i = 0; i < variants.size(); ++i)
    {
        const double seconds = RunVariantAndMeasureSeconds(
            variants[i],
            ioRoot,
            chosenTrials,
            cfg.Seed + static_cast<uint64_t>(i));
        if (seconds <= 0.0) {
            return static_cast<int>(6 + i);
        }
        std::cout << variants[i].Label << " completed in " << std::fixed << std::setprecision(2)
                  << seconds << "s" << std::endl;
    }

    std::vector<std::string> pyArgs = {
        scriptPath,
        "--control",
        controlJson,
        "--variant",
        "simd_avx2=" + simdAvx2Json,
        "--variant",
        "simd_avx2_lto=" + simdAvx2LtoJson,
    };
    if (cudaAvailable) {
        pyArgs.push_back("--variant");
        pyArgs.push_back("cuda=" + cudaJson);
        if (cfg.EnforceCudaGates)
        {
            pyArgs.push_back("--require-cuda-core-encode-delta");
            pyArgs.push_back("5.0");
            pyArgs.push_back("--require-cuda-stress-encode-delta");
            pyArgs.push_back("5.0");
            pyArgs.push_back("--require-cuda-stress-decode-delta");
            pyArgs.push_back("3.0");
            pyArgs.push_back("--require-cuda-stress-encode-min-delta");
            pyArgs.push_back("-8.0");
            pyArgs.push_back("--require-cuda-stress-decode-min-delta");
            pyArgs.push_back("-10.0");
            pyArgs.push_back("--require-cuda-kernel-share-pct");
            pyArgs.push_back("18.0");
            pyArgs.push_back("--require-cuda-core-offload-calls");
            pyArgs.push_back("24");
        }
    }
    pyArgs.push_back("--out");
    pyArgs.push_back(compareJson);

    if (scriptPath.empty())
    {
        std::cerr << "Could not find BenchmarkCompareReport.py (searched near exe dir):\n  "
                  << exeDir << std::endl;
        return 10;
    }

    int pyResult = RunProcess("python", pyArgs, true);
    if (pyResult != 0)
    {
        pyResult = RunProcess("python3", pyArgs, true);
    }
    if (pyResult != 0)
    {
        std::vector<std::string> pyLauncherArgs;
        pyLauncherArgs.push_back("-3");
        pyLauncherArgs.insert(pyLauncherArgs.end(), pyArgs.begin(), pyArgs.end());
        pyResult = RunProcess("py", pyLauncherArgs, true);
    }
    if (pyResult != 0)
    {
        std::cerr << "Python comparison step failed (exit " << pyResult << ").\n";
        if (pyResult == 3 && cfg.EnforceCudaGates) {
            std::cerr << "BenchmarkCompareReport.py exit 3: CUDA performance gate failed (--enforce-cuda-gates).\n"
                      << "Omit --enforce-cuda-gates for a report-only comparison.\n";
        }
        std::cerr << "Otherwise ensure Python 3 is on PATH ('python', 'python3', or 'py'), or run manually:\n  "
                  << "python \"" << scriptPath << "\" --control \"" << controlJson << "\" ...\n";
        return 9;
    }

    std::cout << "Benchmark race report written to: " << compareJson << std::endl;
    return 0;
}

