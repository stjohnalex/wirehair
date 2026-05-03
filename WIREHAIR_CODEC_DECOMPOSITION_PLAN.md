# Wirehair Codec Decomposition Plan

This document defines an incremental decomposition plan for `WirehairCodec.cpp` and
`WirehairCodec.h` to improve maintainability while preserving behavior and API
compatibility.

## Goals

- Keep the public C API unchanged (`include/wirehair/wirehair.h`, `wirehair.cpp`).
- Preserve bit-for-bit codec behavior for existing seeds and test vectors.
- Reduce merge conflicts and review scope by splitting responsibilities.
- Keep each new translation unit focused on one stage of the codec pipeline.

## Non-Goals

- No algorithm changes in the decomposition work itself.
- No throughput tuning mixed into decomposition commits.
- No changes to wire format or block-id semantics.

## Target Layout

- `WirehairCodec.h`
  - Keep only `Codec` public/protected declarations required across translation units.
- `WirehairCodecInternal.h` (new)
  - Private helpers, constants, and compact internal structs.
- `WirehairCodec.Matrix.cpp` (new)
  - Matrix setup, row/column metadata, and initialization helpers.
- `WirehairCodec.Solve.cpp` (new)
  - Peeling/compression/triangle/substitution stages.
- `WirehairCodec.EncodeDecode.cpp` (new)
  - `EncodeFeed`, `Encode`, `DecodeFeed`, and row-generation orchestration.
- `WirehairCodec.Recovery.cpp` (new)
  - Recovery block generation and output reconstruction helpers.
- `WirehairCodec.Diagnostics.cpp` (optional)
  - Debug dump/tracing utilities gated by existing debug macros.

## Phase Plan

1. **Header Hygiene**
   - Move private-only declarations from `WirehairCodec.h` into
     `WirehairCodecInternal.h`.
   - Keep symbol visibility unchanged and avoid inline behavior changes.

2. **Extraction By Responsibility**
   - Move coherent groups of methods into one new `.cpp` at a time.
   - After each extraction, run `unit_test` and A/B benchmark smoke tests.

3. **Build Wiring**
   - Update `CMakeLists.txt` and `msvc/wirehair.vcxproj` source lists.
   - Ensure Debug/Release build parity for CMake + `.sln` flows.

4. **Safety Rails**
   - Add compile-time checks for accidental ODR duplication.
   - Keep helper functions `static`/anonymous-namespace scoped when local.

5. **Final Cleanup**
   - Remove dead declarations from legacy files.
   - Re-run full tests and compare benchmark JSON schema/output compatibility.

## Validation Checklist Per Phase

- Build passes:
  - `wirehair.sln` (`Debug|x64` and `Release|x64`)
  - CMake configure + build for `wirehair` and `unit_test`
- Correctness:
  - `unit_test` completes without new failures.
  - `parity_drive_suite` smoke run still succeeds.
- Performance guardrail:
  - A/B benchmark compare run completes and JSON format remains unchanged.

