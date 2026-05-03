# Dynamic iSCSI Fountain Streaming Project

## Goal

Design and implement a Wirehair-based, read-only streaming architecture for `1-32` mounted iSCSI targets where drives can be removed and added during active streaming, optimized for best-effort throughput.

## Scenario

- Highly dynamic target sets (mount/unmount while streaming).
- Redundancy is adapted online rather than fixed parity groups.
- Transport/control plane is symbol-centric (`generation_id`, `block_id`).

## Scope (Phase 1)

- Read-only streaming pipeline first.
- Parallel symbol fetch from all healthy targets.
- Generation-local decode with cancellation of stragglers once decode succeeds.
- Dedupe by `(generation_id, block_id)`.
- Telemetry for overhead (`needed_symbols - N`) and target health.

## Architecture

1. **TargetMembershipManager**
   - Tracks mounted iSCSI targets and emits join/leave snapshots.
2. **ParallelSymbolScheduler**
   - Pulls symbols concurrently across all healthy targets.
   - Uses adaptive per-target weighting.
3. **DecodeQuorumAssembler**
   - Feeds symbols to Wirehair decoder per generation.
   - Completes generation on `Wirehair_Success`.
4. **ManifestIndex**
   - Maps generations to symbol availability by target.

## Suggested Defaults

- Generation size: `2 MiB`
- Block size: `1200-1400` bytes
- In-flight generations: `4-16`
- Initial extra-symbol margin: `N + 4`, then adaptive based on observed tails

## Validation Plan

- Churn tests with random target attach/detach during streams.
- Duplicate symbol delivery tests.
- Late-join target contribution tests.
- Metrics:
  - Throughput vs active target count
  - Generation decode latency
  - Average/p95 overhead beyond `N`

## Risks

- Non-MDS behavior can produce overhead tails; mitigate with adaptive headroom.
- Metadata drift under churn; keep manifest updates idempotent and monotonic.
- Output ordering complexity; use generation reorder buffer.

