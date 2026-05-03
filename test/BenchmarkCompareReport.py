#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List, Tuple


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct_delta(base: float, value: float) -> float:
    if base == 0.0:
        return 0.0
    return ((value - base) / base) * 100.0


def key_core(item: Dict) -> Tuple[int, int, int]:
    return int(item["n"]), int(item["block_bytes"]), int(item["loss_percent"])


def key_parity(item: Dict) -> Tuple[int]:
    return (int(item["drive_count"]),)


def key_churn(item: Dict) -> Tuple[int]:
    return (int(item["target_count"]),)


def compare_section(base_items: List[Dict], variant_items: List[Dict], key_fn, metric_fields: List[str]) -> List[Dict]:
    by_base = {key_fn(x): x for x in base_items}
    by_variant = {key_fn(x): x for x in variant_items}
    keys = sorted(set(by_base.keys()) & set(by_variant.keys()))
    rows = []
    for k in keys:
        b = by_base[k]
        v = by_variant[k]
        metric_data = {}
        for metric in metric_fields:
            base_val = float(b.get(metric, 0.0))
            var_val = float(v.get(metric, 0.0))
            metric_data[metric] = {
                "control": base_val,
                "variant": var_val,
                "delta_pct": pct_delta(base_val, var_val),
            }
        rows.append({"key": list(k), "metrics": metric_data})
    return rows


WIN_METRIC_KEYS = [
    "core_encode_mbps_avg_delta_pct",
    "core_decode_mbps_avg_delta_pct",
    "core_recover_mbps_avg_delta_pct",
    "storage_write_mbps_delta_pct",
    "storage_read_mbps_delta_pct",
    "thread_scaling_encode_avg_delta_pct",
]


def avg_core_field(items: List[Dict], field: str) -> float:
    if not items:
        return 0.0
    return sum(float(x.get(field, 0.0)) for x in items) / len(items)


def collect_throughput_snapshot(label: str, data: Dict) -> Dict:
    core = data.get("core", [])
    ts = data.get("thread_scaling", [])
    st = data.get("storage", {})
    max_enc_ts = max((float(x.get("encode_mbps", 0.0)) for x in ts), default=0.0)
    max_threads_row = max(ts, key=lambda x: float(x.get("encode_mbps", 0.0))) if ts else {}
    return {
        "label": label,
        "core_encode_mbps": avg_core_field(core, "encode_mbps"),
        "core_decode_mbps": avg_core_field(core, "decode_mbps"),
        "core_recover_mbps": avg_core_field(core, "recover_mbps"),
        "storage_write_mbps": float(st.get("write_mbps", 0.0)),
        "storage_read_mbps": float(st.get("read_mbps", 0.0)),
        "thread_peak_encode_mbps": float(max_enc_ts),
        "thread_peak_threads": float(max_threads_row.get("threads", 0)) if max_threads_row else 0.0,
    }


def print_absolute_leaderboard(entries: List[Dict]) -> None:
    """Side-by-side throughput (same trial seed/config across variants)."""
    print("")
    print("=== ABSOLUTE THROUGHPUT (mean core cases; higher MB/s is faster) ===")
    hdr = f"{'variant':<18} {'core_enc':>10} {'core_dec':>10} {'core_rec':>10} {'stor_w':>9} {'stor_r':>9} {'thr_enc*':>10}"
    print(hdr)
    print("-" * len(hdr))
    ranked = sorted(entries, key=lambda r: r["core_encode_mbps"], reverse=True)
    for r in ranked:
        thr_note = f"{r['thread_peak_encode_mbps']:.1f}@{int(r['thread_peak_threads'])}t"
        print(
            f"{r['label']:<18} "
            f"{r['core_encode_mbps']:>10.2f} "
            f"{r['core_decode_mbps']:>10.2f} "
            f"{r['core_recover_mbps']:>10.2f} "
            f"{r['storage_write_mbps']:>9.2f} "
            f"{r['storage_read_mbps']:>9.2f} "
            f"{thr_note:>10}"
        )
    print("* thr_enc = peak pipeline encode MB/s in thread-scaling sweep.")
    print("")
    print("Fastest by metric (variant name):")
    metrics = [
        ("mean core encode MB/s", "core_encode_mbps"),
        ("mean core decode MB/s", "core_decode_mbps"),
        ("mean core recover MB/s", "core_recover_mbps"),
        ("storage write MB/s", "storage_write_mbps"),
        ("storage read MB/s", "storage_read_mbps"),
        ("peak thread-scale encode MB/s", "thread_peak_encode_mbps"),
    ]
    for title, key in metrics:
        best = max(entries, key=lambda r: r[key])
        print(f"  {title:<34} {best['label']} ({best[key]:.2f})")


def summarize_delta(rows: List[Dict], metric: str) -> float:
    values = [float(row["metrics"][metric]["delta_pct"]) for row in rows if metric in row["metrics"]]
    if not values:
        return 0.0
    return sum(values) / len(values)


def parse_variant_arg(raw: str) -> Tuple[str, str]:
    parts = raw.split("=", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid --variant value: {raw}. Expected label=path")
    return parts[0], parts[1]


def build_variant_report(control: Dict, variant_label: str, variant_data: Dict) -> Dict:
    core_rows = compare_section(
        control.get("core", []),
        variant_data.get("core", []),
        key_core,
        ["encode_mbps", "decode_mbps", "recover_mbps", "avg_extra", "avg_create_us"],
    )
    parity_rows = compare_section(
        control.get("parity", []),
        variant_data.get("parity", []),
        key_parity,
        ["success_rate", "avg_needed"],
    )
    churn_rows = compare_section(
        control.get("churn", []),
        variant_data.get("churn", []),
        key_churn,
        ["success_rate", "avg_needed"],
    )

    c_storage = control.get("storage", {})
    v_storage = variant_data.get("storage", {})
    storage = {
        "write_mbps": {
            "control": float(c_storage.get("write_mbps", 0.0)),
            "variant": float(v_storage.get("write_mbps", 0.0)),
        },
        "read_mbps": {
            "control": float(c_storage.get("read_mbps", 0.0)),
            "variant": float(v_storage.get("read_mbps", 0.0)),
        },
        "success_rate": {
            "control": float(c_storage.get("success_rate", 0.0)),
            "variant": float(v_storage.get("success_rate", 0.0)),
        },
    }
    for metric in storage.values():
        metric["delta_pct"] = pct_delta(metric["control"], metric["variant"])

    summary = {
        "core_encode_mbps_avg_delta_pct": summarize_delta(core_rows, "encode_mbps"),
        "core_decode_mbps_avg_delta_pct": summarize_delta(core_rows, "decode_mbps"),
        "core_recover_mbps_avg_delta_pct": summarize_delta(core_rows, "recover_mbps"),
        "parity_success_rate_avg_delta_pct": summarize_delta(parity_rows, "success_rate"),
        "churn_success_rate_avg_delta_pct": summarize_delta(churn_rows, "success_rate"),
        "storage_write_mbps_delta_pct": storage["write_mbps"]["delta_pct"],
        "storage_read_mbps_delta_pct": storage["read_mbps"]["delta_pct"],
    }

    control_scaling = {int(x.get("threads", 0)): x for x in control.get("thread_scaling", [])}
    variant_scaling = {int(x.get("threads", 0)): x for x in variant_data.get("thread_scaling", [])}
    common_threads = sorted(set(control_scaling.keys()) & set(variant_scaling.keys()))
    scaling_rows = []
    for threads in common_threads:
        c = control_scaling[threads]
        v = variant_scaling[threads]
        scaling_rows.append(
            {
                "threads": threads,
                "encode_delta_pct": pct_delta(float(c.get("encode_mbps", 0.0)), float(v.get("encode_mbps", 0.0))),
                "decode_delta_pct": pct_delta(float(c.get("decode_mbps", 0.0)), float(v.get("decode_mbps", 0.0))),
                "efficiency_delta_pct": pct_delta(float(c.get("efficiency", 0.0)), float(v.get("efficiency", 0.0))),
            }
        )
    summary["thread_scaling_encode_avg_delta_pct"] = (
        sum(x["encode_delta_pct"] for x in scaling_rows) / len(scaling_rows) if scaling_rows else 0.0
    )
    summary["thread_scaling_decode_avg_delta_pct"] = (
        sum(x["decode_delta_pct"] for x in scaling_rows) / len(scaling_rows) if scaling_rows else 0.0
    )

    win_score = 0
    for key in WIN_METRIC_KEYS:
        if summary[key] > 0:
            win_score += 1

    return {
        "label": variant_label,
        "variant_name": variant_data.get("variant", variant_label),
        "core": core_rows,
        "parity": parity_rows,
        "churn": churn_rows,
        "storage": storage,
        "thread_scaling": scaling_rows,
        "summary": summary,
        "win_score": win_score,
        "cuda_perf": variant_data.get("cuda_perf", {}),
    }


def print_variant_summary(report: Dict) -> None:
    summary = report["summary"]
    nwin = len(WIN_METRIC_KEYS)
    print(f"[{report['label']}] wins={report['win_score']}/{nwin} (categories faster than control)")
    print(
        f"  core encode={summary['core_encode_mbps_avg_delta_pct']:.2f}% "
        f"decode={summary['core_decode_mbps_avg_delta_pct']:.2f}% "
        f"recover={summary['core_recover_mbps_avg_delta_pct']:.2f}%"
    )
    print(
        f"  storage write={summary['storage_write_mbps_delta_pct']:.2f}% "
        f"read={summary['storage_read_mbps_delta_pct']:.2f}%"
    )
    print(
        f"  parity success={summary['parity_success_rate_avg_delta_pct']:.2f}% "
        f"churn success={summary['churn_success_rate_avg_delta_pct']:.2f}%"
    )
    print(
        f"  thread scaling encode={summary['thread_scaling_encode_avg_delta_pct']:.2f}% "
        f"decode={summary['thread_scaling_decode_avg_delta_pct']:.2f}%"
    )
    cuda_perf = report.get("cuda_perf", {})
    if cuda_perf.get("available"):
        print(
            "  cuda stages(us) "
            f"setup={int(cuda_perf.get('setup_us', 0))} "
            f"h2d={int(cuda_perf.get('h2d_us', 0))} "
            f"kernel={int(cuda_perf.get('kernel_us', 0))} "
            f"d2h={int(cuda_perf.get('d2h_us', 0))} "
            f"sync={int(cuda_perf.get('sync_us', 0))}"
        )
        print(
            "  cuda stages_event(us) "
            f"h2d_evt={int(cuda_perf.get('h2d_event_us', 0))} "
            f"kernel_evt={int(cuda_perf.get('kernel_event_us', 0))} "
            f"d2h_evt={int(cuda_perf.get('d2h_event_us', 0))} "
            f"e2e_evt={int(cuda_perf.get('e2e_event_us', 0))}"
        )
        print(
            "  cuda utilization "
            f"offload_calls={int(cuda_perf.get('core_offload_calls', 0))} "
            f"kernel_share_pct={float(cuda_perf.get('kernel_share_pct', 0.0)):.2f} "
            f"transfer_sync_share_pct={float(cuda_perf.get('transfer_sync_share_pct', 0.0)):.2f} "
            f"avg_bytes_per_call={float(cuda_perf.get('avg_bytes_per_call', 0.0)):.2f}"
        )


def summarize_filtered_core_delta(report: Dict, metric: str, predicate) -> float:
    rows = report.get("core", [])
    values = []
    for row in rows:
        key = row.get("key", [])
        if len(key) < 3:
            continue
        n = int(key[0])
        block = int(key[1])
        loss = int(key[2])
        if not predicate(n, block, loss):
            continue
        metric_payload = row.get("metrics", {}).get(metric, {})
        values.append(float(metric_payload.get("delta_pct", 0.0)))
    if not values:
        return 0.0
    return sum(values) / len(values)


def min_filtered_core_delta(report: Dict, metric: str, predicate) -> float:
    rows = report.get("core", [])
    values = []
    for row in rows:
        key = row.get("key", [])
        if len(key) < 3:
            continue
        n = int(key[0])
        block = int(key[1])
        loss = int(key[2])
        if not predicate(n, block, loss):
            continue
        metric_payload = row.get("metrics", {}).get(metric, {})
        values.append(float(metric_payload.get("delta_pct", 0.0)))
    if not values:
        return 0.0
    return min(values)


def print_final_results_table(control_label: str, entries: List[Dict], variant_reports: List[Dict]) -> None:
    report_by_label = {x["label"]: x for x in variant_reports}
    ordered = sorted(entries, key=lambda r: r["core_encode_mbps"], reverse=True)

    print("")
    print("=== FINAL COMPARISON TABLE ===")
    header = (
        f"{'variant':<18} {'core_enc':>9} {'core_dec':>9} {'core_rec':>9} "
        f"{'stor_w':>8} {'stor_r':>8} {'thr_peak':>10} {'enc_vs_ctl':>11} {'dec_vs_ctl':>11}"
    )
    print(header)
    print("-" * len(header))

    for row in ordered:
        label = row["label"]
        if label == control_label:
            enc_delta = 0.0
            dec_delta = 0.0
        else:
            report = report_by_label.get(label, {})
            summary = report.get("summary", {})
            enc_delta = float(summary.get("core_encode_mbps_avg_delta_pct", 0.0))
            dec_delta = float(summary.get("core_decode_mbps_avg_delta_pct", 0.0))
        thr = f"{row['thread_peak_encode_mbps']:.1f}@{int(row['thread_peak_threads'])}t"
        print(
            f"{label:<18} "
            f"{row['core_encode_mbps']:>9.2f} "
            f"{row['core_decode_mbps']:>9.2f} "
            f"{row['core_recover_mbps']:>9.2f} "
            f"{row['storage_write_mbps']:>8.2f} "
            f"{row['storage_read_mbps']:>8.2f} "
            f"{thr:>10} "
            f"{enc_delta:>10.2f}% "
            f"{dec_delta:>10.2f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare multiple benchmark variants against a control.")
    parser.add_argument("--control", required=True, help="Control benchmark JSON")
    parser.add_argument("--variant", action="append", default=[], help="Variant benchmark in label=path form. Repeatable.")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    parser.add_argument(
        "--require-cuda-core-encode-delta",
        type=float,
        default=None,
        help="If set, fail unless CUDA core encode avg delta vs control >= this percentage.",
    )
    parser.add_argument(
        "--require-cuda-stress-encode-delta",
        type=float,
        default=None,
        help="If set, fail unless CUDA stress-cluster encode delta (N>=1024, block>=4096, loss>=30) meets this percentage.",
    )
    parser.add_argument(
        "--require-cuda-stress-decode-delta",
        type=float,
        default=None,
        help="If set, fail unless CUDA stress-cluster decode delta (N>=1024, block>=4096, loss>=30) meets this percentage.",
    )
    parser.add_argument(
        "--require-cuda-stress-encode-min-delta",
        type=float,
        default=None,
        help="If set, fail unless the minimum CUDA stress encode delta for any stress case meets this percentage.",
    )
    parser.add_argument(
        "--require-cuda-stress-decode-min-delta",
        type=float,
        default=None,
        help="If set, fail unless the minimum CUDA stress decode delta for any stress case meets this percentage.",
    )
    parser.add_argument(
        "--require-cuda-kernel-share-pct",
        type=float,
        default=None,
        help="If set, fail unless CUDA kernel_share_pct >= this value.",
    )
    parser.add_argument(
        "--require-cuda-core-offload-calls",
        type=float,
        default=None,
        help="If set, fail unless CUDA core_offload_calls >= this value.",
    )
    args = parser.parse_args()

    if not args.variant:
        raise SystemExit("At least one --variant label=path argument is required.")

    control = load_json(args.control)
    variant_reports: List[Dict] = []
    leaderboard_rows: List[Dict] = [
        collect_throughput_snapshot(str(control.get("variant", "control")), control),
    ]
    for variant_arg in args.variant:
        label, path = parse_variant_arg(variant_arg)
        variant_data = load_json(path)
        variant_reports.append(build_variant_report(control, label, variant_data))
        leaderboard_rows.append(collect_throughput_snapshot(label, variant_data))

    ranking = sorted(variant_reports, key=lambda x: (x["win_score"], x["summary"]["core_decode_mbps_avg_delta_pct"]), reverse=True)
    overall_winner = ranking[0]["label"] if ranking else "none"

    output = {
        "control_variant": control.get("variant", "control"),
        "variants": variant_reports,
        "ranking": [entry["label"] for entry in ranking],
        "overall_winner": overall_winner,
    }

    print_absolute_leaderboard(leaderboard_rows)

    print("=== RELATIVE RESULTS (delta % vs control; positive = faster for MB/s) ===")
    print(f"Control: {output['control_variant']}")
    print(f"Heuristic overall winner (variant rows): {overall_winner}")
    print("")
    print("Variant summaries (delta vs control):")
    for entry in ranking:
        print_variant_summary(entry)

    print_final_results_table(str(control.get("variant", "control")), leaderboard_rows, variant_reports)

    if args.require_cuda_core_encode_delta is not None:
        cuda_entry = next((x for x in variant_reports if x.get("label") == "cuda"), None)
        if cuda_entry is None:
            print("CUDA gate requested but no cuda variant was supplied.")
            return 3
        core_delta = float(cuda_entry["summary"]["core_encode_mbps_avg_delta_pct"])
        if core_delta < args.require_cuda_core_encode_delta:
            print(
                f"CUDA core encode gate failed: {core_delta:.2f}% "
                f"< required {args.require_cuda_core_encode_delta:.2f}%"
            )
            return 3

    if (args.require_cuda_stress_encode_delta is not None or
            args.require_cuda_stress_decode_delta is not None or
            args.require_cuda_stress_encode_min_delta is not None or
            args.require_cuda_stress_decode_min_delta is not None):
        cuda_entry = next((x for x in variant_reports if x.get("label") == "cuda"), None)
        if cuda_entry is None:
            print("CUDA stress gate requested but no cuda variant was supplied.")
            return 3
        predicate = lambda n, block, loss: n >= 1024 and block >= 4096 and loss >= 30
        stress_encode = summarize_filtered_core_delta(
            cuda_entry,
            "encode_mbps",
            predicate,
        )
        stress_decode = summarize_filtered_core_delta(
            cuda_entry,
            "decode_mbps",
            predicate,
        )
        stress_encode_min = min_filtered_core_delta(cuda_entry, "encode_mbps", predicate)
        stress_decode_min = min_filtered_core_delta(cuda_entry, "decode_mbps", predicate)
        if args.require_cuda_stress_encode_delta is not None and stress_encode < args.require_cuda_stress_encode_delta:
            print(
                f"CUDA stress encode gate failed: {stress_encode:.2f}% "
                f"< required {args.require_cuda_stress_encode_delta:.2f}%"
            )
            return 3
        if args.require_cuda_stress_decode_delta is not None and stress_decode < args.require_cuda_stress_decode_delta:
            print(
                f"CUDA stress decode gate failed: {stress_decode:.2f}% "
                f"< required {args.require_cuda_stress_decode_delta:.2f}%"
            )
            return 3
        if args.require_cuda_stress_encode_min_delta is not None and stress_encode_min < args.require_cuda_stress_encode_min_delta:
            print(
                f"CUDA stress encode minimum gate failed: {stress_encode_min:.2f}% "
                f"< required {args.require_cuda_stress_encode_min_delta:.2f}%"
            )
            return 3
        if args.require_cuda_stress_decode_min_delta is not None and stress_decode_min < args.require_cuda_stress_decode_min_delta:
            print(
                f"CUDA stress decode minimum gate failed: {stress_decode_min:.2f}% "
                f"< required {args.require_cuda_stress_decode_min_delta:.2f}%"
            )
            return 3

    if args.require_cuda_kernel_share_pct is not None or args.require_cuda_core_offload_calls is not None:
        cuda_entry = next((x for x in variant_reports if x.get("label") == "cuda"), None)
        if cuda_entry is None:
            print("CUDA utilization gate requested but no cuda variant was supplied.")
            return 3
        cuda_perf = cuda_entry.get("cuda_perf", {})
        kernel_share = float(cuda_perf.get("kernel_share_pct", 0.0))
        offload_calls = float(cuda_perf.get("core_offload_calls", 0.0))
        if args.require_cuda_kernel_share_pct is not None and kernel_share < args.require_cuda_kernel_share_pct:
            print(
                f"CUDA kernel share gate failed: {kernel_share:.2f}% "
                f"< required {args.require_cuda_kernel_share_pct:.2f}%"
            )
            return 3
        if args.require_cuda_core_offload_calls is not None and offload_calls < args.require_cuda_core_offload_calls:
            print(
                f"CUDA offload call gate failed: {offload_calls:.0f} "
                f"< required {args.require_cuda_core_offload_calls:.0f}"
            )
            return 3

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote comparison report: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

