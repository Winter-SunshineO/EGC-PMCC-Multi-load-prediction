import argparse
import json
from pathlib import Path

import numpy as np


LOAD_NAMES = ("electricity", "cooling", "heating")
D1_GATE_HORIZONS = (24, 96)


def load_arrays(result_dir):
    result_dir = Path(result_dir)
    y_true = np.load(result_dir / "all_y_true.npy").astype(np.float64)
    y_pred = np.load(result_dir / "all_predict_value.npy").astype(np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch in {result_dir}: {y_true.shape} != {y_pred.shape}")
    return y_true, y_pred


def mape_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true) * 100.0


def bootstrap_delta(delta, n_bootstrap, seed, chunk_size=1000):
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    filled = 0
    n = delta.shape[0]
    while filled < n_bootstrap:
        chunk = min(chunk_size, n_bootstrap - filled)
        indices = rng.integers(0, n, size=(chunk, n))
        samples[filled:filled + chunk] = delta[indices].mean(axis=1)
        filled += chunk
    return samples


def summarize_delta(delta, n_bootstrap, seed):
    boot = bootstrap_delta(delta, n_bootstrap, seed)
    mean_delta = float(np.mean(delta))
    return {
        "baseline_mean": None,
        "candidate_mean": None,
        "mean_delta": mean_delta,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_delta_lt_0": float(np.mean(boot < 0.0)),
        "p_delta_gt_0": float(np.mean(boot > 0.0)),
        "n_windows": int(delta.shape[0]),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def summarize_metric(name, baseline_values, candidate_values, n_bootstrap, seed):
    baseline_values = np.asarray(baseline_values, dtype=np.float64).reshape(-1)
    candidate_values = np.asarray(candidate_values, dtype=np.float64).reshape(-1)
    if baseline_values.shape != candidate_values.shape:
        raise ValueError(f"Metric {name} shape mismatch: {baseline_values.shape} != {candidate_values.shape}")
    summary = summarize_delta(candidate_values - baseline_values, n_bootstrap, seed)
    summary["baseline_mean"] = float(np.mean(baseline_values))
    summary["candidate_mean"] = float(np.mean(candidate_values))
    return name, summary


def default_selected_horizons(horizon):
    return [step for step in [24, 48, 72, 96] if step <= horizon] or [horizon]


def compare_dirs(
    baseline_dir,
    candidate_dir,
    horizon,
    n_bootstrap=10000,
    seed=2020,
    target_tolerance=0.1,
    selected_horizons=None,
):
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    selected_horizons = selected_horizons or default_selected_horizons(horizon)
    base_true, base_pred = load_arrays(baseline_dir)
    cand_true, cand_pred = load_arrays(candidate_dir)
    if base_true.shape != cand_true.shape:
        raise ValueError(f"Target shape mismatch: {base_true.shape} != {cand_true.shape}")
    if not np.allclose(base_true, cand_true, atol=target_tolerance, rtol=0.0):
        max_diff = float(np.max(np.abs(base_true - cand_true)))
        raise ValueError(f"Target arrays do not align; max_abs_diff={max_diff}")

    base_err = mape_error(base_true, base_pred)
    cand_err = mape_error(cand_true, cand_pred)
    if base_err.shape[1] != horizon:
        raise ValueError(f"Horizon {horizon} does not match arrays {base_err.shape}")

    metrics = {}
    name, summary = summarize_metric(
        "overall_full_horizon_mape",
        base_err.mean(axis=(1, 2)),
        cand_err.mean(axis=(1, 2)),
        n_bootstrap,
        seed,
    )
    metrics[name] = summary

    for load_idx, load_name in enumerate(LOAD_NAMES):
        name, summary = summarize_metric(
            f"{load_name}_full_horizon_mape",
            base_err[:, :, load_idx].mean(axis=1),
            cand_err[:, :, load_idx].mean(axis=1),
            n_bootstrap,
            seed + load_idx + 1,
        )
        metrics[name] = summary

    by_horizon = {}
    cooling_idx = 1
    for step in selected_horizons:
        if step < 1 or step > horizon:
            raise ValueError(f"Selected horizon step {step} is outside 1..{horizon}")
        name, summary = summarize_metric(
            f"cooling_h{step}_mape",
            base_err[:, step - 1, cooling_idx],
            cand_err[:, step - 1, cooling_idx],
            n_bootstrap,
            seed + 100 + step,
        )
        by_horizon[name] = summary

    return {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "delta_semantics": "candidate - baseline; negative means candidate has lower MAPE",
        "horizon": int(horizon),
        "shape": list(base_true.shape),
        "target_tolerance": float(target_tolerance),
        "metrics": metrics,
        "selected_by_horizon_cooling": by_horizon,
    }


def compare(args):
    return compare_dirs(
        args.baseline_dir,
        args.candidate_dir,
        args.horizon,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        target_tolerance=args.target_tolerance,
        selected_horizons=args.selected_horizons,
    )


def metric_pass(mean_delta, ci95, require_improvement=False, require_ci_upper_nonpositive=False):
    passed = mean_delta < 0.0 if require_improvement else mean_delta <= 0.0
    if require_ci_upper_nonpositive:
        passed = passed and ci95[1] <= 0.0
    return bool(passed)


def build_single_horizon_gate(report):
    horizon = int(report["horizon"])
    metrics = report["metrics"]
    checks = {}
    if horizon == 24:
        item = metrics["overall_full_horizon_mape"]
        checks["overall_24h_nonworse"] = {
            "metric": "overall_full_horizon_mape",
            "pass": metric_pass(item["mean_delta"], item["ci95"]),
            "rule": "mean_delta <= 0",
            **item,
        }
    elif horizon == 96:
        item = metrics["overall_full_horizon_mape"]
        checks["overall_96h_nonworse"] = {
            "metric": "overall_full_horizon_mape",
            "pass": metric_pass(item["mean_delta"], item["ci95"]),
            "rule": "mean_delta <= 0",
            **item,
        }
        item = metrics["cooling_full_horizon_mape"]
        checks["cooling_96h_improved"] = {
            "metric": "cooling_full_horizon_mape",
            "pass": metric_pass(item["mean_delta"], item["ci95"], require_improvement=True, require_ci_upper_nonpositive=True),
            "rule": "mean_delta < 0 and ci95_upper <= 0",
            **item,
        }

    if not checks:
        return {
            "mode": "single_horizon",
            "pass": None,
            "reason": "D1 gate is only defined for 24h overall, 96h overall, and 96h cooling.",
            "checks": checks,
        }
    return {
        "mode": "single_horizon",
        "pass": bool(all(item["pass"] for item in checks.values())),
        "checks": checks,
    }


def compare_d1_gate(args):
    reports = {}
    for horizon in D1_GATE_HORIZONS:
        reports[str(horizon)] = compare_dirs(
            Path(args.baseline_root) / f"{horizon}-steps",
            Path(args.candidate_root) / f"{horizon}-steps",
            horizon,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            target_tolerance=args.target_tolerance,
            selected_horizons=default_selected_horizons(horizon),
        )

    overall_24 = reports["24"]["metrics"]["overall_full_horizon_mape"]
    overall_96 = reports["96"]["metrics"]["overall_full_horizon_mape"]
    cooling_96 = reports["96"]["metrics"]["cooling_full_horizon_mape"]
    checks = {
        "overall_24h_nonworse": {
            "metric": "overall_full_horizon_mape",
            "horizon": 24,
            "pass": metric_pass(overall_24["mean_delta"], overall_24["ci95"]),
            "rule": "mean_delta <= 0",
            **overall_24,
        },
        "overall_96h_nonworse": {
            "metric": "overall_full_horizon_mape",
            "horizon": 96,
            "pass": metric_pass(overall_96["mean_delta"], overall_96["ci95"]),
            "rule": "mean_delta <= 0",
            **overall_96,
        },
        "cooling_96h_improved": {
            "metric": "cooling_full_horizon_mape",
            "horizon": 96,
            "pass": metric_pass(
                cooling_96["mean_delta"],
                cooling_96["ci95"],
                require_improvement=True,
                require_ci_upper_nonpositive=True,
            ),
            "rule": "mean_delta < 0 and ci95_upper <= 0",
            **cooling_96,
        },
    }
    return {
        "mode": "d1_gate",
        "baseline_root": str(Path(args.baseline_root)),
        "candidate_root": str(Path(args.candidate_root)),
        "delta_semantics": "candidate - baseline; negative means candidate has lower MAPE",
        "pass_rule": "overall 24h mean_delta <= 0; overall 96h mean_delta <= 0; cooling 96h mean_delta < 0 and CI95 upper <= 0.",
        "pass": bool(all(item["pass"] for item in checks.values())),
        "checks": checks,
        "reports_by_horizon": reports,
    }


def print_check(name, item):
    lo, hi = item["ci95"]
    print(
        f"{name}: baseline={item['baseline_mean']:.6f}, "
        f"candidate={item['candidate_mean']:.6f}, delta={item['mean_delta']:.6f}, "
        f"ci95=[{lo:.6f}, {hi:.6f}], pass={item['pass']}"
    )


def print_report(report):
    if report.get("mode") == "d1_gate":
        print(f"D1 gate: {'PASS' if report['pass'] else 'FAIL'}")
        for name in ["overall_24h_nonworse", "overall_96h_nonworse", "cooling_96h_improved"]:
            print_check(name, report["checks"][name])
        return

    for group_name in ["overall_full_horizon_mape", "cooling_full_horizon_mape"]:
        item = report["metrics"][group_name]
        lo, hi = item["ci95"]
        print(
            f"{group_name}: baseline={item['baseline_mean']:.6f}, "
            f"candidate={item['candidate_mean']:.6f}, delta={item['mean_delta']:.6f}, "
            f"ci95=[{lo:.6f}, {hi:.6f}], p_lt_0={item['p_delta_lt_0']:.4f}"
        )
    gate = build_single_horizon_gate(report)
    if gate["pass"] is not None:
        print(f"D1 single-horizon gate: {'PASS' if gate['pass'] else 'FAIL'}")
        for name, item in gate["checks"].items():
            print_check(name, item)


def parse_args():
    parser = argparse.ArgumentParser(description="Paired bootstrap over test windows for saved forecast arrays.")
    parser.add_argument("--baseline-dir")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--baseline-root", help="root containing 24-steps and 96-steps for D1 gate mode")
    parser.add_argument("--candidate-root", help="root containing 24-steps and 96-steps for D1 gate mode")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--target-tolerance", type=float, default=0.1)
    parser.add_argument("--selected-horizons", type=int, nargs="+", default=None)
    args = parser.parse_args()
    single_mode = args.baseline_dir or args.candidate_dir or args.horizon
    gate_mode = args.baseline_root or args.candidate_root
    if single_mode and gate_mode:
        parser.error("Use either --baseline-dir/--candidate-dir/--horizon or --baseline-root/--candidate-root, not both.")
    if gate_mode:
        if not args.baseline_root or not args.candidate_root:
            parser.error("D1 gate mode requires both --baseline-root and --candidate-root.")
    else:
        if not args.baseline_dir or not args.candidate_dir or args.horizon is None:
            parser.error("Single-horizon mode requires --baseline-dir, --candidate-dir, and --horizon.")
    return args


def main():
    args = parse_args()
    if args.baseline_root or args.candidate_root:
        report = compare_d1_gate(args)
    else:
        report = compare(args)
        report["d1_gate"] = build_single_horizon_gate(report)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {output_json}")
    print_report(report)


if __name__ == "__main__":
    main()
