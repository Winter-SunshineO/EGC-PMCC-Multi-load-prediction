import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


LOAD_NAMES = ("electricity", "cooling", "heating")


def load_arrays(result_dir):
    result_dir = Path(result_dir)
    y_true = np.load(result_dir / "all_y_true.npy").astype(np.float64)
    y_pred = np.load(result_dir / "all_predict_value.npy").astype(np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch in {result_dir}: {y_true.shape} != {y_pred.shape}")
    if y_true.ndim != 3:
        raise ValueError(f"expected [N,H,C] arrays in {result_dir}, got {y_true.shape}")
    return y_true, y_pred


def assert_aligned(reference_true, baseline_true, tolerance):
    if reference_true.shape != baseline_true.shape:
        raise ValueError(f"target shape mismatch: {reference_true.shape} != {baseline_true.shape}")
    max_diff = float(np.max(np.abs(reference_true - baseline_true)))
    if max_diff > tolerance:
        raise ValueError(f"target arrays are not aligned: max_abs_diff={max_diff:.6g}, tolerance={tolerance}")
    return max_diff


def absolute_loss(y_true, y_pred, scope):
    err = np.abs(y_true - y_pred)
    if scope == "overall":
        return err.mean(axis=(1, 2))
    if scope in LOAD_NAMES:
        load_idx = LOAD_NAMES.index(scope)
        return err[:, :, load_idx].mean(axis=1)
    raise ValueError(f"unknown scope {scope}")


def newey_west_variance_of_mean(diff, lag):
    diff = np.asarray(diff, dtype=np.float64).reshape(-1)
    n = diff.shape[0]
    if n < 2:
        return float("nan")
    lag = min(max(int(lag), 0), n - 1)
    centered = diff - diff.mean()
    long_run_var = np.mean(centered * centered)
    for k in range(1, lag + 1):
        gamma = np.mean(centered[k:] * centered[:-k])
        weight = 1.0 - k / (lag + 1.0)
        long_run_var += 2.0 * weight * gamma
    long_run_var = max(float(long_run_var), 0.0)
    return long_run_var / n


def dm_test(reference_loss, baseline_loss, lag):
    diff = np.asarray(reference_loss - baseline_loss, dtype=np.float64)
    mean_diff = float(diff.mean())
    var_mean = newey_west_variance_of_mean(diff, lag)
    if not np.isfinite(var_mean) or var_mean <= 0.0:
        statistic = float("-inf") if mean_diff < 0 else float("inf") if mean_diff > 0 else 0.0
        p_less = 0.0 if mean_diff < 0 else 1.0 if mean_diff > 0 else 0.5
    else:
        statistic = float(mean_diff / np.sqrt(var_mean))
        p_less = float(norm.cdf(statistic))
    return {
        "mean_diff": mean_diff,
        "dm_statistic": statistic,
        "p_one_sided_reference_lower_loss": p_less,
        "reference_mean_loss": float(np.mean(reference_loss)),
        "baseline_mean_loss": float(np.mean(baseline_loss)),
        "n_windows": int(diff.shape[0]),
        "hac_lag": int(lag),
    }


def compare(reference_dir, baseline_dir, horizon, scopes, lag, tolerance):
    ref_true, ref_pred = load_arrays(reference_dir)
    base_true, base_pred = load_arrays(baseline_dir)
    max_diff = assert_aligned(ref_true, base_true, tolerance)
    if ref_true.shape[1] != horizon:
        raise ValueError(f"horizon {horizon} does not match reference arrays {ref_true.shape}")
    report = {
        "reference_dir": str(reference_dir),
        "baseline_dir": str(baseline_dir),
        "horizon": int(horizon),
        "shape": list(ref_true.shape),
        "target_alignment_max_abs_diff": max_diff,
        "loss": "mean absolute error over the selected scope for each matched test window",
        "diff_semantics": "reference_loss - baseline_loss; negative means reference has lower loss",
        "alternative": "E[reference_loss - baseline_loss] < 0",
        "scopes": {},
    }
    for scope in scopes:
        report["scopes"][scope] = dm_test(
            absolute_loss(ref_true, ref_pred, scope),
            absolute_loss(base_true, base_pred, scope),
            lag,
        )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Diebold-Mariano tests for saved forecast arrays.")
    parser.add_argument("--reference-dir", required=True, help="Usually result/p2b-cfc-pmd-cecm-only/<H>-steps")
    parser.add_argument("--baseline-dirs", nargs="+", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scopes", nargs="+", default=["overall", "cooling"], choices=["overall", *LOAD_NAMES])
    parser.add_argument("--lag", type=int, default=None, help="Newey-West lag; defaults to horizon - 1")
    parser.add_argument("--target-tolerance", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    lag = args.horizon - 1 if args.lag is None else args.lag
    reports = {}
    for baseline_dir in args.baseline_dirs:
        key = Path(baseline_dir).parent.name
        reports[key] = compare(
            Path(args.reference_dir),
            Path(baseline_dir),
            args.horizon,
            args.scopes,
            lag,
            args.target_tolerance,
        )
    output = {
        "reference_dir": args.reference_dir,
        "horizon": int(args.horizon),
        "primary_hac_lag": int(lag),
        "comparisons": reports,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {output_path}")
    for name, report in reports.items():
        pieces = []
        for scope, item in report["scopes"].items():
            pieces.append(
                f"{scope}: diff={item['mean_diff']:.6f}, "
                f"p={item['p_one_sided_reference_lower_loss']:.4g}"
            )
        print(f"{name} | " + " | ".join(pieces))


if __name__ == "__main__":
    main()
