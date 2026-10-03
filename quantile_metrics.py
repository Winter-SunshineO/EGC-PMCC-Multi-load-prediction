import argparse
import json
from pathlib import Path

import numpy as np


LOAD_NAMES = ("electricity", "cooling", "heating")


def load_quantiles(result_dir):
    result_dir = Path(result_dir)
    prefix = "all" if (result_dir / "all_y_true.npy").exists() else "val"
    y_true = np.load(result_dir / f"{prefix}_y_true.npy").astype(np.float64)
    quantile_path = result_dir / f"{prefix}_quantile_value.npy"
    if not quantile_path.exists() and prefix == "all":
        quantile_path = result_dir / "all_quantile_value.npy"
    y_quantile = np.load(quantile_path).astype(np.float64)
    metadata_path = result_dir / ("quantile_metadata.json" if prefix == "all" else "val_quantile_metadata.json")
    if metadata_path.exists():
        metadata = json.load(open(metadata_path, encoding="utf-8"))
        quantiles = np.asarray(metadata["quantiles"], dtype=np.float64)
    else:
        quantiles = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
        metadata = {"quantiles": quantiles.tolist(), "source": "default"}
    if y_quantile.ndim != 4:
        raise ValueError(f"expected quantile array [N,H,C,Q], got {y_quantile.shape}")
    if y_true.shape != y_quantile.shape[:3]:
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_quantile {y_quantile.shape}")
    if y_quantile.shape[-1] != len(quantiles):
        raise ValueError(f"quantile metadata length {len(quantiles)} does not match array {y_quantile.shape}")
    return y_true, y_quantile, quantiles, metadata


def load_label_mask(result_dir):
    result_dir = Path(result_dir)
    for name in ("all_label_valid_mask.npy", "val_label_valid_mask.npy"):
        path = result_dir / name
        if path.exists():
            return np.load(path).astype(bool)
    return None


def pinball(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def interval_metrics(y_true, y_quantile, quantiles, lower_q=0.1, upper_q=0.9, mask=None):
    lower_idx = int(np.where(np.isclose(quantiles, lower_q))[0][0])
    upper_idx = int(np.where(np.isclose(quantiles, upper_q))[0][0])
    lower = y_quantile[..., lower_idx]
    upper = y_quantile[..., upper_idx]
    width = upper - lower
    valid = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    safe_true = np.where(valid, y_true, 0.0)
    safe_lower = np.where(valid, lower, 0.0)
    safe_upper = np.where(valid, upper, 0.0)
    safe_width = np.where(valid, width, 0.0)
    covered = valid & (y_true >= lower) & (y_true <= upper)
    target_coverage = upper_q - lower_q

    below = valid & (y_true < lower)
    above = valid & (y_true > upper)
    alpha = 1.0 - target_coverage
    winkler = safe_width + (2.0 / alpha) * (safe_lower - safe_true) * below + (2.0 / alpha) * (safe_true - safe_upper) * above

    denom = np.mean(np.abs(safe_true[valid])) if np.any(valid) else 0.0
    count = int(valid.sum())
    return {
        "lower_q": float(lower_q),
        "upper_q": float(upper_q),
        "target_coverage": float(target_coverage),
        "picp": float(np.sum(covered) / count) if count else None,
        "ace_definition": "abs(PICP - nominal coverage)",
        "ace": float(abs(np.sum(covered) / count - target_coverage)) if count else None,
        "pinaw": float(np.sum(safe_width) / count / denom) if count and denom > 1e-12 else None,
        "mean_width": float(np.sum(safe_width) / count) if count else None,
        "winkler": float(np.sum(winkler) / count) if count else None,
        "normalized_winkler": float(np.sum(winkler) / count / denom) if count and denom > 1e-12 else None,
        "valid_count": count,
    }


def crps_quantile(y_true, y_quantile, quantiles, mask=None):
    losses = []
    for idx, quantile in enumerate(quantiles):
        loss = pinball(y_true, y_quantile[..., idx], quantile)
        if mask is not None:
            loss = np.where(np.asarray(mask, dtype=bool), loss, np.nan)
        losses.append(loss)
    stacked = np.stack(losses, axis=-1)
    return float(2.0 * np.nanmean(stacked)) if np.isfinite(stacked).any() else None


def mape(y_true, y_pred, mask=None):
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0) if np.any(valid) else None


def monotonicity_metrics(y_quantile, tolerance=1e-8):
    diffs = np.diff(y_quantile, axis=-1)
    valid = diffs >= -float(tolerance)
    return {
        "fraction": float(np.mean(valid)),
        "violation_count": int(np.size(valid) - np.count_nonzero(valid)),
        "min_diff": float(np.min(diffs)),
        "tolerance": float(tolerance),
    }


def make_report(y_true, y_quantile, quantiles, metadata, mask=None):
    median_idx = int(np.where(np.isclose(quantiles, 0.5))[0][0])
    report = {
        "shape": list(y_quantile.shape),
        "quantiles": [float(q) for q in quantiles],
        "metadata": metadata,
        "median_point_mape": mape(y_true, y_quantile[..., median_idx], mask),
        "crps_quantile": crps_quantile(y_true, y_quantile, quantiles, mask),
        "interval_80": interval_metrics(y_true, y_quantile, quantiles, 0.1, 0.9, mask),
        "quantile_monotonicity": monotonicity_metrics(y_quantile),
        "by_load": [],
        "by_horizon": [],
    }

    for load_idx, load_name in enumerate(LOAD_NAMES):
        yt = y_true[:, :, load_idx]
        yq = y_quantile[:, :, load_idx, :]
        local_mask = None if mask is None else np.asarray(mask)[:, :, load_idx]
        report["by_load"].append(
            {
                "load": load_name,
                "load_idx": load_idx,
                "median_point_mape": mape(yt, yq[..., median_idx], local_mask),
                "crps_quantile": crps_quantile(yt, yq, quantiles, local_mask),
                "interval_80": interval_metrics(yt, yq, quantiles, 0.1, 0.9, local_mask),
            }
        )

    for step in range(y_true.shape[1]):
        yt = y_true[:, step, :]
        yq = y_quantile[:, step, :, :]
        local_mask = None if mask is None else np.asarray(mask)[:, step, :]
        report["by_horizon"].append(
            {
                "horizon_step": step + 1,
                "median_point_mape": mape(yt, yq[..., median_idx], local_mask),
                "crps_quantile": crps_quantile(yt, yq, quantiles, local_mask),
                "interval_80": interval_metrics(yt, yq, quantiles, 0.1, 0.9, local_mask),
            }
        )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Quantile forecast metrics from saved arrays.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    y_true, y_quantile, quantiles, metadata = load_quantiles(args.result_dir)
    label_mask = load_label_mask(args.result_dir)
    if y_true.shape[1] != args.horizon:
        raise ValueError(f"horizon {args.horizon} does not match arrays {y_true.shape}")
    report = make_report(y_true, y_quantile, quantiles, metadata, mask=label_mask)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    interval = report["interval_80"]
    print(f"Wrote {output_json}")
    print(
        "summary: "
        f"median_mape={report['median_point_mape']:.4f}, "
        f"picp80={interval['picp']:.4f}, "
        f"pinaw80={interval['pinaw']:.4f}, "
        f"ace80={interval['ace']:.4f}, "
        f"crps_q={report['crps_quantile']:.4f}, "
        f"winkler80={interval['winkler']:.4f}"
    )


if __name__ == "__main__":
    main()
