import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quantile_metrics import load_label_mask, load_quantiles, make_report


def make_candidate_scales(args):
    if args.candidate_scales:
        scales = [float(scale) for scale in args.candidate_scales]
    else:
        count = int(round((args.scale_max - args.scale_min) / args.scale_step)) + 1
        scales = [args.scale_min + idx * args.scale_step for idx in range(count)]
    scales = sorted({round(float(scale), 10) for scale in scales if scale > 0.0})
    if not scales:
        raise ValueError("no positive candidate scales were provided")
    return scales


def spread_scale_quantiles(y_quantile, quantiles, scale):
    median_idx = int(np.where(np.isclose(quantiles, 0.5))[0][0])
    median = y_quantile[..., median_idx : median_idx + 1]
    scaled = median + float(scale) * (y_quantile - median)
    scaled[..., median_idx] = y_quantile[..., median_idx]
    return scaled


def compact_metrics(report, scale):
    interval = report["interval_80"]
    return {
        "scale": float(scale),
        "median_point_mape": float(report["median_point_mape"]),
        "picp80": float(interval["picp"]),
        "ace80": float(interval["ace"]),
        "pinaw80": float(interval["pinaw"]),
        "crps_quantile": float(report["crps_quantile"]),
        "winkler80": float(interval["winkler"]),
        "normalized_winkler80": float(interval["normalized_winkler"]),
    }


def choose_scale(scan, coverage_min, coverage_max, ace_max):
    passing = [
        row
        for row in scan
        if coverage_min <= row["picp80"] <= coverage_max
        and abs(row["ace80"]) <= ace_max
    ]
    if passing:
        selected = min(passing, key=lambda row: (row["winkler80"], abs(row["ace80"]), row["scale"]))
        return selected, "passed_picp_gate_min_winkler"

    return None, "rejected_no_picp_gate_candidate"


def canonical_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_calibration_metadata(report, calibration):
    report = dict(report)
    metadata = dict(report.get("metadata", {}))
    metadata["posthoc_spread_calibration"] = calibration
    report["metadata"] = metadata
    report["calibration"] = calibration
    return report


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc spread scale calibration for saved quantile forecasts.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--coverage-min", type=float, default=0.78)
    parser.add_argument("--coverage-max", type=float, default=0.82)
    parser.add_argument("--ace-max", type=float, default=0.02)
    parser.add_argument("--scale", type=float, default=None, help="fixed global spread scale to apply")
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--scale-step", type=float, default=0.01)
    parser.add_argument("--candidate-scales", type=float, nargs="+", default=None)
    parser.add_argument("--output-raw-json", default=None)
    parser.add_argument("--output-scan-json", default=None)
    parser.add_argument("--output-calibrated-json", default=None)
    parser.add_argument("--output-quantile-npy", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    y_true, y_quantile, quantiles, metadata = load_quantiles(result_dir)
    label_mask = load_label_mask(result_dir)
    if y_true.shape[1] != args.horizon:
        raise ValueError(f"horizon {args.horizon} does not match arrays {y_true.shape}")

    raw_json = Path(args.output_raw_json) if args.output_raw_json else result_dir / "quantile_metrics_raw.json"
    scan_json = Path(args.output_scan_json) if args.output_scan_json else result_dir / "quantile_calibration_scan.json"
    calibrated_json = (
        Path(args.output_calibrated_json)
        if args.output_calibrated_json
        else result_dir / "quantile_metrics_calibrated.json"
    )
    calibrated_npy = (
        Path(args.output_quantile_npy)
        if args.output_quantile_npy
        else result_dir / "all_quantile_value_calibrated.npy"
    )

    raw_report = make_report(y_true, y_quantile, quantiles, metadata, mask=label_mask)
    raw_summary = compact_metrics(raw_report, scale=1.0)

    scan = []
    for scale in make_candidate_scales(args):
        report = make_report(y_true, spread_scale_quantiles(y_quantile, quantiles, scale), quantiles, metadata, mask=label_mask)
        scan.append(compact_metrics(report, scale=scale))

    if args.scale is not None:
        matching = [row for row in scan if abs(row["scale"] - args.scale) <= 1e-10]
        if matching:
            selected = matching[0]
        else:
            report = make_report(y_true, spread_scale_quantiles(y_quantile, quantiles, args.scale), quantiles, metadata, mask=label_mask)
            selected = compact_metrics(report, scale=args.scale)
            scan.append(selected)
            scan = sorted(scan, key=lambda row: row["scale"])
        if not (args.coverage_min <= selected["picp80"] <= args.coverage_max and abs(selected["ace80"]) <= args.ace_max):
            selected = None
            selection_reason = "rejected_fixed_scale_fails_picp_gate"
        else:
            selection_reason = "fixed_scale_passed_picp_gate"
        best_scan_selected, best_scan_reason = choose_scale(scan, args.coverage_min, args.coverage_max, args.ace_max)
    else:
        selected, selection_reason = choose_scale(scan, args.coverage_min, args.coverage_max, args.ace_max)
        best_scan_selected, best_scan_reason = selected, selection_reason
    scan_hash = canonical_hash(scan)
    if selected is None:
        calibration = {
            "method": "posthoc_global_spread_scale",
            "status": "calibration_rejected",
            "main_table_eligible": False,
            "selection_reason": selection_reason,
            "coverage_min": float(args.coverage_min),
            "coverage_max": float(args.coverage_max),
            "ace_max": float(args.ace_max),
            "raw": raw_summary,
            "best_scan_selected": best_scan_selected,
            "best_scan_reason": best_scan_reason,
            "scan_sha256": scan_hash,
            "candidate_count": len(scan),
        }
        write_json(raw_json, add_calibration_metadata(raw_report, {**calibration, "applied": False}))
        write_json(scan_json, {"calibration": calibration, "scan": scan})
        print(f"Wrote {raw_json}")
        print(f"Wrote {scan_json}")
        raise SystemExit("calibration rejected: no candidate satisfies the frozen PICP/ACE gate")
    calibrated_quantile = spread_scale_quantiles(y_quantile, quantiles, selected["scale"])
    calibrated_report = make_report(y_true, calibrated_quantile, quantiles, metadata, mask=label_mask)

    calibration = {
        "method": "posthoc_global_spread_scale",
        "status": "selected",
        "main_table_eligible": True,
        "selection_rule": "PICP gate [coverage_min, coverage_max], then minimum validation Winkler; ties abs ACE then scale",
        "selected_scale": float(selected["scale"]),
        "selection_reason": selection_reason,
        "coverage_min": float(args.coverage_min),
        "coverage_max": float(args.coverage_max),
        "ace_max": float(args.ace_max),
        "raw": raw_summary,
        "selected": selected,
        "best_scan_selected": best_scan_selected,
        "best_scan_reason": best_scan_reason,
        "scan_sha256": scan_hash,
        "candidate_count": len(scan),
        "q50_unchanged_max_abs_diff": float(
            np.max(np.abs(y_quantile[..., np.where(np.isclose(quantiles, 0.5))[0][0]] - calibrated_quantile[..., np.where(np.isclose(quantiles, 0.5))[0][0]]))
        ),
    }

    write_json(raw_json, add_calibration_metadata(raw_report, {**calibration, "applied": False}))
    write_json(scan_json, {"calibration": calibration, "scan": scan})
    write_json(calibrated_json, add_calibration_metadata(calibrated_report, {**calibration, "applied": True}))
    calibrated_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(calibrated_npy, calibrated_quantile.astype(np.float32))

    selected_interval = calibrated_report["interval_80"]
    print(f"Wrote {raw_json}")
    print(f"Wrote {scan_json}")
    print(f"Wrote {calibrated_json}")
    print(f"Wrote {calibrated_npy}")
    print(
        "selected: "
        f"scale={selected['scale']:.4f}, "
        f"picp80={selected_interval['picp']:.4f}, "
        f"ace80={selected_interval['ace']:.4f}, "
        f"pinaw80={selected_interval['pinaw']:.4f}, "
        f"crps_q={calibrated_report['crps_quantile']:.4f}, "
        f"winkler80={selected_interval['winkler']:.4f}, "
        f"q50_diff={calibration['q50_unchanged_max_abs_diff']:.6g}"
    )


if __name__ == "__main__":
    main()
