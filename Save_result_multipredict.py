import argparse
import json
import os

import numpy as np

from metrics import LOAD_NAMES, metric_by_horizon, metric_by_load, metric_by_load_and_horizon, metric_overall


def load_arrays(result_dir):
    y_true_path = os.path.join(result_dir, "all_y_true.npy")
    y_pred_path = os.path.join(result_dir, "all_predict_value.npy")
    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    return y_true, y_pred


def compute_report(y_true, y_pred, horizon=None, load_names=LOAD_NAMES, mask=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {y_true.shape} != {y_pred.shape}")
    if y_true.ndim != 3:
        raise ValueError(f"expected [N, H, C] arrays, got shape {y_true.shape}")
    if horizon is not None and y_true.shape[1] != horizon:
        raise ValueError(f"horizon {horizon} does not match array shape {y_true.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != y_true.shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} != {y_true.shape}")

    exclude_zero = os.environ.get("EGC_MAPE_ZERO_POLICY", "legacy").strip().lower() == "exclude"
    return {
        "shape": list(y_true.shape),
        "horizon": int(y_true.shape[1]),
        "metric_semantics": {
            "mape": (
                "mean(abs((y_true - y_pred) / y_true)) * 100 over valid nonzero truths; "
                "zero truths are counted separately"
                if exclude_zero else
                "legacy raw mean(abs((y_true - y_pred) / y_true)) * 100; no epsilon floor"
            ),
            "smape": "mean(2*abs(y_true-y_pred)/(abs(y_true)+abs(y_pred)+eps))*100 with eps=1e-6",
            "wape": "sum(abs(y_true-y_pred))/sum(abs(y_true))*100",
            "nrmse_mean": "rmse/mean(abs(y_true))*100",
            "corr": "mean Pearson-style correlation over non-constant axes",
        },
        "label_valid_count": int(mask.sum()) if mask is not None else int(y_true.size),
        "label_total_count": int(y_true.size),
        "overall": metric_overall(y_true, y_pred, mask=mask),
        "by_load": metric_by_load(y_true, y_pred, load_names, mask=mask),
        "by_horizon": metric_by_horizon(y_true, y_pred, mask=mask),
        "by_load_and_horizon": metric_by_load_and_horizon(y_true, y_pred, load_names, mask=mask),
    }


def _format_metric_row(metrics):
    return (
        "mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}, smape: {smape:.4f}, "
        "wape: {wape:.4f}, nrmse_mean: {nrmse_mean:.4f}, corr: {corr:.4f}"
    ).format(**metrics)


def print_report(report):
    print(f"shape: {report['shape']} | horizon: {report['horizon']}")
    print("metric semantics:")
    for name, desc in report["metric_semantics"].items():
        print(f"  {name}: {desc}")

    print("\n============ overall full-horizon ===========")
    print(_format_metric_row(report["overall"]))

    print("\n============ by load full-horizon ===========")
    for row in report["by_load"]:
        print(f"{row['load']}: {_format_metric_row(row)}")

    print("\n============ by horizon ===========")
    for row in report["by_horizon"]:
        print(f"h={row['horizon_step']}: {_format_metric_row(row)}")

    print("\n============ load x horizon mape ===========")
    for row in report["by_load_and_horizon"]:
        print(f"h={row['horizon_step']} {row['load']}: mape={row['mape']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute full-horizon metrics from saved prediction arrays.")
    parser.add_argument("--result-dir", default="./result", help="Directory containing all_y_true.npy and all_predict_value.npy")
    parser.add_argument("--horizon", type=int, default=None, help="Expected forecast horizon length")
    parser.add_argument("--output-json", default=None, help="Optional path to write a structured JSON report")
    return parser.parse_args()


def main():
    args = parse_args()
    y_true, y_pred = load_arrays(args.result_dir)
    report = compute_report(y_true, y_pred, args.horizon)
    print_report(report)

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
