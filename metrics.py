import math
import os

import numpy as np


LOAD_NAMES = ("electricity", "cooling", "heating")
EPSILON = 1e-6


def _exclude_zero_mape():
    return os.environ.get("EGC_MAPE_ZERO_POLICY", "legacy").strip().lower() == "exclude"


def _as_array(values):
    return np.asarray(values)


def _as_float(value):
    return float(value)


def _filter_valid(y_true, y_pre, mask=None):
    """Return finite target/prediction pairs selected by an optional label mask."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pre = np.asarray(y_pre, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pre)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return y_true[valid], y_pre[valid]


def MAE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = np.abs(y_true - y_pre).mean()
    return re


def RMSE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = math.sqrt(((y_true - y_pre) ** 2).mean())
    return re


def MAPE(y_true, y_pre):
    y_true = np.asarray(y_true, dtype=np.float64).reshape((-1, 1))
    y_pre = np.asarray(y_pre, dtype=np.float64).reshape((-1, 1))

    if _exclude_zero_mape():
        nonzero = y_true != 0.0
        if not np.any(nonzero):
            return float("nan")
        y_true = y_true[nonzero]
        y_pre = y_pre[nonzero]

    # e = (y_true + y_pre) / 2 + 1e-2
    # re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()
    # P0 keeps this legacy definition so archived Cfc baselines remain comparable.
    re = np.mean(np.abs((y_true - y_pre) / y_true)) * 100

    return re


def SMAPE(y_true, y_pre, eps=EPSILON):
    y_true = np.asarray(y_true, dtype=np.float64).reshape((-1, 1))
    y_pre = np.asarray(y_pre, dtype=np.float64).reshape((-1, 1))
    re = np.mean(2.0 * np.abs(y_true - y_pre) / (np.abs(y_true) + np.abs(y_pre) + eps)) * 100.0
    return re


def WAPE(y_true, y_pre, eps=EPSILON):
    y_true = np.asarray(y_true, dtype=np.float64).reshape((-1, 1))
    y_pre = np.asarray(y_pre, dtype=np.float64).reshape((-1, 1))
    denom = np.sum(np.abs(y_true))
    if denom <= eps:
        return float("nan")
    return np.sum(np.abs(y_true - y_pre)) / denom * 100.0


def NRMSE_MEAN(y_true, y_pre, eps=EPSILON):
    y_true = np.asarray(y_true, dtype=np.float64).reshape((-1, 1))
    denom = np.mean(np.abs(y_true))
    if denom <= eps:
        return float("nan")
    return RMSE(y_true, y_pre) / denom * 100.0


def correlation(y_true, y_pre):
    y_true = _as_array(y_true)
    y_pre = _as_array(y_pre)
    sigma_p = y_pre.std(axis=0)
    sigma_g = y_true.std(axis=0)
    mean_p = y_pre.mean(axis=0)
    mean_g = y_true.mean(axis=0)
    denom = sigma_p * sigma_g
    index = denom != 0
    corr = np.zeros_like(denom, dtype=np.float64)
    corr[index] = ((y_pre - mean_p) * (y_true - mean_g)).mean(axis=0)[index] / denom[index]
    if not np.any(index):
        return float("nan")
    return _as_float(corr[index].mean())


def metric_overall(y_true, y_pre, mask=None):
    y_true, y_pre = _filter_valid(y_true, y_pre, mask)
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "smape": float("nan"), "wape": float("nan"), "nrmse_mean": float("nan"), "corr": float("nan"), "valid_count": 0, "mape_valid_count": 0, "valid_zero_count": 0, "mape_valid_fraction": float("nan")}
    nonzero_count = int(np.count_nonzero(y_true != 0.0))
    valid_zero_count = int(y_true.size - nonzero_count)
    mape_valid_count = nonzero_count if _exclude_zero_mape() else int(y_true.size)
    return {
        "mae": _as_float(MAE(y_true, y_pre)),
        "rmse": _as_float(RMSE(y_true, y_pre)),
        "mape": _as_float(MAPE(y_true, y_pre)),
        "smape": _as_float(SMAPE(y_true, y_pre)),
        "wape": _as_float(WAPE(y_true, y_pre)),
        "nrmse_mean": _as_float(NRMSE_MEAN(y_true, y_pre)),
        "corr": _as_float(correlation(y_true, y_pre)),
        "valid_count": int(y_true.size),
        "mape_valid_count": mape_valid_count,
        "valid_zero_count": valid_zero_count,
        "mape_valid_fraction": _as_float(mape_valid_count / y_true.size),
    }


def metric_by_load(y_true, y_pre, load_names=LOAD_NAMES, mask=None):
    y_true = _as_array(y_true)
    y_pre = _as_array(y_pre)
    records = []
    for load_idx in range(y_true.shape[-1]):
        name = load_names[load_idx] if load_idx < len(load_names) else f"load_{load_idx}"
        local_mask = None if mask is None else mask[:, :, load_idx:load_idx + 1]
        metrics = metric_overall(y_true[:, :, load_idx:load_idx + 1], y_pre[:, :, load_idx:load_idx + 1], local_mask)
        records.append({"load": name, "load_idx": load_idx, **metrics})
    return records


def metric_by_horizon(y_true, y_pre, mask=None):
    y_true = _as_array(y_true)
    y_pre = _as_array(y_pre)
    records = []
    for horizon_idx in range(y_true.shape[1]):
        local_mask = None if mask is None else mask[:, horizon_idx:horizon_idx + 1, :]
        metrics = metric_overall(y_true[:, horizon_idx:horizon_idx + 1, :], y_pre[:, horizon_idx:horizon_idx + 1, :], local_mask)
        records.append({"horizon_step": horizon_idx + 1, **metrics})
    return records


def metric_by_load_and_horizon(y_true, y_pre, load_names=LOAD_NAMES, mask=None):
    y_true = _as_array(y_true)
    y_pre = _as_array(y_pre)
    records = []
    for horizon_idx in range(y_true.shape[1]):
        for load_idx in range(y_true.shape[-1]):
            name = load_names[load_idx] if load_idx < len(load_names) else f"load_{load_idx}"
            local_mask = None if mask is None else mask[:, horizon_idx:horizon_idx + 1, load_idx:load_idx + 1]
            metrics = metric_overall(
                y_true[:, horizon_idx:horizon_idx + 1, load_idx:load_idx + 1],
                y_pre[:, horizon_idx:horizon_idx + 1, load_idx:load_idx + 1],
                local_mask,
            )
            records.append({
                "horizon_step": horizon_idx + 1,
                "load": name,
                "load_idx": load_idx,
                **metrics,
            })
    return records
