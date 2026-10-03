"""Auditable point-training protocols shared by formal and attribution runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import torch


TARGET_NAMES = ("KW", "CHWTON", "HTmmBTU")
A3_RECIPE = "a3"
LEGACY_RECIPE = "legacy"


def sha256_json(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def fit_target_robust_scaler(
    raw_labels: np.ndarray,
    hard_valid_mask: np.ndarray,
    train_end: int,
    target_names: Sequence[str] = TARGET_NAMES,
) -> dict[str, Any]:
    """Fit target-wise median/IQR scaling on hard-valid training labels only."""
    labels = np.asarray(raw_labels, dtype=np.float64)
    valid_mask = np.asarray(hard_valid_mask, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != len(target_names):
        raise ValueError("raw_labels must be a [rows, targets] array")
    if valid_mask.shape != labels.shape:
        raise ValueError("hard_valid_mask must match raw_labels")
    if not 0 < int(train_end) <= labels.shape[0]:
        raise ValueError("train_end must select a non-empty training prefix")

    targets: list[dict[str, Any]] = []
    centers: list[float] = []
    scales: list[float] = []
    for index, name in enumerate(target_names):
        raw = labels[:train_end, index]
        valid = valid_mask[:train_end, index] & np.isfinite(raw)
        values = raw[valid]
        if values.size == 0:
            raise ValueError(f"no hard-valid finite training labels for {name}")
        center = float(np.median(values))
        q25, q75 = np.percentile(values, [25.0, 75.0])
        iqr = float(q75 - q25)
        scale = float(max(iqr / 1.349, 1e-6))
        centers.append(center)
        scales.append(scale)
        targets.append({
            "target": str(name),
            "fit_rows": int(train_end),
            "valid_rows": int(values.size),
            "median": center,
            "iqr": iqr,
            "scale": scale,
            "zero_count": int(np.sum(valid & (raw == 0))),
            "finite_extreme_values_retained": True,
        })

    payload: dict[str, Any] = {
        "method": "training_only_median_iqr_over_1.349",
        "fit_split": "train_only",
        "fit_rows": int(train_end),
        "target_names": list(target_names),
        "center": centers,
        "scale": scales,
        "retain_zero_labels": True,
        "retain_extreme_finite_labels": True,
        "clip_or_winsorize_targets": False,
        "targets": targets,
    }
    payload["scaler_sha256"] = sha256_json(payload)
    return payload


def fit_target_bounded_tau(
    raw_labels: np.ndarray,
    hard_valid_mask: np.ndarray,
    train_end: int,
    target_names: Sequence[str] = TARGET_NAMES,
) -> dict[str, Any]:
    """Fit tau_k from positive, finite, hard-valid training labels only."""
    labels = np.asarray(raw_labels, dtype=np.float64)
    valid_mask = np.asarray(hard_valid_mask, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != len(target_names):
        raise ValueError("raw_labels must be a [rows, targets] array")
    if valid_mask.shape != labels.shape:
        raise ValueError("hard_valid_mask must match raw_labels")

    values: list[float] = []
    targets: list[dict[str, Any]] = []
    for index, name in enumerate(target_names):
        raw = labels[:train_end, index]
        hard_valid = valid_mask[:train_end, index] & np.isfinite(raw)
        positive = raw[hard_valid & (raw > 0)]
        if positive.size == 0:
            raise ValueError(f"no positive hard-valid training labels for {name}")
        tau = float(max(1e-2, 0.01 * np.median(np.abs(positive))))
        values.append(tau)
        targets.append({
            "target": str(name),
            "fit_split": "train_only",
            "valid_count": int(hard_valid.sum()),
            "positive_count": int(positive.size),
            "zero_count": int(np.sum(hard_valid & (raw == 0))),
            "tau_kw_equivalent": tau,
        })
    payload: dict[str, Any] = {
        "formula": "max(1e-2, 0.01 * median(abs(positive hard-valid training labels)))",
        "fit_split": "train_only",
        "target_names": list(target_names),
        "tau": values,
        "targets": targets,
    }
    payload["target_tau_sha256"] = sha256_json(payload)
    return payload


def apply_target_robust_scaler(data: Any, scaler: dict[str, Any]) -> None:
    """Apply the fitted target scaler to historical inputs and label outputs."""
    centers = np.asarray(scaler["center"], dtype=np.float64)
    scales = np.asarray(scaler["scale"], dtype=np.float64)
    if centers.shape != (3,) or scales.shape != (3,):
        raise ValueError("A3 target scaler must contain exactly three centers/scales")
    data.scale_mean[:3] = centers
    data.scale_std[:3] = scales
    data.dat[:, :3] = (data.rawdat[:, :3] - centers) / scales
    data.label_dat[:, :3] = (data.raw_labels[:, :3] - centers) / scales
    train_end = int(data.train_size * data.n)
    valid_end = int(0.9 * data.n)
    data._split(train_end, valid_end, data.n)


def inverse_transform_targets(values: Any, center: Sequence[float], scale: Sequence[float]) -> Any:
    """Inverse-transform the last target dimension for NumPy or torch values."""
    if torch.is_tensor(values):
        center_t = torch.as_tensor(center, dtype=values.dtype, device=values.device)
        scale_t = torch.as_tensor(scale, dtype=values.dtype, device=values.device)
        return values * scale_t + center_t
    values_np = np.asarray(values)
    return values_np * np.asarray(scale) + np.asarray(center)


def bounded_percentage_loss_physical(
    target_physical: torch.Tensor,
    prediction_physical: torch.Tensor,
    tau: Sequence[float] | torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the A3 bounded percentage loss on the physical kW-equivalent scale."""
    if valid_mask is None:
        valid = torch.isfinite(target_physical) & torch.isfinite(prediction_physical)
    else:
        valid = valid_mask.to(dtype=torch.bool)
        valid = valid & torch.isfinite(target_physical) & torch.isfinite(prediction_physical)
    if not torch.any(valid):
        return prediction_physical.sum() * 0.0
    safe_target = torch.where(valid, target_physical, torch.zeros_like(target_physical))
    safe_prediction = torch.where(valid, prediction_physical, torch.zeros_like(prediction_physical))
    tau_t = torch.as_tensor(tau, dtype=target_physical.dtype, device=target_physical.device)
    denominator = torch.maximum(torch.abs(safe_target), tau_t)
    terms = torch.abs(safe_prediction - safe_target) / denominator
    return terms.masked_select(valid).mean() * 100.0


def build_a3_protocol_manifest(
    scaler: dict[str, Any],
    target_tau: dict[str, Any],
    *,
    test_materialized: bool,
    test_accessed: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "point_training_recipe": A3_RECIPE,
        "target_loss": "target_wise_bounded_percentage_physical_space",
        "target_scaler": "training_only_median_iqr_over_1.349",
        "retain_zero_labels": True,
        "retain_extreme_finite_labels": True,
        "clip_or_winsorize_targets": False,
        "scaler_sha256": scaler["scaler_sha256"],
        "target_tau_sha256": target_tau["target_tau_sha256"],
        "test_materialized": bool(test_materialized),
        "test_accessed": bool(test_accessed),
    }
    payload["a3_protocol_sha256"] = sha256_json(payload)
    return payload
