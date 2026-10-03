"""One registered v7 S2 common-scaffold fit; validation only, no legacy executor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from release_common import atomic_json, require, verify_runtime, base, origin_table, read_json, specs
from reviewer_cells import MatchedRecurrentForecaster, fix_seed, evaluate_validation
from util import DataLoaderS
from metrics import metric_overall
from training_protocols import (
    apply_target_robust_scaler, bounded_percentage_loss_physical, build_a3_protocol_manifest,
    fit_target_bounded_tau, fit_target_robust_scaler, inverse_transform_targets,
)


def check_task(task, plan):
    candidates = [s["config"] for s in specs(os.environ["EGC_RUN_ROOT"])
                  if s["stage"] == "S2" and s.get("task", {}).get("phase") == "hpo"]
    cfg = task["config"]
    require(task["phase"] in ("hpo", "final"), "unsupported S2 phase")
    require(cfg in candidates, "S2 configuration is not a frozen candidate")
    require(task["seed"] in (2021, 2023, 2024), "wrong S2 seed")
    require(task["horizon"] == cfg["horizon"] and task["method"] == cfg["variant"], "S2 candidate metadata differs")
    if task["phase"] == "hpo":
        require(task["seed"] == 2021, "wrong S2 HPO seed")
    registry = next(r for r in plan["s2_parameter_match_registry"]
                    if (r["variant"], r["horizon"]) == (cfg["variant"], cfg["horizon"]))
    require(registry["status"] == "matched", "failed_parameter_match")
    return registry


def run(path):
    verify_runtime()
    task = json.loads(path.read_text(encoding="utf-8"))
    plan = read_json(ROOT / "configs/analysis.json")
    match = check_task(task, plan)
    cfg = task["config"]
    unit, result = path.parent, path.parent / "result"
    require(not result.exists(), "refusing repeated S2 fit")
    result.mkdir()
    (unit / "model").mkdir()
    os.environ["EGC_MAPE_ZERO_POLICY"] = "exclude"
    torch.set_num_threads(3)
    fix_seed(task["seed"])
    device = torch.device(os.environ.get("EGC_DEVICE", "cuda:0"))
    model = MatchedRecurrentForecaster(
        cfg["variant"], cfg["hidden_size"], cfg["horizon"], cfg["period_set"],
        common_hidden=cfg["common_hidden"], dropout=cfg["dropout"]).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(total == match["total_parameters"] and model.update_block_parameters() == match["update_parameters"],
            "registered parameter counts differ")
    data = DataLoaderS(str(base.DATA), .8, .1, device, cfg["horizon"], 168, 2,
                       split_policy="embargo", materialize_test=False)
    require(data.test is None, "test tensors must not exist")
    train_end = int(data.train_size * data.n)
    scaler = fit_target_robust_scaler(data.raw_labels, data.label_valid_mask, train_end)
    tau = fit_target_bounded_tau(data.raw_labels, data.label_valid_mask, train_end)
    apply_target_robust_scaler(data, scaler)
    require(data.test is None, "scaler must not materialize test")
    flags = dict(seed=task["seed"], evaluation_mode="validation-only",
                 test_accessed=False, test_materialized=False)
    atomic_json(result / "config.json", {**cfg, **flags, "trial_id": task["task_id"],
                "config_hash": base.canonical_hash(cfg), "model_parameters": total,
                "update_block_parameters": model.update_block_parameters(), "point_training_recipe": "a3"})
    atomic_json(result / "split_manifest.json", {**data.split_metadata, **flags})
    atomic_json(result / "scaler.json", scaler)
    atomic_json(result / "target_tau.json", tau)
    atomic_json(result / "a3_protocol.json", build_a3_protocol_manifest(
        scaler, tau, test_materialized=False, test_accessed=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    history, steps, stale, best, best_epoch = [], 0, 0, float("inf"), None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    for epoch in range(1, (1 if os.environ.get("EGC_SMOKE") == "1" else cfg["epochs"]) + 1):
        model.train()
        losses = []
        for x, y, mask in data.get_batches(data.train[0], data.train[1], 256, True,
                                          masks=data.train_label_mask):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x.to(device))
            require(torch.isfinite(pred).all().item(), "nonfinite S2 predictions")
            target = inverse_transform_targets(y[:, :, :3].to(device), scaler["center"], scaler["scale"])
            values = inverse_transform_targets(pred, scaler["center"], scaler["scale"])
            loss = bounded_percentage_loss_physical(target, values, tau["tau"], mask.to(device))
            require(torch.isfinite(loss).item(), "nonfinite S2 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
            if steps >= (2 if os.environ.get("EGC_SMOKE") == "1" else 12000):
                break
        mape, mae, corr, _, prediction, _ = evaluate_validation(data, model, device, 256, cfg["horizon"])
        require(np.isfinite(prediction).all() and np.isfinite(mape), "nonfinite validation output")
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_mape=mape,
                            validation_mae=mae, optimizer_steps=steps, lr=cfg["lr"]))
        print(f"epoch={epoch} mape={mape:.7f} updates={steps}", flush=True)
        if mape < best:
            best, best_epoch, stale = mape, epoch, 0
            torch.save(model.state_dict(), unit / "model/best.pt")
        else:
            stale += 1
        if stale >= 8 or steps >= 12000:
            break
    torch.save(model.state_dict(), unit / "model/last.pt")
    model.load_state_dict(torch.load(unit / "model/best.pt", map_location=device, weights_only=True))
    _, _, _, y, prediction, mask = evaluate_validation(data, model, device, 256, cfg["horizon"])
    require(np.isfinite(prediction).all(), "nonfinite final S2 output")
    for name, value in (("val_y_true.npy", y), ("val_predict_value.npy", prediction),
                        ("val_label_valid_mask.npy", mask)):
        np.save(result / name, value)
    overall = metric_overall(y, prediction, mask)
    atomic_json(result / "val_metrics_full.json", dict(overall=overall, split="validation", **flags))
    atomic_json(result / "validation_history.json", history)
    atomic_json(result / "run_summary.json", dict(
        **flags, status="completed", trial_id=task["task_id"], model_parameters=total,
        epochs_completed=len(history), best_epoch=best_epoch, optimizer_steps_completed=steps,
        training_wall_seconds=time.perf_counter()-started, cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
        validation_mape=overall["mape"], optimizer="AdamW", scheduler="none_shared_for_all_four_cells"))
    origins = origin_table(data.split_metadata['target_end_index_ranges']['validation'][0], cfg['horizon'], len(y))
    origins.to_csv(result / "forecast_origins.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    run(parser.parse_args().task.resolve())
