"""Portable execution of the public reviewer recipe; start with `check` or `--help`."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from release_common import (atomic_json, canonical_hash, completed_record, protocol,
                            read_json, require, sha256, source_hashes, specs, verify_data)


def choose(args):
    all_specs = specs(args.run_root)
    selected = [s for s in all_specs if s['stage'] == args.stage
                and (not args.horizons or s['horizon'] in args.horizons)
                and (not args.seeds or s['seed'] in args.seeds)
                and (not args.methods or s['method'] in args.methods)]
    require(selected, 'No runs match the filters')
    # Follow parent links, preserving the published order and seed identities.
    by_id = {s['id']: s for s in all_specs}
    wanted = {s['id'] for s in selected}
    def add_dependencies(key):
        for parent in by_id[key]['dependencies']:
            if by_id[parent]['stage'] == args.stage and parent not in wanted:
                wanted.add(parent)
                add_dependencies(parent)
    for key in list(wanted):
        add_dependencies(key)
    return [s for s in all_specs if s['id'] in wanted], by_id


def adapted_command(spec, args):
    command = list(spec['command'])
    if '--device' in command:
        command[command.index('--device') + 1] = args.device
    if args.smoke:
        changes = {'--epochs': '1', '--max_optimizer_updates': '2', '--max-optimizer-updates': '2',
                   '--n-estimators': '3'}
        for key, value in changes.items():
            if key in command:
                command[command.index(key) + 1] = value
        for key in ('--config_hash', '--config-hash'):
            if key in command:
                command[command.index(key) + 1] = canonical_hash({'smoke_command':command, 'spec':spec['id']})
    return command


def register(args):
    root = args.run_root.resolve()
    require(root != ROOT and ROOT not in root.parents or root.is_relative_to(ROOT / 'runs'),
            'Use runs/<name> inside this checkout, or a separate output directory')
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'release_run.json'
    identity = dict(source_sha256=source_hashes(), device=args.device, smoke=bool(args.smoke),
                    data_sha256=read_json(ROOT / 'configs/data_checksums.json'),
                    protocol='v7_published_recipe_retraining_5.0')
    if path.exists():
        saved = read_json(path)
        require(all(saved[k] == v for k,v in identity.items()),
                'Run fingerprint/device/smoke mode changed; use a new run root')
    else:
        require(not any(root.iterdir()), 'An unregistered output directory must be empty')
        atomic_json(path, dict(**identity, created_utc=datetime.now(timezone.utc).isoformat(),
                              python=sys.version, packages={p:importlib.metadata.version(p) for p in
                              ('torch','numpy','pandas','scipy','scikit-learn','lightgbm')}))


def check(args):
    count = verify_data()
    plan = specs(args.run_root)
    by_id = {s['id']: s for s in plan}
    require(len(plan) == 499 and len(by_id) == 499, 'Unexpected or duplicate run specifications')
    stages = Counter(s['stage'] for s in plan)
    require(stages == dict(selection=76, main=120, S1=96, S2=96, S3=111), 'Protocol matrix differs')
    seen = set()
    for spec in plan:
        require(set(spec['dependencies']) <= seen, f"Dependency ordering error: {spec['id']}")
        if 'command' in spec:
            require(Path(spec['command'][1]).is_file(), f"Missing entry point: {spec['command'][1]}")
            require(Path(spec['unit']).resolve().is_relative_to(args.run_root.resolve()), 'Output escaped run root')
        seen.add(spec['id'])
    import torch
    versions = {p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','scipy','scikit-learn','lightgbm')}
    print(json.dumps(dict(status='passed', data_files=count, recipe_valid=True, packages=versions,
                          cuda_available=torch.cuda.is_available(), trained_models_in_release=False), indent=2))


def train_stage(args):
    selected, by_id = choose(args)
    print(json.dumps(dict(status='ready', stage=args.stage, device=args.device,
                          smoke=args.smoke, execute=args.execute, run_root=str(args.run_root.resolve())), indent=2))
    if not args.execute:
        print('Recipe order validated. Add --execute to run the selected stage.')
        return
    import torch
    require(args.device == 'cpu' or torch.cuda.is_available(),
            'CUDA is unavailable in this Python environment. Activate the CUDA environment or use --device cpu.')
    verify_data()
    register(args)
    os.environ.update(EGC_RUN_ROOT=str(args.run_root.resolve()), EGC_DEVICE=args.device,
                      EGC_SMOKE='1' if args.smoke else '0', EGC_MAPE_ZERO_POLICY='exclude',
                      OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONUNBUFFERED='1',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    env = os.environ.copy()
    for index, spec in enumerate(selected, 1):
        for parent in spec['dependencies']:
            completed_record(by_id[parent])
        if 'reuse_unit' in spec:
            print(f"[{index}/{len(selected)}] reference {spec['id']}", flush=True)
            continue
        unit = Path(spec['unit'])
        command = adapted_command(spec, args)
        record_path = unit / 'release_execution.json'
        if record_path.exists():
            require(args.resume, f'Existing run requires --resume: {unit}')
            previous = completed_record(spec)
            require(previous['command'] == command, 'Resume command differs')
            print(f"[{index}/{len(selected)}] verified {spec['id']}", flush=True)
            continue
        require(not unit.exists(), f'Partial output retained; use a new run root: {unit}')
        unit.mkdir(parents=True)
        if 'task' in spec:
            atomic_json(unit / 'task.json', spec['task'])
        record = dict(spec_id=spec['id'], status='running', command=command, smoke=args.smoke,
                      stage=spec['stage'], method=spec['method'], horizon=spec['horizon'], seed=spec['seed'],
                      started_utc=datetime.now(timezone.utc).isoformat(), test_accessed=False)
        atomic_json(record_path, record)
        started = time.perf_counter()
        print(f"[{index}/{len(selected)}] running {spec['id']}", flush=True)
        try:
            with (unit/'stdout.log').open('w',encoding='utf-8') as out, (unit/'stderr.log').open('w',encoding='utf-8') as err:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                        timeout=args.timeout_seconds or None)
            require(result.returncode == 0, f"Training failed; see {unit / 'stderr.log'}")
            required = ['val_metrics_full.json','val_y_true.npy','val_predict_value.npy',
                        'val_label_valid_mask.npy','config.json','split_manifest.json','forecast_origins.csv']
            require(all((unit/'result'/p).is_file() for p in required), f'Missing output in {unit}')
            split = read_json(unit/'result/split_manifest.json')
            require(split.get('test_accessed') is False and split.get('test_materialized') is False,
                    'Training materialized test data')
            import numpy as np
            prediction=np.load(unit/'result/val_predict_value.npy',allow_pickle=False)
            require(np.isfinite(prediction).all(), 'Nonfinite validation prediction')
            record.update(status='completed', wall_seconds=time.perf_counter()-started,
                          output_sha256={p.relative_to(unit).as_posix():sha256(p) for p in sorted(unit.rglob('*'))
                                         if p.is_file() and p != record_path})
        except BaseException as exc:
            record.update(status='failed', error=str(exc), wall_seconds=time.perf_counter()-started)
            atomic_json(record_path,record)
            raise
        atomic_json(record_path,record)
    print('Requested recipe runs completed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','train','evaluate','analyze','selection-report','supplement-report','s4'))
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/paper')
    parser.add_argument('--stage',choices=('selection','main','S1','S2','S3'),default='main')
    parser.add_argument('--horizons',nargs='+',type=int,choices=(24,48,72,96))
    parser.add_argument('--seeds',nargs='+',type=int,choices=(2021,2023,2024))
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--split',choices=('validation','test'),default='validation',help='Saved arrays to analyze')
    parser.add_argument('--include-s2',action='store_true',help='Include supplementary final models in S4')
    parser.add_argument('--timeout-seconds',type=int,default=0,help='Optional wall-time guard per child; 0 disables it')
    parser.add_argument('--execute',action='store_true',help='Required to train, evaluate test, or benchmark')
    parser.add_argument('--resume',action='store_true',help='Verify and skip completed units; never overwrite a failed run')
    parser.add_argument('--smoke',action='store_true',help='1 epoch/2 updates, 3 trees; use a separate run root')
    args=parser.parse_args()
    args.run_root = args.run_root.resolve()
    require(args.timeout_seconds >= 0, 'Timeout must be nonnegative')
    os.environ.update(EGC_MAPE_ZERO_POLICY='exclude', EGC_RUN_ROOT=str(args.run_root.resolve()), EGC_DEVICE=args.device)
    if args.action=='check': check(args)
    elif args.action=='train': train_stage(args)
    else:
        require((args.run_root/'release_run.json').exists(), 'Train into this run root first')
        record=read_json(args.run_root/'release_run.json')
        require(record['device']==args.device, 'Use the same --device as training')
        require(record['source_sha256']==source_hashes(), 'Source/configuration changed since training')
        require(bool(record['smoke']) == bool(args.smoke),
                'Use --smoke when reading smoke artifacts; keep formal runs in a separate run root')
        smoke_safe_actions = {'analyze', 'selection-report', 'supplement-report', 's4'}
        require(not record['smoke'] or args.action in smoke_safe_actions
                or (args.action == 'evaluate' and args.smoke),
                'Smoke artifacts cannot supply S4 or formal supplementary evidence')
        verify_data()
        if args.action in ('analyze', 'evaluate', 's4'):
            require(args.stage == 'main', 'Use --stage main here; use supplement-report for S1/S2/S3')
        if args.action=='evaluate':
            from release_evaluate import evaluate
            evaluate(args)
        elif args.action=='s4':
            from release_s4 import run_s4
            run_s4(args)
        else:
            from release_analysis import analyze, selection_report, supplement_report
            {'analyze':analyze,'selection-report':selection_report,'supplement-report':supplement_report}[args.action](args)


if __name__=='__main__':
    main()
