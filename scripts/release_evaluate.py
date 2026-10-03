"""Explicit, audited inference after frozen-configuration training."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from release_common import (DATA, ROOT, atomic_json, completed_record, protocol, read_json,
                            require, sha256, specs, verify_data, origin_table)
import release_kernels as kernels
import train
from quantile_calibration import spread_scale_quantiles


def float32_replay_identity(reference: np.ndarray, replay: np.ndarray) -> dict:
    """Audit that an independently replayed parent remains numerically identical."""
    if reference.shape != replay.shape:
        raise ValueError(f'float32 replay shape mismatch: {reference.shape} != {replay.shape}')
    if not np.isfinite(reference).all() or not np.isfinite(replay).all():
        raise ValueError('float32 replay contains non-finite values')
    policy = {
        'absolute_tolerance': 1e-6,
        'relative_tolerance': 1e-6,
        'max_float32_ulps': 8.0,
        'scope': 'independent float32 replay of the same frozen parent checkpoint',
    }
    atol, rtol, max_ulps = policy['absolute_tolerance'], policy['relative_tolerance'], policy['max_float32_ulps']
    reference32 = np.asarray(reference, dtype=np.float32)
    replay32 = np.asarray(replay, dtype=np.float32)
    difference = np.abs(reference32.astype(np.float64) - replay32.astype(np.float64))
    scale = np.maximum(np.abs(reference32.astype(np.float64)), np.abs(replay32.astype(np.float64)))
    relative = np.divide(difference, np.maximum(scale, 1.0))
    spacing = np.maximum(np.spacing(np.abs(reference32)).astype(np.float64),
                         np.spacing(np.abs(replay32)).astype(np.float64))
    ulps = np.divide(difference, spacing, out=np.zeros_like(difference), where=spacing > 0)
    passed = bool(np.all((difference <= atol + rtol * scale) & ((difference <= atol) | (ulps <= max_ulps))))
    return {
        'status': 'PASS' if passed else 'FAIL',
        'max_abs_diff': float(difference.max(initial=0.0)),
        'max_relative_diff': float(relative.max(initial=0.0)),
        'max_float32_ulps': float(ulps.max(initial=0.0)),
        'nonzero_count': int(np.count_nonzero(difference)),
        'value_count': int(difference.size),
        'policy': policy,
        'test_accessed': False,
    }


def make_entry(spec):
    checkpoint=Path(spec['unit'])/'model/best.pt'
    return {**spec, 'checkpoint':str(checkpoint) if checkpoint.exists() else None,
            'checkpoint_sha256':sha256(checkpoint) if checkpoint.exists() else None}


def forward(entry, by_id, device, split):
    if entry['method']!='p4':
        return kernels.infer(entry,device,split)
    result=Path(entry['unit'])/'result'
    cfg=read_json(result/'config.json')
    data=train.DataLoaderS(str(DATA),.8,.1,torch.device(device),entry['horizon'],168,cfg['normalize'],
                           split_policy='embargo',materialize_test=split=='test')
    kernels.restore_scaler(data,cfg,result)
    model=train.load_checkpoint_cpu(entry['checkpoint']).eval()
    parent=make_entry(by_id[entry['dependencies'][0]])
    parent_model=train.load_checkpoint_cpu(parent['checkpoint']).eval()
    x,y=data.test if split=='test' else data.valid
    mask=data.test_label_mask if split=='test' else data.valid_label_mask
    with kernels.matching_execution_flags(model,parent_model):
        values=train.plow(data,x,y[:,:,:3],model.to(device).eval(),256,entry['horizon'],3,
                          label_mask=mask,return_mask=True,return_cecm_correction=True)
    y,pred,q,mask,correction=[v.detach().cpu().numpy() if v is not None else None for v in values]
    return dict(y=y,pred=pred,q=q,mask=mask,correction=correction,
                levels=[float(v) for v in model.quantiles],
                first_target_end=int(data.split_metadata['target_end_index_ranges'][split][0]))


def evaluate(args):
    all_specs=specs(args.run_root)
    by_id={s['id']:s for s in all_specs}
    selected=[s for s in all_specs if s['stage']=='main'
              and (not args.horizons or s['horizon'] in args.horizons)
              and (not args.seeds or s['seed'] in args.seeds)
              and (not args.methods or s['method'] in args.methods)]
    require(selected,'No main runs match evaluation filters')
    # P4 needs its separately saved point parent for the identity check.
    needed={s['id'] for s in selected}
    for s in selected:
        if s['method']=='p4': needed.update(s['dependencies'])
    selected=[s for s in all_specs if s['id'] in needed]
    if not args.execute:
        print(f'Would evaluate {len(selected)} frozen main units on test. Add --execute to perform the audited pass.')
        return
    verify_data()
    torch.set_num_threads(3)
    entries=[make_entry(s) for s in selected]
    for spec in selected: completed_record(spec)
    audit_path=args.run_root/'test_access_audit.json'
    audit=read_json(audit_path) if audit_path.exists() else {'format_version':1,'attempts':[]}
    for entry in entries:
        output=args.run_root/'main/locked_test'/entry['run_key']
        previous=[r for r in audit['attempts'] if r['spec_id']==entry['id']]
        if previous:
            require(args.resume and previous[-1]['status']=='completed','Repeated or failed test access requires a new run root')
            for relative,digest in previous[-1]['output_sha256'].items():
                require(sha256(output/relative)==digest,'Saved test export changed')
            continue
        require(not output.exists(),f'Untracked test output exists: {output}')
        row=dict(spec_id=entry['id'],status='attempted',started_utc=datetime.now(timezone.utc).isoformat(),
                 checkpoint_sha256=entry['checkpoint_sha256'],smoke=args.smoke)
        audit['attempts'].append(row)
        atomic_json(audit_path,audit)
        try:
            print('Test:',entry['run_key'],flush=True)
            values=forward(entry,by_id,args.device,'test')
            y,pred,mask=values['y'],values['pred'],values['mask'].astype(bool)
            shape=(2460-entry['horizon'],entry['horizon'],3)
            require(y.shape==pred.shape==mask.shape==shape and np.isfinite(pred).all(),'Test shape/finite check failed')
            require(not np.any(mask & ~np.isfinite(y)),'Invalid test label mask')
            output.mkdir(parents=True)
            for name,value in [('test_y_true.npy',y),('test_predict_value.npy',pred),('test_label_valid_mask.npy',mask)]:
                np.save(output/name,value)
            if values['correction'] is not None:
                np.save(output/'test_cecm_correction.npy',values['correction'])
            if entry['method']=='p4':
                parent=by_id[entry['dependencies'][0]]
                parent_pred=np.load(args.run_root/'main/locked_test'/parent['run_key']/'test_predict_value.npy')
                identity=float32_replay_identity(parent_pred,pred)
                require(identity['status']=='PASS','P4 median differs from its frozen point parent')
                atomic_json(output/'parent_replay_identity.json',identity)
            if values['q'] is not None:
                q,levels=values['q'],np.asarray(values['levels'])
                require(np.isfinite(q).all() and (np.diff(q,axis=-1)>=0).all(),'Invalid quantiles')
                require(np.array_equal(q[...,int(np.argmin(abs(levels-.5)))],pred),'P4 median identity failed')
                np.save(output/'test_quantile_raw.npy',q)
                np.save(output/'test_quantile_value.npy',spread_scale_quantiles(q,levels,protocol()['calibration_scale']))
                atomic_json(output/'quantile_metadata.json',dict(quantiles=levels.tolist(),scale_application_count=1,
                                                                 selected={'scale':protocol()['calibration_scale']}))
            atomic_json(output/'test_metrics_full.json',train.build_metrics_payload(y,pred,entry['horizon'],'test',mask))
            origin_table(values['first_target_end'],entry['horizon'],len(y),split='test').to_csv(output/'forecast_origins.csv',index=False)
            atomic_json(output/'manifest_entry.json',entry)
            row.update(status='completed',output_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file()})
        except BaseException as exc:
            row.update(status='failed',error=str(exc))
            atomic_json(audit_path,audit)
            raise
        atomic_json(audit_path,audit)
    print('Test exports completed; audit:',audit_path)
