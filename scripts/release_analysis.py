"""Offline summaries of portable v7 runs; no model forwards."""
from pathlib import Path
import numpy as np
import pandas as pd
from release_common import (ROOT, atomic_json, completed_record, protocol, read_json,
                            require, selected_specs, test_record)
from release_statistics import summarize, origin_loss, mbb_seed_aggregate, holm_adjust
from metrics import metric_overall, metric_by_load, metric_by_horizon, LOAD_NAMES
from quantile_calibration import spread_scale_quantiles
from quantile_metrics import interval_metrics, crps_quantile

METRICS = ['mape','mae','rmse','smape','wape','nrmse_mean','corr']
INTERVAL_METRICS = ['picp','ace','pinaw','mean_width','winkler','crps_quantile']


def save_summary(frame, groups, metrics, path, args):
    expected = tuple(args.seeds or (2021,2023,2024))
    result = summarize(frame, groups, metrics, expected_seeds=expected)
    result['smoke'] = bool(args.smoke)
    result['paper_seed_set'] = not args.smoke and set(expected) == {2021,2023,2024}
    result.to_csv(path, index=False)
    return result


def load_prediction(result, prefix, horizon):
    y = np.load(result/f'{prefix}_y_true.npy', allow_pickle=False)
    pred = np.load(result/f'{prefix}_predict_value.npy', allow_pickle=False)
    mask = np.load(result/f'{prefix}_label_valid_mask.npy', allow_pickle=False).astype(bool)
    origins = pd.read_csv(result/'forecast_origins.csv')
    require(y.shape == pred.shape == mask.shape == (2460-horizon,horizon,3),
            f'Unexpected prediction shape: {result}')
    require(np.isfinite(pred).all() and not np.any(mask & ~np.isfinite(y)), 'Nonfinite valid values')
    require(len(origins)==len(y) and np.all(np.diff(origins.origin_index)==1), 'Misaligned origins')
    return y,pred,mask,origins


def analyze(args):
    rows = selected_specs(args)
    require(rows, 'No main units match the filters')
    split, prefix = args.split, 'val' if args.split=='validation' else 'test'
    out = args.run_root/'analysis'/split
    out.mkdir(parents=True, exist_ok=True)
    point,loads,leads,intervals = [],[],[],[]
    losses,references,path = {},{},protocol()['point_path']
    for spec in rows:
        completed_record(spec)
        result = Path(spec['unit'])/'result' if split=='validation' else test_record(args.run_root,spec)
        y,pred,mask,origins = load_prediction(result,prefix,spec['horizon'])
        h = spec['horizon']
        if h in references:
            ry,rm,ro = references[h]
            require(np.array_equal(mask,rm) and np.array_equal(origins.origin_index,ro.origin_index)
                    and np.allclose(y,ry,atol=.02,rtol=1e-6,equal_nan=True), 'Cross-model target alignment failed')
        else:
            references[h] = (y,mask,origins)
        meta = {k:spec[k] for k in ('method','horizon','seed')}
        meta.update(split=split,smoke=args.smoke,unit=str(spec['unit']))
        point.append({**meta,**metric_overall(y,pred,mask)})
        loads.extend({**meta,**r} for r in metric_by_load(y,pred,mask=mask))
        leads.extend({**meta,**r} for r in metric_by_horizon(y,pred,mask=mask))
        losses[spec['run_key']] = origin_loss(y,pred,mask)
        if spec['method']=='p4':
            qmeta = read_json(result/('val_quantile_metadata.json' if split=='validation' else 'quantile_metadata.json'))
            levels = np.asarray(qmeta['quantiles'],dtype=float)
            raw = np.load(result/('val_quantile_value.npy' if split=='validation' else 'test_quantile_raw.npy'),allow_pickle=False)
            calibrated = (spread_scale_quantiles(raw,levels,protocol()['calibration_scale'])
                          if split=='validation' else np.load(result/'test_quantile_value.npy',allow_pickle=False))
            if split=='test':
                require(qmeta['scale_application_count']==1 and qmeta['selected']['scale']==1.25,
                        'Expected one frozen calibration application')
            for variant,q in [('raw',raw),('calibrated',calibrated)]:
                require(q.shape==(*y.shape,len(levels)) and np.isfinite(q).all()
                        and (np.diff(q,axis=-1)>=0).all(), 'Invalid saved quantiles')
                for load,index in [('overall',None),*zip(LOAD_NAMES,range(3))]:
                    yt,yq,mk = (y,q,mask) if index is None else (y[:,:,index],q[:,:,index],mask[:,:,index])
                    intervals.append({**meta,'variant':variant,'load':load,
                                      **interval_metrics(yt,yq,levels,mask=mk),
                                      'crps_quantile':crps_quantile(yt,yq,levels,mask=mk)})
    frames=[]
    for records in (point,loads,leads):
        frame=pd.DataFrame(records)
        picked=frame[frame.apply(lambda r:r.method==path[str(r.horizon)],axis=1)].copy()
        picked['method']='selected_point_path'
        frames.append(pd.concat([frame,picked],ignore_index=True))
    for frame,name,groups in zip(frames,('point','per_load','lead_time'),
                                (['method','horizon'],['method','horizon','load'],['method','horizon','horizon_step'])):
        frame.to_csv(out/f'{name}_by_seed.csv',index=False)
        save_summary(frame,groups,METRICS,out/f'{name}_summary.csv',args)
    if intervals:
        frame=pd.DataFrame(intervals)
        frame.to_csv(out/'calibration_by_seed.csv',index=False)
        save_summary(frame,['method','horizon','variant','load'],INTERVAL_METRICS,out/'calibration_summary.csv',args)
    inference_status='requires_test_all_four_horizons_and_three_seeds'
    keys=[f'{m}/{h}h/seed_{s}' for h in (24,48,72,96) for m in (path[str(h)],'gru') for s in (2021,2023,2024)]
    if split=='test' and not args.smoke and all(k in losses for k in keys):
        from run_dm_test import dm_test
        dm_rows,mbb_rows=[],[]
        for h in (24,48,72,96):
            ref=np.stack([losses[f'{path[str(h)]}/{h}h/seed_{s}'] for s in (2021,2023,2024)])
            base=np.stack([losses[f'gru/{h}h/seed_{s}'] for s in (2021,2023,2024)])
            diff=ref-base
            dm=dm_test(ref.mean(axis=0),base.mean(axis=0),h-1)
            dm_rows.append(dict(horizon=h,comparator='gru',test_family='primary_dm',
                                p_raw=dm['p_one_sided_reference_lower_loss'],**dm))
            mbb_rows.append(dict(horizon=h,comparator='gru',test_family='primary_mbb',**mbb_seed_aggregate(list(diff),h)))
            tab=references[h][2].copy()
            for i,s in enumerate((2021,2023,2024)): tab[f'difference_seed_{s}']=diff[i]
            tab.to_csv(out/f'origin_loss_differences_{h}h.csv',index=False)
        for records,name in ((dm_rows,'dm_holm_results'),(mbb_rows,'mbb_holm_results')):
            holm_adjust(records,'p_raw')
            pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
        inference_status='completed'
    atomic_json(out/'analysis_manifest.json',dict(status='completed',split=split,smoke=args.smoke,
                physical_units=len(rows),additional_model_forwards=0,additional_test_accesses=0,
                registered_inference=inference_status))
    print(f'Wrote {split} summaries for {len(rows)} physical units to {out}')


def selection_report(args):
    rows=selected_specs(args,'selection')
    require(rows,'No selection runs match the filters')
    out=args.run_root/'analysis/selection'
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for order,spec in enumerate(rows):
        record=completed_record(spec)
        metrics=read_json(Path(spec['unit'])/'result/val_metrics_full.json')['overall']
        records.append(dict(method=spec['method'],horizon=spec['horizon'],trial_id=spec['id'].split('/',1)[1],
                            candidate_order=order,validation_mape=metrics['mape'],
                            child_wall_seconds=record['wall_seconds'],smoke=args.smoke))
    frame=pd.DataFrame(records)
    frame.to_csv(out/'replayed_trials.csv',index=False)
    winners=frame.sort_values(['validation_mape','candidate_order']).groupby(['method','horizon'],as_index=False).first()
    frozen=pd.DataFrame(read_json(ROOT/'configs/analysis.json')['selection'])
    winners=winners.merge(frozen[['method','horizon','selected_trial_id']],on=['method','horizon'],how='left')
    winners['same_as_published']=winners.trial_id==winners.selected_trial_id
    winners.to_csv(out/'observed_vs_published_selection.csv',index=False)
    atomic_json(out/'selection_manifest.json',dict(replayed_trials=len(frame),smoke=args.smoke,
                published_configs_modified=False,mode='published_frozen_configuration_retraining'))
    print(f'Wrote selection replay comparison to {out}; main recipes remain frozen')


def supplement_report(args):
    from release_supplement import report
    report(args)
