"""Validation-only S1/S2/S3 reports for the frozen published recipes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from release_common import ROOT, read_json, completed_record, selected_specs, specs, require, atomic_json
from release_analysis import METRICS, save_summary, load_prediction


def report(args):
    require(args.stage in ('S1','S2','S3'), 'Choose --stage S1, S2, or S3')
    rows=selected_specs(args,args.stage)
    require(rows,'No supplementary runs match the filters')
    all_specs={s['id']:s for s in specs(args.run_root)}
    out=args.run_root/'analysis'/args.stage
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    cache={}
    def cost(spec):
        if spec['id'] not in cache:
            rec=completed_record(spec)
            if 'reuse_unit' in spec:
                value=cost(all_specs[spec['dependencies'][0]])
            else:
                value=rec['wall_seconds']+sum(cost(all_specs[p]) for p in spec['dependencies'])
            cache[spec['id']]=value
        return cache[spec['id']]
    for spec in rows:
        rec=completed_record(spec)
        metrics=read_json(Path(spec['unit'])/'result/val_metrics_full.json')['overall']
        records.append(dict(id=spec['id'],method=spec['method'],horizon=spec['horizon'],seed=spec['seed'],
                            unit=spec['unit'],smoke=args.smoke,reused='reuse_unit' in spec,
                            phase=spec.get('task',{}).get('phase',args.stage),
                            config=json.dumps(spec['config'],sort_keys=True),
                            child_wall_seconds=rec['wall_seconds'],cumulative_parent_wall_seconds=cost(spec),**metrics))
    frame=pd.DataFrame(records)
    frame.to_csv(out/'validation_runs.csv',index=False)
    if args.stage=='S1':
        order=read_json(ROOT/'configs/analysis.json')['s1_candidate_order']
        frame['candidate_order']=frame.id.map(order)
        selected=frame.sort_values(['mape','candidate_order']).groupby(['method','horizon'],as_index=False).first()
        selected.to_csv(out/'selected_configs.csv',index=False)
        frame.groupby(['method','horizon']).agg(nominal_trials=('id','count'),physical_units=('unit','nunique'),
            reused_trials=('reused','sum'),referenced_child_seconds=('child_wall_seconds','sum')).to_csv(out/'budget_summary.csv')
        frame.to_csv(out/'validation_by_trial.csv',index=False)
    if args.stage=='S2':
        final=frame[frame.phase=='final']
        if len(final):
            save_summary(final,['method','horizon'],METRICS+['child_wall_seconds'],
                         out/'final_validation_summary.csv',args)
        hpo=frame[frame.phase=='hpo'].copy()
        if len(hpo):
            hpo['candidate_order']=hpo.id.map(lambda k:all_specs[k]['config']['candidate_order'])
            hpo.sort_values(['mape','candidate_order']).groupby(['method','horizon'],as_index=False).first().to_csv(
                out/'observed_hpo_winners.csv',index=False)
        registry=read_json(ROOT/'configs/analysis.json')['s2_parameter_match_registry']
        pd.DataFrame(registry).to_csv(out/'parameter_match_registry.csv',index=False)
    if args.stage=='S3':
        targets=read_json(ROOT/'configs/analysis.json')['s3_targets']
        by_id={r['id']:r for r in records}
        paired=[]
        for target in targets:
            key=target['spec_id']
            if key not in by_id: continue
            row=by_id[key]
            ref=all_specs[f"main/{target['method']}/{target['horizon']}h/seed_{target['seed']}"]
            completed_record(ref)
            y,pred,mask,origin=load_prediction(Path(row['unit'])/'result','val',target['horizon'])
            ry,rp,rm,ro=load_prediction(Path(ref['unit'])/'result','val',target['horizon'])
            require(np.array_equal(mask,rm) and np.array_equal(origin.origin_index,ro.origin_index),'S3 reference alignment differs')
            ref_mape=read_json(Path(ref['unit'])/'result/val_metrics_full.json')['overall']['mape']
            paired.append({**row,**target,'level':json.dumps(target['level']), 'reference_mape':ref_mape,
                           'delta_mape_pp':row['mape']-ref_mape,'origins_aligned':True})
        paired=pd.DataFrame(paired)
        require(len(paired)>0,'No S3 target nodes match the filters')
        paired.to_csv(out/'sensitivity_paired_deltas.csv',index=False)
        summary=save_summary(paired,['method','horizon','parameter','level_order','level'],
                             METRICS+['delta_mape_pp'],out/'sensitivity_summary.csv',args)
        labels=[]
        for (h,parameter),group in paired.groupby(['horizon','parameter']):
            levels=list(group.groupby('level_order'))
            complete=len(group)==9 and all(set(g.seed)=={2021,2023,2024} for _,g in levels)
            means=[g.mape.mean() for _,g in levels]
            deltas=[g.delta_mape_pp.to_numpy() for _,g in levels]
            clipped=bool(group.boundary_clipped.any())
            status=('inconclusive' if args.smoke or clipped or not complete else
                    'sensitive' if max(means)-min(means)>.5 or any(d.mean()>.5 for d in deltas) else
                    'stable' if all((d<=.5).sum()>=2 for d in deltas) else 'inconclusive')
            labels.append(dict(horizon=h,parameter=parameter,stability_label=status,boundary_clipped=clipped,
                               factor_mean_range_mape_pp=max(means)-min(means)))
        summary=summary.merge(pd.DataFrame(labels),on=['horizon','parameter'])
        summary.to_csv(out/'sensitivity_summary.csv',index=False)
    atomic_json(out/'report_manifest.json',dict(stage=args.stage,status='completed',units=len(rows),
                smoke=args.smoke,test_accessed=False,main_reselection=False,
                mode='published_frozen_configuration_retraining'))
    print(f'Wrote {args.stage} validation metrics and budget report to {out}')
