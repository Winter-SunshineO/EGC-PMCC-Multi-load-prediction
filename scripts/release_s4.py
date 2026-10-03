"""S4 inference timing and cached CECM diagnostics for release runs."""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from release_common import (completed_record, require, sha256, specs, selected_specs,
                            read_json, atomic_json, test_record)
from release_kernels import latency_input, measure_blocks, parameter_rows, seasons
from release_analysis import save_summary
from metrics import LOAD_NAMES


def cecm_tables(args, rows, out):
    parameters,effects,coverage=[],[],[]
    for spec in rows:
        if spec['stage']!='main': continue
        result=Path(spec['unit'])/'result'
        path=result/'cecm_parameters.json'
        if not path.exists(): continue
        snap=read_json(path)
        if snap.get('status')=='not_applicable': continue
        if 'cecm_gate' not in snap: continue
        meta={k:spec[k] for k in ('method','horizon','seed')}
        meta['conditioning']=snap.get('conditioning','static')
        parameters.extend({**meta,**r} for r in parameter_rows(snap))
        datasets=[('train_audit',result/'train_cecm_correction.npy',result/'train_forecast_origins.csv'),
                  ('validation',result/'val_cecm_correction.npy',result/'forecast_origins.csv')]
        test=args.run_root/'main/locked_test'/spec['run_key']
        if (test/'test_cecm_correction.npy').exists():
            test_record(args.run_root,spec)
            datasets.append(('test',test/'test_cecm_correction.npy',test/'forecast_origins.csv'))
        for split,correction_path,origin_path in datasets:
            require(correction_path.exists() and origin_path.exists(),'Missing saved CECM diagnostics')
            values=np.load(correction_path,allow_pickle=False,mmap_mode='r')
            origins=pd.read_csv(origin_path)
            assigned=seasons(origins.origin_timestamp)
            require(values.shape==(len(origins),spec['horizon'],3) and np.isfinite(values).all(),'Invalid correction array')
            require(np.array_equal(assigned,origins.origin_season),'Season assignment differs')
            coverage.append({**meta,'split':split,'seasons':';'.join(sorted(set(assigned))),
                             'n_origins':len(origins),'correction_sha256':sha256(correction_path)})
            for season in sorted(set(assigned)):
                block=np.asarray(values[np.asarray(assigned==season)],dtype=np.float64)
                for index,load in enumerate(LOAD_NAMES):
                    effects.append({**meta,'split':split,'season':season,'load':load,
                                    'signed_mean':float(block[:,:,index].mean()),
                                    'absolute_mean':float(np.abs(block[:,:,index]).mean()),
                                    'n_origins':len(block),'unit':'kW'})
    for name,records in [('cecm_parameters_by_seed',parameters),('cecm_effect_by_season',effects),('cecm_coverage',coverage)]:
        if records: pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
    if parameters:
        save_summary(pd.DataFrame(parameters),['method','horizon','conditioning','parameter','row','column'],
                     ['value'],out/'cecm_parameters_summary.csv',args)
    if effects:
        save_summary(pd.DataFrame(effects),['method','horizon','conditioning','split','season','load'],
                     ['signed_mean','absolute_mean'],out/'cecm_effect_summary.csv',args)
    return len({(r['method'],r['horizon'],r['seed']) for r in parameters})


def run_s4(args):
    require(args.execute,'S4 measures real inference latency; add --execute')
    rows=selected_specs(args)
    if args.include_s2:
        rows.extend(s for s in selected_specs(args,'S2') if s.get('task',{}).get('phase')=='final')
    require(rows,'No checkpoints match the latency filters')
    records={s['id']:completed_record(s) for s in rows}
    by_id={s['id']:s for s in specs(args.run_root)}
    out=args.run_root/'analysis/S4'
    out.mkdir(parents=True,exist_ok=True)
    raw,per_seed,costs=[],[],[]
    torch.set_num_threads(3)
    cost_cache={}
    def cost(spec):
        if spec['id'] not in cost_cache:
            rec=records.get(spec['id']) or completed_record(spec)
            cost_cache[spec['id']]=rec['wall_seconds']+sum(cost(by_id[k]) for k in spec['dependencies'])
        return cost_cache[spec['id']]
    for spec in rows:
        family='S2' if spec['stage']=='S2' else 'main'
        meta=dict(family=family,method=spec['method'],horizon=spec['horizon'],seed=spec['seed'])
        if spec.get('kind')!='deterministic_physical':
            costs.append({**meta,'child_wall_seconds':records[spec['id']]['wall_seconds'],
                          'cumulative_parent_wall_seconds':cost(spec)})
        checkpoint=Path(spec['unit'])/'model/best.pt'
        entry={**spec,'family':family,'checkpoint':str(checkpoint) if checkpoint.exists() else None,
               'checkpoint_sha256':sha256(checkpoint) if checkpoint.exists() else None}
        path=out/'latency_records'/(spec['id'].replace('/','__')+'.json')
        if path.exists():
            require(args.resume,'Existing S4 timings require --resume')
            timing=read_json(path)
            require(timing['checkpoint_sha256']==entry['checkpoint_sha256'],'Latency checkpoint differs')
        else:
            call,device,shape,first=latency_input(entry)
            measurements=measure_blocks(call,device,**(dict(warmups=1,blocks=1,repeats=1) if args.smoke else {}))
            timing=dict(**meta,checkpoint_sha256=entry['checkpoint_sha256'],device=str(device),
                        input_shape=list(shape),first_origin=first,smoke=args.smoke,measurements=measurements)
            atomic_json(path,timing)
            del call
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        values=timing['measurements']
        raw.extend({**meta,**v} for v in values)
        per_seed.append({**meta,'batch_latency_ms':float(np.mean([v['batch_latency_ms'] for v in values])),
                         'per_origin_latency_ms':float(np.mean([v['batch_latency_ms'] for v in values]))/256})
        print('Latency:',spec['id'],flush=True)
    pd.DataFrame(raw).to_csv(out/'latency_measurements.csv',index=False)
    pd.DataFrame(per_seed).to_csv(out/'latency_by_seed.csv',index=False)
    save_summary(pd.DataFrame(per_seed),['family','method','horizon'],['batch_latency_ms','per_origin_latency_ms'],
                 out/'latency_summary.csv',args)
    if costs:
        pd.DataFrame(costs).to_csv(out/'training_time_by_seed.csv',index=False)
        save_summary(pd.DataFrame(costs),['family','method','horizon'],
                     ['child_wall_seconds','cumulative_parent_wall_seconds'],out/'training_time_summary.csv',args)
    cecm_count=cecm_tables(args,rows,out)
    atomic_json(out/'s4_manifest.json',dict(status='completed',latency_units=len(rows),smoke=args.smoke,
                cecm_units=cecm_count,additional_test_forwards=0,
                timing_scope='observed_device_load; run on an idle device for the formal benchmark'))
    print(f'Wrote S4 timing and cached CECM diagnostics to {out}')
