"""Paths, integrity checks, and run records for the portable v7 release."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/preprocessed_forecasting_v7/dataset_input.csv'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode('utf-8')).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def resolve(value, run_root):
    if isinstance(value, str):
        return value.replace('{repo}', str(ROOT)).replace('{run}', str(Path(run_root).resolve())).replace('{python}', sys.executable)
    if isinstance(value, list):
        return [resolve(v, run_root) for v in value]
    if isinstance(value, dict):
        return {resolve(k, run_root): resolve(v, run_root) for k, v in value.items()}
    return value


def specs(run_root):
    return resolve(read_json(ROOT / 'configs/run_specs.json')['runs'], run_root)


def protocol():
    return read_json(ROOT / 'configs/protocol.json')


def verify_data():
    expected = read_json(ROOT / 'configs/data_checksums.json')
    verified = 0
    for relative, digest in expected.items():
        # The merged upstream records are an optional local preprocessing
        # source. They are intentionally excluded from the Git release.
        if relative == 'data/merged_data21-23.csv' and not (ROOT / relative).exists():
            continue
        require((ROOT / relative).is_file(), f'Missing data file: {relative}')
        require(sha256(ROOT / relative) == digest, f'Data checksum changed: {relative}')
        verified += 1
    return verified


def source_hashes():
    paths = [*ROOT.glob('*.py'), *(ROOT / 'layers').glob('*.py'),
             *(ROOT / 'scripts').glob('*.py'), *(ROOT / 'configs').glob('*.json')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def verify_runtime():
    """A worker must use the code/config fingerprint registered for its run."""
    path = Path(os.environ['EGC_RUN_ROOT']) / 'release_run.json'
    saved = read_json(path)
    require(saved['source_sha256'] == source_hashes(), 'Code or configuration changed inside this run; use a new run root')
    return saved


def completed_record(spec):
    if 'reuse_unit' in spec:
        by_id = {s['id']: s for s in specs(os.environ['EGC_RUN_ROOT'])}
        require(len(spec['dependencies']) == 1, 'An alias must name one physical source')
        parent = by_id[spec['dependencies'][0]]
        require(Path(parent['unit']).resolve() == Path(spec['unit']).resolve(), 'Alias path differs')
        return completed_record(parent)
    record = Path(spec['unit']) / 'release_execution.json'
    require(record.exists(), f"Missing completed prerequisite: {spec['id']}")
    data = read_json(record)
    require(data.get('status') == 'completed', f"Prerequisite is not complete: {spec['id']}")
    require(data.get('spec_id') == spec['id'], f"Run identity differs: {spec['id']}")
    for relative, digest in data['output_sha256'].items():
        require(sha256(Path(spec['unit']) / relative) == digest, f"Output changed: {spec['id']}/{relative}")
    return data


def selected_specs(args, stage='main'):
    return [s for s in specs(args.run_root) if s['stage'] == stage
            and (not args.horizons or s['horizon'] in args.horizons)
            and (not args.seeds or s['seed'] in args.seeds)
            and (not args.methods or s['method'] in args.methods)]


def test_record(run_root, spec):
    output = Path(run_root) / 'main/locked_test' / spec['run_key']
    audit = read_json(Path(run_root) / 'test_access_audit.json')
    matches = [r for r in audit['attempts'] if r['spec_id'] == spec['id']]
    require(len(matches) == 1 and matches[0]['status'] == 'completed',
            f"Missing completed test export: {spec['id']}")
    for name, digest in matches[0]['output_sha256'].items():
        require(sha256(output / name) == digest, f"Test output changed: {spec['id']}/{name}")
    return output


def origin_table(first_end, horizon, count, split='validation'):
    import numpy as np
    import pandas as pd
    start = pd.Timestamp(read_json(DATA.with_name('preprocessing_metadata.json'))['time_start'])
    end = np.arange(count) + first_end
    def season(month):
        return 'DJF' if month in (12, 1, 2) else 'MAM' if month in (3, 4, 5) else 'JJA' if month in (6, 7, 8) else 'SON'
    columns = {'sample_index': np.arange(count)}
    for prefix, index in [('origin', end-horizon), ('target_start', end-horizon+1), ('target_end', end)]:
        timestamps = start + pd.to_timedelta(index, unit='h')
        columns.update({prefix+'_index': index, prefix+'_timestamp': [t.isoformat() for t in timestamps],
                        prefix+'_season': [season(t.month) for t in timestamps]})
    return pd.DataFrame({**columns, 'horizon_h': horizon, 'split': split})


base = SimpleNamespace(DATA=DATA, canonical_hash=canonical_hash, sha256=sha256,
                       POINT_STAGES=('p0', 'p2a', 'p2b'))
