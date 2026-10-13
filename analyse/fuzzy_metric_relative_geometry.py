#!/usr/bin/env python3
"""Frozen 12-state geometry diagnosis. No labels enter mapping inference or retrieval.

Distances, IDs, and selected neighbors are persisted before the evaluator opens
112 + 29 whitelisted validation means. There is no training or GP inference.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import datetime
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from analyse.fuzzy_metric_diagnosis_supplement import read, write, csv_write, no_optimization
from analyse.fuzzy_metric_replay import digest, file_hash, tensor_hash
from analyse.joint_configuration_metric_gate import hp_feature_vector
from surrogate.fuzzy_metric import JointMetricModel

OLD = ROOT / 'results/posthoc/fuzzy-metric-stage2-diagnosis-20260910-v1'
SUP = ROOT / 'results/posthoc/fuzzy-metric-stage2-readonly-supplement-20260910-v1'
OUT = ROOT / 'results/posthoc/fuzzy-metric-relative-geometry-20260910-v1'
METHODS = ('regression', 'N2')
METRICS = ('global_order', 'local10_order', 'De_pp', 'Dstar_pp', 'E_pp', 'Drandom_pp', 'De_minus_random_pp')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def order_score(distances, gaps, scale=None):
    """Count unordered pairs; IDs never break ties for scoring.

    Local evaluation passes the maximum of the full 112-reference distance row,
    so the same query uses one numerical tolerance in global and local scoring.
    """
    d, g = np.asarray(distances, dtype=float), np.asarray(gaps, dtype=float)
    if d.ndim != 1 or d.shape != g.shape or len(d) < 2:
        raise ValueError('two equally shaped vectors with at least two references required')
    if not np.isfinite(d).all() or not np.isfinite(g).all() or min(d.min(), g.min()) < 0:
        raise ValueError('distances and gaps must be finite and nonnegative')
    s = float(d.max()) if scale is None else float(scale)
    if not math.isfinite(s) or s < float(d.max()):
        raise ValueError('scale must cover the full query distance row')
    i, j = np.triu_indices(len(d), 1)
    dg, dd = g[i] - g[j], d[i] - d[j]
    performance_ties = np.abs(dg) <= 1e-12
    valid = ~performance_ties
    distance_ties = (np.abs(dd) <= 1e-10 * s) & valid
    ordered = valid & ~distance_ties
    concordant = int(np.sum(ordered & ((dg > 0) == (dd > 0))))
    discordant = int(np.sum(ordered)) - concordant
    ntie, nvalid = int(distance_ties.sum()), int(valid.sum())
    return {'score': (concordant + .5 * ntie) / nvalid if nvalid else None,
            'NA_reason': None if nvalid else 'all performance gaps numerically tied; no comparable pair',
            'concordant': concordant, 'discordant': discordant, 'distance_ties': ntie,
            'performance_ties': int(performance_ties.sum()), 'valid_denominator': nvalid,
            'total_pairs': len(i), 'distance_scale': s, 'distance_tie_tolerance': 1e-10 * s}


def select_neighbors(values, ids, k):
    if len(values) != len(ids) or len(set(ids)) != len(ids) or not 0 < k <= len(ids):
        raise ValueError('unique reference identity and valid k required')
    return sorted(range(len(ids)), key=lambda j: (float(values[j]), ids[j]))[:k]


def attainable_gap(de, dstar):
    raw = float(de - dstar)
    if raw < -1e-12:
        raise ValueError(f'negative excess gap: {raw}; inspect identity, labels, sorting')
    return max(0., raw), raw


def evaluate_query(distances, gaps, ids, embedding_indices):
    d, g = np.asarray(distances, dtype=float), np.asarray(gaps, dtype=float)
    # embedding_indices must already have been chosen and saved before label access.
    if embedding_indices != select_neighbors(d, ids, 5):
        raise ValueError('saved pre-label neighbors differ')
    oracle = select_neighbors(g, ids, 5)
    local = select_neighbors(g, ids, 10)
    global_score = order_score(d, g)
    local_score = order_score(d[local], g[local], scale=float(d.max()))
    de, dstar, drandom = float(g[embedding_indices].mean()), float(g[oracle].mean()), float(g.mean())
    excess, raw = attainable_gap(de, dstar)
    return {'global_order': global_score['score'], 'local10_order': local_score['score'],
            **{f'global_{k}': v for k, v in global_score.items() if k != 'score'},
            **{f'local10_{k}': v for k, v in local_score.items() if k != 'score'},
            'De_pp': 100 * de, 'Dstar_pp': 100 * dstar, 'E_pp': 100 * excess,
            'E_raw_accuracy': raw, 'Drandom_pp': 100 * drandom, 'De_minus_random_pp': 100 * (de - drandom),
            'embedding_neighbor_ids': [ids[j] for j in embedding_indices],
            'embedding_neighbor_distances': d[embedding_indices].tolist(),
            'embedding_neighbor_gaps_pp': (100 * g[embedding_indices]).tolist(),
            'oracle_neighbor_ids': [ids[j] for j in oracle], 'oracle_neighbor_gaps_pp': (100 * g[oracle]).tolist(),
            'local10_reference_ids': [ids[j] for j in local],
            'local10_gap_min_pp': float(100 * g[local].min()), 'local10_gap_max_pp': float(100 * g[local].max()),
            'exceeds_tolerance_0.006_fraction': float(np.mean(g[embedding_indices] > .006)),
            'exceeds_tolerance_0.012_fraction': float(np.mean(g[embedding_indices] > .012))}


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite summary input')
    if not len(a):
        return {'count': 0, **{k: None for k in ('mean', 'median', 'min', 'p25', 'p75', 'max', 'std_population')}}
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'min': float(a.min()), 'p25': float(np.percentile(a, 25)), 'p75': float(np.percentile(a, 75)),
            'max': float(a.max()), 'std_population': float(a.std())}


def weighted_values(rows, key, weighting):
    valid = [r for r in rows if r[key] is not None]
    if weighting == 'query_equal':
        return [r[key] for r in valid]
    if weighting != 'architecture_equal':
        raise ValueError('unknown weighting')
    by = defaultdict(list)
    for r in valid:
        by[r['architecture_fingerprint']].append(r[key])
    return [float(np.mean(v)) for _, v in sorted(by.items())]


def summarize(rows, state_id, subset='all29'):
    out = []
    for weighting in ('query_equal', 'architecture_equal'):
        for key in METRICS:
            valid = [r for r in rows if r[key] is not None]
            out.append({'state_id': state_id, 'subset': subset, 'weighting': weighting, 'metric': key,
                        'query_count': len(rows), 'valid_query_count': len(valid), 'excluded_query_count': len(rows)-len(valid),
                        'architecture_count': len({r['architecture_fingerprint'] for r in rows}),
                        'valid_architecture_count': len({r['architecture_fingerprint'] for r in valid}),
                        'unit': '0_to_1' if key.endswith('order') else 'percentage_points',
                        **distribution(weighted_values(rows, key, weighting))})
    return out


def pair_common(a, b, key, weighting):
    """Both sides use the intersection of valid query IDs, including for architecture means."""
    a, b = {r['candidate_id']: r for r in a}, {r['candidate_id']: r for r in b}
    if set(a) != set(b):
        raise ValueError('paired query identity differs')
    diffs, excluded = [], []
    for cid in sorted(a):
        if a[cid]['architecture_fingerprint'] != b[cid]['architecture_fingerprint']:
            raise ValueError('paired architecture differs')
        if a[cid][key] is None or b[cid][key] is None:
            excluded.append(cid)
        else:
            diffs.append({'candidate_id': cid, 'architecture_fingerprint': a[cid]['architecture_fingerprint'],
                          key: b[cid][key] - a[cid][key]})
    values = weighted_values(diffs, key, weighting)
    return {'delta': float(np.mean(values)) if values else None, 'valid_query_count': len(diffs),
            'excluded_query_count': len(excluded), 'excluded_query_ids': excluded,
            'valid_architecture_count': len({r['architecture_fingerprint'] for r in diffs})}


def paired_tables(by_state):
    comparisons = [(f'{m}: M300-M100', (m, 100), (m, 300)) for m in METHODS]
    comparisons += [(f'M{step}: N2-regression', ('regression', step), ('N2', step)) for step in (100, 300)]
    out, per_query = [], []
    for name, a, b in comparisons:
        for seed in (0, 1, 2):
            sa, sb = f'{a[0]}-s{seed}-M{a[1]}', f'{b[0]}-s{seed}-M{b[1]}'
            if sa not in by_state or sb not in by_state:
                continue
            br = {r['candidate_id']: r for r in by_state[sb]}
            for ar in by_state[sa]:
                r = br[ar['candidate_id']]
                per_query.append({'comparison': name, 'seed': seed, 'candidate_id': r['candidate_id'],
                                  'architecture_fingerprint': r['architecture_fingerprint'],
                                  **{k: r[k]-ar[k] if r[k] is not None and ar[k] is not None else None for k in METRICS}})
        for weighting in ('query_equal', 'architecture_equal'):
            for key in METRICS:
                results = {}
                for seed in (0, 1, 2):
                    sa, sb = f'{a[0]}-s{seed}-M{a[1]}', f'{b[0]}-s{seed}-M{b[1]}'
                    results[str(seed)] = pair_common(by_state[sa], by_state[sb], key, weighting) if sa in by_state and sb in by_state else {'delta': None, 'reason': 'state missing'}
                values = [v['delta'] for v in results.values() if v['delta'] is not None]
                out.append({'comparison': name, 'weighting': weighting, 'metric': key,
                            'improvement_direction': 'positive' if key.endswith('order') else ('invariant' if key in ('Dstar_pp','Drandom_pp') else 'negative'),
                            'seed_results': results, **{f'seed{s}': results[str(s)]['delta'] for s in (0,1,2)},
                            **distribution(values)})
    return out, per_query


def restore_mapping(method, seed, step, protocol, candidates, base):
    folder = OLD / 'tasks' / f'{method}-s{seed}-mapping'
    path = folder / f'M{step}.pt'
    completion = read(folder / 'completed.json')
    if file_hash(path) != completion['artifacts'][path.name]:
        raise ValueError('checkpoint hash differs from saved completion')
    state = torch.load(path, map_location='cpu', weights_only=False)
    ids, qids = protocol['inner']['train_ids'], protocol['inner']['query_ids']
    if state['train_ids'] != ids or state['query_ids'] != qids:
        raise ValueError('checkpoint candidate order differs')
    if state['setting'] != protocol['settings'][method] or state['model_seed'] != seed:
        raise ValueError('checkpoint method or seed differs')
    # Only inspect the stored step counters. Never construct/load an optimizer.
    if {float(v['step']) for v in state['optimizer']['state'].values()} != {float(step)}:
        raise ValueError('checkpoint optimizer-step identity differs')
    if state['common_config']['lambda_e'] != .001:
        raise ValueError('size constraint differs from fixed protocol')
    train, query = [[candidates[c] for c in which] for which in (ids, qids)]
    if digest({'train': train, 'query': query}) != state['candidate_identity_hash']:
        raise ValueError('candidate architecture/HP identity differs')
    scaler = state['standardization']
    if scaler['fit_ids'] != ids or scaler['hp_feature_order'] != ['log10_lr','dropout','log2_hidden_dim','log10_l2']:
        raise ValueError('saved preprocessing identity differs')
    def encoded(rows):
        z = torch.tensor([base[c['architecture_fingerprint']] for c in rows], dtype=torch.float32)
        hp = torch.tensor([hp_feature_vector(c) for c in rows], dtype=torch.float32)
        zm, zs, hm, hs = [torch.tensor(scaler[k], dtype=torch.float32) for k in ('z_mean','z_scale','hp_mean','hp_scale')]
        return (z-zm)/zs, (hp-hm)/hs
    z, h = encoded(train)
    qz, qh = encoded(query)
    model = JointMetricModel(z_dim=z.shape[1], seed=seed)
    if tensor_hash(model.state_dict()) != state['initialization_hash']:
        raise ValueError('model architecture/initialization differs')
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False)
    model.eval()
    before = tensor_hash(model.state_dict())
    e, _ = model(z, h)
    qe, _ = model(qz, qh)
    if not torch.equal(e, state['train_embeddings']) or e.shape != (112,8) or qe.shape != (29,8):
        raise ValueError('saved reference embeddings were not exactly restored')
    d = torch.cdist(qe.double(), e.double()).numpy()
    if not np.isfinite(d).all() or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('invalid frozen inference')
    if tensor_hash(model.state_dict()) != before:
        raise ValueError('mapping parameters changed during inference')
    prediction_path = OLD / 'tasks' / f'{method}-s{seed}-M{step}-gp' / 'G20_predictions_before_reveal.json'
    gp_completion = read(prediction_path.parent / 'completed.json')
    if file_hash(prediction_path) != gp_completion['artifacts'][prediction_path.name]:
        raise ValueError('old pre-reveal prediction file changed')
    old = read(prediction_path)
    if old['train_ids'] != ids or old['query_ids'] != qids or old['standardization'] != scaler or old['candidate_identity_hash'] != state['candidate_identity_hash']:
        raise ValueError('old neighbor record identity differs')
    old_rows = {r['candidate_id']: r for r in old['predictions']}
    neighbors, maxdiff = [], 0.
    for i, cid in enumerate(qids):
        near = select_neighbors(d[i], ids, 5)
        if [ids[j] for j in near] != old_rows[cid]['neighbor_ids']:
            raise ValueError('old 5 neighbor IDs not reproduced')
        diff = float(np.max(np.abs(d[i,near] - np.asarray(old_rows[cid]['neighbor_distances']))))
        if diff != 0.:
            raise ValueError(f'old neighbor distances not exactly reproduced: {diff}')
        maxdiff = max(maxdiff, diff)
        neighbors.append(near)
    return {'train_ids': ids, 'query_ids': qids, 'train_embeddings': e.tolist(), 'query_embeddings': qe.tolist(),
            'euclidean_distances': d.tolist(), 'embedding_neighbor_indices': neighbors,
            'embedding_neighbor_ids': [[ids[j] for j in row] for row in neighbors],
            'standardization': scaler, 'candidate_identity_hash': state['candidate_identity_hash'],
            'checkpoint': str(path), 'checkpoint_sha256': file_hash(path), 'parameter_hash': before,
            'old_prediction': str(prediction_path), 'old_prediction_sha256': file_hash(prediction_path),
            'checks': {'train_embeddings_bitwise_equal': True, 'model_parameters_unchanged': True,
                       'old_top5_ids_equal': True, 'old_top5_distance_max_difference': maxdiff,
                       'dimension': 8, 'device': 'cpu', 'eval_mode': True, 'requires_grad': False}}


def matrix_csv(path, qids, ids, matrix):
    csv_write(path, [{'query_candidate_id': q, **{c: float(v) for c,v in zip(ids,row)}} for q,row in zip(qids,matrix)])


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise FileExistsError('result already initialized; use a fresh --output to preserve previous evidence')
    started = time.monotonic()
    protocol = read(OLD / 'protocol.json')
    sup_identity = read(SUP / 'config_and_identity.json')
    prior = read(ROOT / protocol['config']['old_output'] / 'protocol.json')
    if file_hash(OLD/'protocol.json') != sup_identity['old_protocol_sha256']:
        raise ValueError('diagnosis protocol changed since readonly supplement')
    if file_hash(ROOT/protocol['config']['old_output']/'protocol.json') != sup_identity['old_stage2_protocol_sha256']:
        raise ValueError('original Stage2 protocol changed')
    for p, h in {**prior['source_identity'], **protocol['sources'], **sup_identity['sources']}.items():
        if file_hash(ROOT/p) != h:
            raise ValueError(f'original source changed: {p}')
    for p, h in sup_identity['unlabeled_inputs'].items():
        if file_hash(p) != h:
            raise ValueError(f'unlabeled input changed: {p}')
    ids, qids = protocol['inner']['train_ids'], protocol['inner']['query_ids']
    allowed = set(ids+qids)
    if len(ids) != 112 or len(qids) != 29 or len(allowed) != 141 or allowed & set(protocol['outer_holdout_ids_excluded_from_new_comparisons']):
        raise ValueError('fixed 112/29 split boundary differs')
    if sup_identity['train_ids'] != ids or sup_identity['query_ids'] != qids:
        raise ValueError('whitelisted source identity differs')
    settings = protocol['settings']
    if (settings['N2']['epsilon'], settings['N2']['lambda_f'], settings['regression']['lambda_f'], sup_identity['lambda_e']) != (.02,.01,0.,.001):
        raise ValueError('fixed settings differ')
    prep = ROOT / prior['config']['prepared']
    candidate_rows, base = read(prep/'candidates.json'), read(prep/'base_encoding.json')
    if digest(candidate_rows) != prior['prepared_manifest']['pool_hash'] or digest(base) != prior['prepared_manifest']['base_encoding_hash']:
        raise ValueError('saved prepared input identities differ')
    candidates = {r['candidate_id']: r for r in candidate_rows if r['candidate_id'] in allowed}
    if set(candidates) != allowed:
        raise ValueError('candidate identity missing')
    arch_t, arch_q = [{candidates[c]['architecture_fingerprint'] for c in which} for which in (ids,qids)]
    if arch_t & arch_q or len(arch_q) != 16:
        raise ValueError('architecture leakage or unexpected query architecture count')
    states = [(m,s,n) for m in METHODS for s in (0,1,2) for n in (100,300)]
    sources = ['analyse/fuzzy_metric_relative_geometry.py', 'tests/test_fuzzy_metric_relative_geometry.py',
               'analyse/fuzzy_metric_diagnosis_supplement.py', 'analyse/fuzzy_metric_replay.py',
               'analyse/joint_configuration_metric_gate.py', 'surrogate/fuzzy_metric.py']
    input_paths = [OLD/'protocol.json', SUP/'config_and_identity.json', prep/'candidates.json', prep/'base_encoding.json',
                   SUP/'training_labels.json', SUP/'query_validation_means.json', SUP/'metrics_24_combinations.json']
    input_paths += [OLD/'tasks'/f'{m}-s{s}-mapping'/f'M{n}.pt' for m,s,n in states]
    frozen = {'frozen_at_utc': utc(), 'purpose': 'additional development diagnosis; not independent blind validation or replacement of original Stage2 gate',
              'methods': list(METHODS), 'seeds': [0,1,2], 'mapping_steps': [100,300], 'state_count':12,
              'settings': settings, 'lambda_e':.001, 'distance': 'raw 8D Euclidean; float32 embeddings promoted to float64 for torch.cdist; no normalization',
              'performance_tie_absolute_accuracy':1e-12, 'distance_tie_relative':1e-10,
              'distance_scale': 'maximum distance to all 112 references for that query, shared by global and local10 scoring',
              'score': 'performance ties excluded; same order 1; reversed 0; distance tie 0.5; zero valid pairs NA',
              'selection_ties': 'exact raw value then candidate_id; numerical scoring ties never broken by ID',
              'k_embedding':5, 'k_oracle':5, 'k_local':10, 'local_scope':'oracle closest10; references not claimed retrieved by model',
              'main_aggregation':'equal mean over29 queries per state; distributions also saved',
              'supplementary_aggregation':'within-architecture query mean, then equal mean over16 query architectures',
              'pairing':'same model seed; common valid queries per metric on both sides; exclusions disclosed',
              'high_performance_rule':'top ceil(29*0.1)=3 query validation means descending, exact ties by candidate_id; references all112',
              'gap_unit':'percentage_points; underlying means accuracy; E_raw_accuracy retained; below -1e-12 raises',
              'random_reference':'exact mean absolute gap over112 = expected mean gap of uniform without-replacement random5; not prediction MAE',
              'training_budget': {'GNN':0,'VAE':0,'mapping':0,'GP':0}, 'optimization_step_budget':0,
              'threads':1, 'device':'cpu', 'train_ids':ids, 'query_ids':qids,
              'train_architecture_count':len(arch_t), 'query_architecture_count':len(arch_q), 'architecture_overlap':0,
              'label_sources':'already whitelisted prior supplement files; no global/A/B/outer/test performance file decoded',
              'B_labels_read':0,'outer38_labels_read':0,'test_labels_read':0,
              'source_hashes':{p:file_hash(ROOT/p) for p in sources}, 'input_hashes':{str(p):file_hash(p) for p in input_paths}}
    write(output/'protocol.json',frozen)
    write(output/'candidate_identity_before_labels.json', {'train':[candidates[c] for c in ids], 'query':[candidates[c] for c in qids]})
    for p in sources:
        dest = output/'source_snapshot'/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/p,dest)
    # This phase has no label argument and opens no query-label file.
    restored, failures = {}, []
    with no_optimization() as guard:
        for method, seed, step in states:
            sid = f'{method}-s{seed}-M{step}'
            try:
                snap = restore_mapping(method,seed,step,protocol,candidates,base)
                snap.update({'state_id':sid,'method':method,'seed':seed,'mapping_steps':step,'saved_before_label_access_utc':utc()})
                folder = output/'states'/sid
                write(folder/'geometry_before_labels.json',snap)
                matrix_csv(folder/'distance_matrix.csv',qids,ids,snap['euclidean_distances'])
                restored[sid] = snap
            except Exception:
                failures.append({'state_id':sid,'time':utc(),'traceback':traceback.format_exc()})
                write(output/'restore_failures.json',failures)
        write(output/'distance_freeze.json', {'saved_at_utc':utc(), 'state_count':len(restored), 'failures':failures,
              'files':{str(p.relative_to(output)):file_hash(p) for p in (output/'states').rglob('*') if p.is_file()}})
        # Only after all recoverable geometry/retrieval files are durably written.
        reveal_at = utc()
        train_records = read(SUP/'training_labels.json')
        query_means = read(SUP/'query_validation_means.json')
        if set(train_records) != set(ids) or set(query_means) != set(qids):
            raise ValueError('whitelisted mean file contains unexpected/missing IDs')
        train_means = {c:float(train_records[c]['validation_mean']) for c in ids}
        query_means = {c:float(query_means[c]) for c in qids}
        del train_records
        means = {'train':train_means,'query':query_means,'first_label_access_utc':reveal_at,
                 'meaning':'previous observed validation means over GNN seeds0/1/2; no new GNN training; development-only'}
        write(output/'allowed_validation_means.json',means)
        csv_write(output/'allowed_validation_means.csv',[{'role':role,'candidate_id':c,'architecture_fingerprint':candidates[c]['architecture_fingerprint'],'validation_mean':v} for role,dic in [('train',train_means),('query',query_means)] for c,v in dic.items()])
        yt,yq = np.array([train_means[c] for c in ids]),np.array([query_means[c] for c in qids])
        if not np.isfinite(yt).all() or not np.isfinite(yq).all():
            raise ValueError('nonfinite allowed means')
        gaps = np.abs(yq[:,None]-yt[None,:])
        top = sorted(qids,key=lambda c:(-query_means[c],c))[:math.ceil(len(qids)*.1)]
        write(output/'high_performance_queries.json',{'rule':frozen['high_performance_rule'],'reference_count':112,
              'queries':[{'candidate_id':c,'architecture_fingerprint':candidates[c]['architecture_fingerprint'],'validation_mean':query_means[c]} for c in top]})
        old_metrics = read(SUP/'metrics_24_combinations.json')
        reference = []
        for r in old_metrics:
            reference.append({'method':r['method'],'seed':r['seed'],'mapping_steps':r['mapping_steps'],'gp_steps':r['gp_steps'],
                'old_query_median_neighbor_gap_pp':100*r['main_neighborhood_metric'],
                **{f'{k}_pp':100*r[k] for k in ('head_mae','knn_mae','gp_mae','constant_mae')},
                **{f'exceeds_tolerance_{tol}_fraction':r[f'wrong_{tol}'] for tol in ('0.006','0.012')},
                'use':'existing error reference only; GP20/100 do not duplicate geometry states'})
        write(output/'old_prediction_reference.json',reference)
        csv_write(output/'old_prediction_reference.csv',reference)
        by_state, summaries, checks = {}, [], []
        for sid,snap in restored.items():
            d = np.array(snap['euclidean_distances'])
            rows = []
            for i,cid in enumerate(qids):
                rows.append({'state_id':sid,'method':snap['method'],'seed':snap['seed'],'mapping_steps':snap['mapping_steps'],
                             'candidate_id':cid,'architecture_fingerprint':candidates[cid]['architecture_fingerprint'],
                             'query_validation_mean':query_means[cid],
                             **evaluate_query(d[i],gaps[i],ids,snap['embedding_neighbor_indices'][i])})
            by_state[sid]=rows
            folder=output/'states'/sid
            write(folder/'performance_gap_matrix.json',{'query_ids':qids,'train_ids':ids,'unit':'percentage_points','performance_gaps_pp':(100*gaps).tolist()})
            matrix_csv(folder/'performance_gap_matrix_pp.csv',qids,ids,100*gaps)
            write(folder/'per_query.json',rows)
            csv_write(folder/'per_query.csv',rows)
            summaries += summarize(rows,sid) + summarize([r for r in rows if r['candidate_id'] in top],sid,'top3_descriptive')
            previous = next(r for r in reference if (r['method'],r['seed'],r['mapping_steps'],r['gp_steps'])==(snap['method'],snap['seed'],snap['mapping_steps'],20))
            median_diff=abs(float(np.median([r['De_pp'] for r in rows]))-previous['old_query_median_neighbor_gap_pp'])
            tol_diffs={t:abs(float(np.mean([r[f'exceeds_tolerance_{t}_fraction'] for r in rows]))-previous[f'exceeds_tolerance_{t}_fraction']) for t in ('0.006','0.012')}
            if median_diff>1e-12 or max(tol_diffs.values())>1e-12:
                raise ValueError(f'old neighbor metric did not reproduce for {sid}')
            checks.append({'state_id':sid,**snap['checks'],'old_query_median_metric_difference_pp':median_diff,
                           'old_tolerance_stat_differences':tol_diffs,'global_pair_total':sum(r['global_total_pairs'] for r in rows),
                           'global_valid_denominator':sum(r['global_valid_denominator'] for r in rows),
                           'local_pair_total':sum(r['local10_total_pairs'] for r in rows), 'E_min_pp':min(r['E_pp'] for r in rows),
                           'E_tiny_negative_clamp_count':sum(r['E_raw_accuracy']<0 for r in rows)})
        rows=[r for subset in by_state.values() for r in subset]
        write(output/'per_query_348.json',rows)
        csv_write(output/'per_query_348.csv',rows)
        write(output/'summaries.json',summaries)
        csv_write(output/'summaries.csv',summaries)
        paired,paired_query=paired_tables(by_state)
        write(output/'paired_differences.json',paired)
        csv_write(output/'paired_differences.csv',paired)
        write(output/'paired_per_query.json',paired_query)
        csv_write(output/'paired_per_query.csv',paired_query)
        for cid in qids:
            for key in ('Dstar_pp','Drandom_pp'):
                if len({r[key] for r in rows if r['candidate_id']==cid}) != 1:
                    raise ValueError('oracle/random reference changes across states')
        freeze=read(output/'distance_freeze.json')
        if not all(file_hash(output/p)==h for p,h in freeze['files'].items()):
            raise ValueError('pre-label geometry changed during evaluation')
        write(output/'checks.json',{'completed_states':len(restored),'expected_states':12,'query_state_rows':len(rows),'expected_rows':348,
              'query_count':29,'query_architecture_count':16,'train_count':112,'architecture_overlap':0,
              'same_reference_pool_all_states':True,'same_Dstar_Drandom_per_query':True,'pre_label_geometry_files_unchanged':True,
              'distance_freeze_at':freeze['saved_at_utc'],'first_label_access_at':reveal_at,
              'global_pair_comparisons':sum(r['global_total_pairs'] for r in rows),'local_pair_comparisons':sum(r['local10_total_pairs'] for r in rows),
              'global_NA_count':sum(r['global_order'] is None for r in rows),'local_NA_count':sum(r['local10_order'] is None for r in rows),
              'global_performance_ties':sum(r['global_performance_ties'] for r in rows),'global_distance_ties':sum(r['global_distance_ties'] for r in rows),
              'global_valid_denominator':sum(r['global_valid_denominator'] for r in rows),
              'restore_failures':failures,'per_state':checks,'zero_optimization_guard':guard.copy(),
              'new_training_counts':{'GNN':0,'VAE':0,'mapping':0,'GP':0},'new_optimization_steps':{'mapping':0,'GP':0},
              'GP_objects_constructed':0,'VAE_objects_constructed':0,'label_counts':{'train_means':112,'query_means':29,'B':0,'outer38':0,'test':0},
              'elapsed_seconds':time.monotonic()-started,'device':'cpu','torch_threads':torch.get_num_threads()})
    print(json.dumps({'output':str(output),'states':len(restored),'rows':len(rows),'zero_optimization':guard,'failures':len(failures)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readonly',action='store_true',required=True)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(args.output)


if __name__=='__main__':
    main()
