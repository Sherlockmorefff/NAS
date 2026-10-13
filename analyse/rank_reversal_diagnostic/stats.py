"""Preregistered paired analysis; all results in percentage points."""
import numpy as np
from .common import *

def summary(x):
    x = np.asarray(x, dtype=float)
    return {'n': len(x), 'mean': float(x.mean()) if len(x) else None,
            'std_ddof1': float(x.std(ddof=1)) if len(x) > 1 else None}

def paired(d, indices=None):
    x = np.asarray(d, dtype=float)
    if len(x) == 0:
        return {'n': 0, 'differences_pp': [], 'CI95_pp': None, 'mean_pp': None}
    if indices is None:
        indices = np.random.default_rng(20261007).integers(0, len(x), (10000, len(x)))
    return {'n': len(x), 'differences_pp': x.tolist(), 'mean_pp': float(x.mean()),
            'median_pp': float(np.median(x)), 'std_pp': float(x.std(ddof=1)) if len(x) > 1 else None,
            'CI95_pp': np.quantile(x[indices].mean(1), [.025, .975]).tolist(),
            'wins': int((x > 0).sum()), 'ties': int((x == 0).sum()), 'losses': int((x < 0).sum())}

def ranked(values, candidates):
    ids = {c['alias']: c['config_id'] for c in candidates}
    ordered = sorted(values, key=lambda a: (-values[a], ids[a]))
    return {'order': ordered, 'ranks': {a: ordered.index(a) + 1 for a in ordered},
            'tied_maxima': [a for a in ordered if values[a] == max(values.values())]}

def analyze(root):
    candidates = read(root / 'candidate_manifest.json')['candidates']
    tasks = read(root / 'task_manifest.json')['tasks']
    rows = []
    for task in tasks:
        folder = root / 'training' / task['task_id']
        row = dict(task)
        if (folder / 'result.json').exists():
            r = read(folder / 'result.json')
            verify(folder, r['files'])
            row.update({k: r[k] for k in ['state', 'selection', 'early_stop', 'best_epoch', 'stopped_epoch',
                                          'stop_reason', 'worker_seconds', 'gpu_event_seconds', 'peak_gpu_bytes', 'identity']})
            row['selection_pp'] = 100 * r['selection']
            row['checkpoint_path'] = str(folder / 'best_model.pt')
            row['checkpoint_sha256'] = r['files']['best_model.pt']
            row['random_initializations'] = sum(x['random_initialization'] for x in jsonlines(folder / 'attempts.jsonl'))
            row['strict_resumes'] = sum(x['strict_resume'] for x in jsonlines(folder / 'attempts.jsonl'))
            tp = root / 'test_evaluations' / (task['task_id'] + '.json')
            if tp.exists():
                assert task['phase'] == 'D1'
                t = read(tp)
                assert t['checkpoint_sha256'] == row['checkpoint_sha256']
                row['old_test_pp'] = 100 * t['test_accuracy']
                row['test_forwards'] = 1
            else:
                row['test_forwards'] = 0
        elif (folder / 'failure.json').exists():
            row.update({'state': 'FAILED', 'failure': read(folder / 'failure.json'), 'test_forwards': 0})
        else:
            row.update({'state': 'NOT_EXECUTED', 'test_forwards': 0})
        rows.append(row)
    d1 = [r for r in rows if r['phase'] == 'D1']
    d2 = [r for r in rows if r['phase'] == 'D2']
    csvwrite(root / 'D1_RESULTS.csv', d1)
    csvwrite(root / 'D2_RESULTS.csv', d2)
    complete = [r for r in rows if r['state'] == 'COMPLETE']
    index = {(r['split'], r['seed'], r['alias']): r for r in complete}
    seeds = read(root / 'seed_manifest.json')
    aliases = [c['alias'] for c in candidates]
    d1stats = {}
    for a in aliases:
        rr = [r for r in d1 if r['alias'] == a and r['state'] == 'COMPLETE']
        d1stats[a] = {'selection_pp': summary([r['selection_pp'] for r in rr]),
                      'old_test_pp': summary([r['old_test_pp'] for r in rr if 'old_test_pp' in r])}
    pairs = {}
    pairrows = []
    for b in aliases[1:]:
        common = [s for s in seeds['D1'] if all(('original', s, a) in index and
                  'old_test_pp' in index[('original', s, a)] for a in ['C0', b])]
        sel = [index['original', s, 'C0']['selection_pp'] - index['original', s, b]['selection_pp'] for s in common]
        test = [index['original', s, 'C0']['old_test_pp'] - index['original', s, b]['old_test_pp'] for s in common]
        pairs[b] = {'seeds': common, 'selection': paired(sel), 'old_test': paired(test),
                    'reversal_seeds': [s for s, x, y in zip(common, sel, test) if x * y < 0],
                    'tie_in_either_seeds': [s for s, x, y in zip(common, sel, test) if x == 0 or y == 0],
                    'mean_reversal': bool(len(sel) and np.mean(sel) * np.mean(test) < 0)}
        if sel:
            draws = np.random.default_rng(20261007).integers(0, len(sel), (10000, 3))
            p3 = np.asarray(sel)[draws].mean(1)
            pairs[b]['approximate_three_seed_mean_selection'] = {'CI95_pp': np.quantile(p3, [.025, .975]).tolist(),
                'bootstrap_std_pp': float(p3.std(ddof=1)), 'basis': 'new-seed empirical paired-difference distribution, descriptive approximation'}
        pairrows.extend({'comparison': 'C0-' + b, 'seed': s, 'selection_difference_pp': x, 'old_test_difference_pp': y,
                         'strict_reversal': x * y < 0} for s, x, y in zip(common, sel, test))
    rankings = []
    for s in seeds['D1']:
        for metric in ['selection_pp', 'old_test_pp']:
            vals = {a: index['original', s, a][metric] for a in aliases
                    if ('original', s, a) in index and metric in index['original', s, a]}
            if vals:
                rankings.append({'seed': s, 'metric': metric, **ranked(vals, candidates)})
    splitstats = {}
    splitpairs = {}
    split_pairrows = []
    for sp in [f'split{i}' for i in range(1, 6)]:
        values = {}
        for a in aliases:
            rr = [index[sp, s, a]['selection_pp'] for s in seeds['D2_per_split'] if (sp, s, a) in index]
            values[a] = summary(rr)
        splitstats[sp] = {'configurations': values, 'ranking': ranked({a: v['mean'] for a, v in values.items() if v['n']}, candidates)}
    for b in aliases[1:]:
        per = []
        for sp in splitstats:
            common = [s for s in seeds['D2_per_split'] if all((sp, s, a) in index for a in ['C0', b])]
            diff = [index[sp, s, 'C0']['selection_pp'] - index[sp, s, b]['selection_pp'] for s in common]
            reference = [s for s in common if all(('original', s, a) in index for a in ['C0', b])]
            change = [(index[sp, s, 'C0']['selection_pp'] - index[sp, s, b]['selection_pp']) -
                      (index['original', s, 'C0']['selection_pp'] - index['original', s, b]['selection_pp']) for s in reference]
            v = {'split': sp, 'paired_seeds': common, 'paired_differences_pp': diff, **summary(diff),
                 'reference_seeds': reference, 'change_vs_original_same_seeds': summary(change)}
            per.append(v)
            split_pairrows.extend({'comparison': 'C0-' + b, 'split': sp, 'seed': s, 'difference_pp': d} for s, d in zip(common, diff))
        means = [v['mean'] for v in per if v['n']]
        splitpairs[b] = {'per_split': per, 'across_split': paired(means),
            'split_mean_variance_ddof1': float(np.var(means, ddof=1)) if len(means) > 1 else None,
            'mean_within_split_paired_seed_variance': float(np.mean([v['std_ddof1'] ** 2 for v in per if v['std_ddof1'] is not None])),
            'exploratory_only': True}
    csvwrite(root / 'D1_PAIRED.csv', pairrows)
    csvwrite(root / 'D2_PAIRED.csv', split_pairrows)
    meansel = {a: v['selection_pp']['mean'] for a, v in d1stats.items() if v['selection_pp']['n']}
    meantest = {a: v['old_test_pp']['mean'] for a, v in d1stats.items() if v['old_test_pp']['n']}
    champions = [v['ranking']['order'][0] for v in splitstats.values() if v['ranking']['order']]
    metrics = {'D1_configurations': d1stats, 'D1_comparisons': pairs, 'D1_seed_rankings': rankings,
        'D1_mean_selection_ranking': ranked(meansel, candidates), 'D1_mean_old_test_ranking': ranked(meantest, candidates),
        'D2_splits': splitstats, 'D2_comparisons': splitpairs,
        'D2_champion_counts': {a: champions.count(a) for a in aliases},
        'D2_champion_change_from_original_D1_shared3': None,
        'complete_matrix': len(complete) == 125, 'inference_limits': '10 paired seeds; 5 overlapping masks on one graph; old test previously consulted; no formal multiplicity claims'}
    shared3 = {a: np.mean([index['original', s, a]['selection_pp'] for s in seeds['D2_per_split']
                           if ('original', s, a) in index]) for a in aliases
               if any(('original', s, a) in index for s in seeds['D2_per_split'])}
    ref_rank = ranked(shared3, candidates)
    metrics['D1_shared3_selection_reference'] = {'means_pp': shared3, 'ranking': ref_rank}
    metrics['D2_champion_change_from_original_D1_shared3'] = sum(c != ref_rank['order'][0] for c in champions) if ref_rank['order'] else None
    write(root / 'analysis_metrics.json', metrics)
    write(root / 'analysis_rows.json', rows)
    return metrics
