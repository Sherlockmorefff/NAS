"""Freeze the specified experiment using current server files, no new held-out access."""
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from .common import *

def inventory_gpus():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in csv.reader(raw.splitlines()):
        i, uuid, name, total, used, utilization = [x.strip() for x in line]
        rows.append({'index': int(i), 'uuid': uuid, 'name': name, 'total_MiB': int(total),
            'used_MiB': int(used), 'utilization_percent': int(utilization)})
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    occupied = {r[1].strip() for r in csv.reader(active.splitlines()) if len(r) >= 2}
    eligible = [r for r in rows if r['uuid'] not in occupied and r['total_MiB'] - r['used_MiB'] >= 2048]
    if len(eligible) < 2:
        raise RuntimeError('RESOURCE_BLOCKED: two unused eligible GPUs are required; no single-GPU fallback')
    return {'inventory_at': now(), 'all': rows, 'active_compute_processes': active,
        'workers': {str(i + 1): r for i, r in enumerate(sorted(eligible, key=lambda x: x['index'])[:2])},
        'binding': 'CUDA_VISIBLE_DEVICES=<physical UUID>; each child sees exactly cuda:0; one child per worker'}

def historical_seeds(repo):
    values, files = set(), {}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if 'seed' in k.lower():
                    if type(v) is int:
                        values.add(v)
                    elif isinstance(v, list) and all(type(z) is int for z in v):
                        values.update(v)
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for value in x:
                walk(value)
    for base in ['results', 'legacy_artifacts', 'artifacts', 'configs']:
        for p in (repo / base).rglob('*.json'):
            if p.name not in {'args.json', 'seed_manifest.json', 'training_task_manifest.json',
                'fresh_training_manifest.json', 'task_manifest.json', 'final_eval_config.json',
                'performance_protocol.json', 'replay_protocol.json', 'cora_confirmation_seeds_0_29.json',
                'cora_layernorm_confirmation_seeds.json'}:
                continue
            value = read(p)
            walk(value); files[str(p.relative_to(repo))] = sha(p)
    return {'values': sorted(values), 'files': files,
        'scope': 'existing seed/argument/task/protocol metadata only; no score-based seed selection; unrecorded work not certified'}

def make_development_splits(data, original, forbidden_identities):
    import numpy as np
    labels = data['development_y']
    def describe(train, val, seed):
        roles = {'train': sorted(train), 'selection': sorted(val), 'early_stop': sorted(val)}
        counts = {role: {str(c): int((labels[nodes] == c).sum()) for c in range(7)}
            for role, nodes in roles.items()}
        return {'indices': roles, 'identity': digest(roles), 'index_hashes': {r: digest(v) for r,v in roles.items()},
            'class_counts': counts, 'counts': {r: len(v) for r,v in roles.items()}, 'split_seed': seed}
    base = describe(original['train'], original['selection'], None)
    splits = {'original': base}; seen = {base['identity']} | set(forbidden_identities)
    seed = 2026100901; rejected = []
    for number in range(1, 3):
        while True:
            rng = np.random.default_rng(seed); train, val = [], []
            for c in range(7):
                members = [i for i in range(640) if int(labels[i]) == c]
                order = rng.permutation(members).tolist()
                train.extend(order[:20]); val.extend(order[20:])
            item = describe(train, val, seed)
            seed += 1
            if item['identity'] not in seen:
                break
            rejected.append(seed - 1)
        assert item['class_counts'] == base['class_counts']
        assert len(set(item['indices']['train']) | set(item['indices']['selection'])) == 640
        seen.add(item['identity']); splits['split' + str(number)] = item
    return {'splits': splits, 'pool': list(range(640)), 'pool_hash': digest(list(range(640))),
        'algorithm': 'PCG64, sorted classes and node indices; 20 train per class; remainder validation; same mask for stopping/scoring',
        'rejected_identity_only_duplicate_seeds': rejected, 'old_test_or_H_labels_used': False,
        'M_changes_training_node_identity': True, 'formal_M_allowed': True}

def preflight(root, repo):
    if (root / 'FREEZE.json').exists():
        verify_run(root, repo); return
    root.mkdir(parents=True, exist_ok=True)
    old, diag = repo / OLD, repo / DIAG
    old_identity = read(old / 'performance_identity.json')
    diagnostic = read(diag / 'freeze.json')
    verify(repo, old_identity['code']); verify(repo, diagnostic['code']); verify(diag, diagnostic['inputs'])
    p4 = read(old / 'P4_complete.json'); verify(old, p4['files'])
    write(root / 'CACHE_ACCEPTANCE.json', {'state': 'PASS', 'verified_files': len(p4['files']),
        'old_tasks': 528, 'old_configurations': 176, 'missing_or_mismatched_files': 0,
        'source': str(old), 'P4_complete_sha256': sha(old / 'P4_complete.json'),
        'reuse_role': 'query only; old seed0/1/2 mean/SE; same scientific recipe and numerical epoch',
        'new_wrapper_code_not_used_to_claim_new_seed_or_mask_cache_equivalence': True, 'time': now()})
    from deterministic_runtime import prepare_deterministic_environment, configure_torch_determinism
    prepare_deterministic_environment()
    import torch
    import numpy as np
    configure_torch_determinism(torch); torch.set_default_dtype(torch.float32); torch.set_num_threads(1)
    from analyse.rank_reversal_diagnostic.audit import make_model
    data = torch.load(diag / 'development_data.pt', map_location='cpu', weights_only=False)
    assert torch.all(data['development_y'][640:] == -100)
    assert data['x'].dtype == torch.float32 and data['x'].shape == (2708,1433)
    assert data['num_classes'] == 7
    shutil.copy2(diag / 'development_data.pt', root / 'development_data.pt')
    previous_splits = read(diag / 'split_manifest.json')['splits']
    original = previous_splits['original']['indices']
    forbidden = [digest({r: s['indices'][r] for r in ['train','selection','early_stop']})
        for s in previous_splits.values()]
    splits = make_development_splits(data, original, forbidden)
    write(root / 'SPLITS.json', splits)
    qualified = read(old / 'C_qualified_manifest.json')['configurations']
    configs = {}; canonical = set()
    for c in qualified:
        normalized = effective_config(c); ncid = digest(normalized)
        if ncid in canonical:
            raise ValueError('Effective duplicate in fixed candidate space; do not silently change pool')
        canonical.add(ncid)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = make_model(c, data['num_features'], data['num_classes'], torch.device('cpu'))
        assert model.ops == c['operations'] and model.edges == c['edges'] and len(model.layer_norms) == 0
        assert c['candidate_id'] == digest({'architecture_fingerprint':c['architecture_fingerprint'],'hp_fingerprint':c['hp_fingerprint']})
        configs[c['candidate_id']] = {'configuration': c, 'effective_configuration': normalized,
            'configuration_hash': digest(normalized), 'normalized_config_id': c['candidate_id'],
            'config_ID_basis':'existing complete canonical architecture + HP fingerprints; preserve original U tie order',
            'parameter_count': sum(p.numel() for p in model.parameters())}
    assert len(configs) == 176
    write(root / 'CONFIGURATIONS.json', {'by_id': configs, 'count': 176,
        'source': str(old / 'C_qualified_manifest.json'), 'source_sha256': sha(old / 'C_qualified_manifest.json'),
        'no_latent_redecode': True})
    public = read(old / 'public_C.json')
    assert set(public) == {'ids','x_raw','wl','architecture_ids'} and set(public['ids']) == set(configs)
    write(root / 'PUBLIC_C.json', public)
    history = historical_seeds(repo); write(root / 'HISTORICAL_SEEDS.json', history)
    unavailable = set(history['values']) | {0,1,2}
    searches = []; candidate = 2026100800
    while len(searches) < 20:
        if candidate not in unavailable:
            searches.append(candidate)
        candidate += 1
    seeds = {}; next_seed = 2026101000
    for group in range(20):
        selected = []
        while len(selected) < 16:
            if next_seed not in unavailable and next_seed not in searches:
                selected.append(next_seed)
            next_seed += 1
        seeds[str(group)] = {'search': searches[group], 'rerank': selected[:6], 'evaluation': selected[6:],
            'M_each_split': selected[:3]}
    write(root / 'SEEDS.json', {'groups': seeds, 'query': [0,1,2],
        'between_groups': 'disjoint rerank/evaluation seeds', 'within_group': 'common paired seeds across candidates/rules',
        'selection': 'ascending unused identities, no historical scores'})
    initial = {str(g): np.random.default_rng(np.random.SeedSequence([seeds[str(g)]['search'],0])).permutation(sorted(configs)).tolist()[:50]
        for g in range(20)}
    write(root / 'INITIALIZATIONS.json', initial)
    cache = read(old / 'private/C_validation_cache.json')
    frozen_cache = {}
    for cid, value in cache.items():
        counts = [round(v * 500) for v in value['validation_values']]
        assert len(counts) == 3 and all(n / 500 == v for n,v in zip(counts,value['validation_values']))
        assert value['seeds'] == [0,1,2]
        frozen_cache[cid] = {'candidate_id': cid, 'mean': value['mean'], 'se': value['se'],
            'query_correct_sum': sum(counts), 'query_denominator': 1500, 'query_seeds': [0,1,2]}
    write(root / 'private/QUERY_CACHE.json', frozen_cache)
    gpu = inventory_gpus(); write(root / 'GPUS.json', gpu)
    recipe = read(diag / 'training_recipe.json')
    recipe.update(seed='SEEDS.json', entrypoint=MODULE + '.training.train_task',
        numerical_epoch='unmodified structured_kernel_followup.validation_train.epoch_core',
        device_rule=gpu['binding'], test_during_training=False, role='validation only')
    write(root / 'TRAINING_RECIPE.json', recipe)
    old_protocol = read(old / 'performance_protocol.json')
    protocol = {'version': 'old-C-return-rule-1', 'frozen_at': now(), 'methods': ['W-I'], 'rules': list(RULES),
        'groups': 20, 'candidate_pool': {'configurations':176,'architectures':44,'source':OLD},
        'query_target': 'unchanged cached mean three best public validation scores; SE unchanged',
        'GP': old_protocol['gp'], 'acquisition': old_protocol['acquisition'],
        'U': {'initial':50,'ordinary':20,'terminal':1,'query_trains':213,'rerank':0,'final':10,'total':223},
        'S': {'initial':50,'ordinary':10,'terminal':1,'query_trains':183,'rerank':30,'final':10,'total':223},
        'M': {'shared_SM_search_and_top5':True,'query_trains':183,'rerank':30,'final':10,'total':223},
        'terminal': 'SM: posterior mean at n60; U: posterior mean at n70; no n61/n71 fit; no rerank feedback',
        'top5': 'own observed valid effective-unique configurations; exact query counts then normalized full config ID',
        'return': {'U':'exact mean original3','S':'exact mean new6 only','M':'mean3 per split then equal mean2'},
        'ties': 'integer correct counts / denominators; existing full canonical candidate_id ascending (original U tie order)',
        'M_permission': '640 developer labels allowed; train-node identities change; not strict fixed-public-training benchmark',
        'evaluation': 'immutable return, public masks, ten new paired seeds, mean best validation; no replacement or seed picking',
        'endpoint': 'original 500-node validation; same nodes for stopping and score; conditional development only',
        'old_test_access': False, 'H_access': False, 'next_stage': False,
        'logical_total':13380, 'new_tasks':{'query':0,'S':600,'M':600,'evaluation_min':200,'evaluation_max':600},
        'statistics': {'comparisons':['S-U','M-U','M-S'], 'bootstrap_n':10000,'seed':2026100899,
            'adjusted_confidence':1-.05/3,'adjusted_tail':.05/6,'descriptive_confidence':.95,
            'sampling':'crossed paired group x evaluation-replicate positions; identical draws for all rules; query and rerank tables/masks fixed',
            'seed_detail':'actual evaluation seed values are disjoint across groups; columns are paired replicate positions, not 200 independent searches',
            'selection':'default U; S-U lower>0 => S; M-U lower>0 and S fails => M; both pass: M only if M-S lower>0, else S',
            'incomplete':'no formal 20-group selection; report missing tasks and actual costs; no imputation'},
        'execution': {'workers':2,'tasks_per_GPU':1,'pilot':6,'pilot_choice':'parameter-count extremes and distinct architectures from legal queued tasks, before new scores',
            'timer':'conservative pilot ETA +900s; preserve first check; recheck every300s; persistent server controller',
            'retry':'complete-state resume only; no from-initialization automatic retry; failures retained'},
        'training_recipe_sha256':sha(root / 'TRAINING_RECIPE.json')}
    write(root / 'PROTOCOL.json', protocol)
    dependencies = {**old_identity['code'], **diagnostic['code']}
    for p in sorted((repo / 'analyse/final_configuration_confirmation_v1').glob('*.py')):
        dependencies[str(p.relative_to(repo))] = sha(p)
    for p in sorted((repo / 'tests/final_configuration_confirmation_v1').glob('*.py')):
        dependencies[str(p.relative_to(repo))] = sha(p)
    verify(repo, dependencies)
    write(root / 'CODE_MANIFEST.json', {'files': dependencies, 'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'workspace_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        'versions':{n:importlib.metadata.version(n) for n in ['torch','torch-geometric','numpy','scipy','gpytorch','botorch','pytest']},
        'python':sys.version,'platform':platform.platform(), 'core_changes':0})
    write(root / 'SOURCE_COSTS.json', {'query_cache': {'tasks':528,'task_seconds':p4['task_seconds'],
        'P4_wall_seconds':p4['wall_seconds'],'amortization':'historical fixed cache, 0 new query training; logical calls still charged'},
        'representation_and_generation': {'source':str(old),'protocol_sha256':sha(old / 'performance_protocol.json'),
            'pool_source_checkpoint_hashes':sorted({c['source_checkpoint_sha256'] for c in qualified}),
            'new_labelled_history':0,'new_VAE_or_candidate_generation':0,
            'historical_upstream_compute_cost':'not reconstructed as zero; original source artifacts retained; current work reuses fixed pool/features'}})
    write(root / 'H_QUALIFICATION.json', {'status':'NOT_CERTIFIED','candidate_indices':{'first':640,'last':1707,'count':1068},
        'excluded_known_development':{'first':0,'last':639},'excluded_old_test':{'first':1708,'last':2707},
        'evidence':'current masked loss/validation/test paths and October diagnostics; historical archive_best has legacy_unrecorded data/split bindings',
        'missing':'historical completed run to actual data/split/runtime binding; architecture eligibility is not node-label eligibility',
        'new_labels_or_scores_read':0,'member_choice_by_class_or_scores':False,'blocks':'new final confirmation only'})
    names = ['PROTOCOL.json','CODE_MANIFEST.json','TRAINING_RECIPE.json','SPLITS.json','SEEDS.json',
        'INITIALIZATIONS.json','PUBLIC_C.json','CONFIGURATIONS.json','development_data.pt','GPUS.json',
        'CACHE_ACCEPTANCE.json','HISTORICAL_SEEDS.json','SOURCE_COSTS.json','H_QUALIFICATION.json','private/QUERY_CACHE.json']
    inputs = {n:sha(root / n) for n in names}; freeze = {'inputs':inputs,'time':now()}
    freeze['identity'] = digest(freeze); write(root / 'FREEZE.json', freeze)
    for rel in dependencies:
        dest = root / 'frozen_code' / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(repo / rel,dest)
    write(root / 'TASK_MANIFEST.json', {'tasks':[],'immutable_group_manifests':{},'pending_dependencies':20,
        'planned_roles':{'rerank_seed':600,'rerank_mask':600,'evaluation_logical':600,'evaluation_unique':[200,600]},
        'logical_budget':13380,'query_cache_tasks_verified':528})
    write(root / 'STATUS.json', {'state':'READY','phase':'PROTOCOL_FROZEN','time':now(),'formal_initializations':0})
    csvwrite(root / 'BUDGET_LEDGER.csv', [{'group':g,'rule':r,'query':213 if r=='U' else 183,
        'rerank':0 if r=='U' else 30,'evaluation':10,'logical_total':223,'state':'PLANNED'} for g in range(20) for r in RULES])
    (root / 'PRETRAINING_COSTS.md').write_text('正式逻辑额度13,380；旧查询缓存528任务已逐文件验证，新增查询训练0。\n'
        'S/M新种子逐组不重复，因此复评物理任务精确为600+600=1200。\n'
        '返回配置由合法轨迹和复评决定；评估物理任务为10×Σ每组不同返回配置数，精确范围200–600。\n'
        '总新增1400–1800；每批配置身份在其合法依赖完成后、训练前封存。先运行6个正式任务，包含在上述额度内。\n'
        '不按全池真值生成前5；不预先生成未查询候选的训练。历史上游表示/候选生成费用来源另见SOURCE_COSTS.json。\n')
    print(json.dumps({'state':'READY','run_dir':str(root),'logical':13380,'new_training':[1400,1800],'GP_fits':420,'GP_starts':1260}), flush=True)
