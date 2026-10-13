"""Read authoritative server evidence, then freeze all scientific choices."""
import importlib.metadata
import inspect
import platform
import shutil
import subprocess
import sys
import numpy as np
import torch
from .common import *
from .data import reconstruct_development, make_splits

def make_model(c, in_ch, out_ch, device):
    import eval_utils as original
    return original.DynamicGNN(c, in_ch, out_ch, dropout=float(c['dropout']),
        hidden_dim=int(c['hidden_dim']), gcnii_alpha=float(c.get('gcnii_alpha', .1)),
        gcnii_theta=float(c.get('gcnii_theta', .5)), gat_heads=int(c['gat_heads']),
        sage_aggr=c['sage_aggr'], gin_eps=float(c['gin_eps']),
        gat_heads_by_layer=c.get('gat_heads_by_layer'), sage_aggr_by_layer=c.get('sage_aggr_by_layer'),
        gin_eps_by_layer=c.get('gin_eps_by_layer'), component_variant='baseline',
        activation_variant='relu').to(device)

def seed_inventory(repo, ids):
    import re
    observations = []
    manifests = []
    for base in ('results', 'artifacts'):
        for p in (repo / base).rglob('*'):
            if not p.is_file():
                continue
            name = str(p.relative_to(repo))
            for cid in ids:
                for match in re.finditer(re.escape(cid) + r'_s(\d+)(?:/|\.|$)', name):
                    observations.append({'path': name, 'candidate_id': cid, 'seed': int(match.group(1)),
                                         'basis': 'existing task identity path; scores not opened'})
            if p.suffix == '.json' and 'manifest' in p.name and ('task' in p.name or 'training' in p.name):
                manifests.append(p)
    for p in manifests:
        value = read(p)
        def walk(x):
            if isinstance(x, dict):
                cid = x.get('candidate_id') or x.get('config_id')
                if cid in ids and isinstance(x.get('seed'), int):
                    observations.append({'path': str(p.relative_to(repo)), 'candidate_id': cid,
                                         'seed': x['seed'], 'basis': 'task manifest identity only'})
                for v in x.values():
                    if isinstance(v, (dict, list)):
                        walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(value)
    unavailable = {r['seed'] for r in observations} | {0, 1, 2}
    seeds = []
    n = 100
    while len(seeds) < 10:
        if n not in unavailable:
            seeds.append(n)
        n += 1
    return {'D1': seeds, 'D2_per_split': seeds[:3], 'rule': 'ascending from 100; conservative exclusion of any existing exact config task identity, without scores',
            'observed_seed_values': sorted(unavailable), 'identity_records': observations,
            'task_manifest_files': {str(p.relative_to(repo)): sha(p) for p in manifests},
            'scope': 'all server results/artifacts task identity paths and training/task manifests; does not claim recovery of unrecorded deleted work',
            'scoring_values_used_to_choose_seed': False}

def preflight(root, repo, prompt):
    if (root / 'freeze.json').exists():
        verify_run(root, repo)
        return
    root.mkdir(parents=True, exist_ok=True)
    old = repo / OLD
    code = read(old / 'performance_identity.json')['code']
    differences = {p: {'frozen': h, 'current': sha(repo / p)} for p, h in code.items() if sha(repo / p) != h}
    assert not differences, 'Frozen training dependency changed'
    oldp = read(old / 'performance_protocol.json')
    dataset = read(repo / DATA_MANIFEST)
    assert all(dataset[k] == v for k, v in oldp['data'].items())
    ret = read(old / 'returned_configurations.json')
    ids = sorted({r['candidate_id'] for r in ret['returns']})
    assert len(ids) == 5 and len(ret['returns']) == 30
    q = read(old / 'C_qualified_manifest.json')['configurations']
    assert len(q) == 176 and len({c['architecture_fingerprint'] for c in q}) == 44
    cache = read(old / 'private/C_validation_cache.json')
    assert len(cache) == 176
    ordered = sorted(cache, key=lambda cid: (-cache[cid]['mean'], cid))
    assert set(ordered[:2]).issubset(ids), 'Original validation top two not both returned'
    ids = sorted(ids, key=lambda cid: (-cache[cid]['mean'], cid))
    assert ids[:2] == ordered[:2]
    oldtests = read(old / 'test_results.json')
    oldrows = []
    candidates = []
    reconstruction = []
    for i, cid in enumerate(ids):
        c = next(c for c in q if c['candidate_id'] == cid)
        assert c['candidate_id'] == digest({'architecture_fingerprint': c['architecture_fingerprint'],
                                          'hp_fingerprint': c['hp_fingerprint']})
        assert c['normalization'] == 'none' and c['hp_mode'] == 'global4'
        model = make_model(c, dataset['num_features'], dataset['num_classes'], torch.device('cpu'))
        assert len(model.layer_norms) == 0 and model.edges == c['edges'] and model.ops == c['operations']
        checkpoints = []
        vals = []
        tests = []
        for seed in [0, 1, 2]:
            tid = cid + '_s' + str(seed)
            rec = ret['checkpoints'][tid]
            p = Path(rec['path'])
            assert sha(p) == rec['sha256']
            state = torch.load(p, map_location='cpu', weights_only=False)
            assert state['identity'] == rec['training_identity']
            model.load_state_dict(state['model'], strict=True)
            result = read(old / 'training' / tid / 'result.json')
            assert result['best_restored_validation_accuracy'] == state['validation_accuracy']
            t = read(old / 'test_evaluations' / (tid + '.json'))
            assert t['checkpoint_sha256'] == rec['sha256']
            vals.append(result['best_validation_accuracy'])
            tests.append(t['test_accuracy'])
            checkpoints.append(rec)
            oldrows.append({'alias': f'C{i}', 'config_id': cid, 'seed': seed,
                            'selection': vals[-1], 'old_test': tests[-1], 'checkpoint_sha256': rec['sha256']})
        assert np.isclose(np.mean(vals), cache[cid]['mean'], atol=1e-15, rtol=0)
        assert np.isclose(np.std(vals, ddof=1) / np.sqrt(3), cache[cid]['se'], atol=1e-15, rtol=0)
        assert np.isclose(np.mean(tests), oldtests['configurations'][cid]['mean'], atol=1e-15, rtol=0)
        candidates.append({'alias': f'C{i}', 'config_id': cid, 'configuration': c,
            'configuration_hash': digest(c), 'original_returns': [r for r in ret['returns'] if r['candidate_id'] == cid],
            'original_checkpoints': checkpoints, 'old_selection_mean': float(np.mean(vals)),
            'old_test_mean': float(np.mean(tests))})
        reconstruction.append({'config_id': cid, 'ops': model.ops, 'edges': model.edges,
            'layer_classes': [type(layer).__name__ for layer in model.layers],
            'parameters': sum(p.numel() for p in model.parameters()), 'all_three_checkpoints_strict_load': True,
            'layer_norm_count': len(model.layer_norms), 'gnn_training_or_forward': 0})
    dev, original = reconstruct_development(repo, dataset)
    from analyse.structured_kernel_followup.validation_train import save_torch
    save_torch(root / 'development_data.pt', dev)
    splits = make_splits(original, dev['development_y'])
    seeds = seed_inventory(repo, ids)
    recipe = {**oldp['recipe'], 'seed': 'see seed_manifest.json',
        'max_epochs': 300, 'patience': 80, 'min_improvement': 0.0, 'checkpoint_comparison': 'strict val > best_val; initial best_val=0; first exact tie wins',
        'early_stop_and_selection_same_nodes': True, 'restoration': 'load best_state and verify exact selection accuracy',
        'checkpoint_every_epochs': 1, 'resume': 'model optimizer scheduler all RNG next epoch best state and patience counter',
        'numerical_epoch': 'unmodified frozen validation_train.epoch_core',
        'precision': 'float32', 'test_during_training': False,
        'cache_identity': 'run code/protocol/config/dataset/all masks/recipe/checkpoint rule/seed',
        'checkpoint_io_change': 'save after every complete epoch; numerical training equations unchanged'}
    comparison = {'primary': ['C0', 'C1'], 'supplementary': [['C0', x] for x in ['C2', 'C3', 'C4']],
        'candidate_order': 'old validation mean descending, full config_id ascending on exact ties',
        'ranking_tie_break': 'full config_id ascending; statistical ties exact accuracy equality',
        'units': 'percentage points', 'D1_unit': 'shared new seed', 'D2_unit': 'split, after mean of paired seeds',
        'missing': 'pairwise common completed seeds only; no imputation; descriptive only if incomplete',
        'bootstrap': {'n': 10000, 'seed': 20261007, 'CI': [.025, .975], 'D1': 'shared indices across all configurations and metrics',
                      'D2': 'shared whole-split indices; exploratory, 5 overlapping masks on same graph'},
        'multiple_testing': 'no formal hypothesis tests or multiplicity-adjusted significance claims; supplementary descriptive CIs unadjusted',
        'old3_uncertainty': 'bootstrap means of 3 draws from D1 paired-difference empirical distribution; approximate, not an independent CI for historical sample',
        'D2_reference': 'same first three D1 seeds on original mask; pairwise seed matching',
        'reversal': 'strict sign(selection C0-C1) * sign(test C0-C1) < 0; ties reported separately'}
    tasks = []
    for phase, split_names, train_seeds in [('D1', ['original'], seeds['D1']),
                                            ('D2', [f'split{i}' for i in range(1, 6)], seeds['D2_per_split'])]:
        for split in split_names:
            for seed in train_seeds:
                for c in candidates:
                    tasks.append({'task_id': f'{phase}_{split}_{c["alias"]}_s{seed}', 'phase': phase,
                                  'split': split, 'alias': c['alias'], 'config_id': c['config_id'],
                                  'configuration_hash': c['configuration_hash'], 'seed': seed})
    assert len(tasks) == 125
    protocol = {'version': 'rank-reversal-1', 'frozen_at': now(), 'D1': 50, 'D2': 75,
        'maximum_random_initializations_including_failures': 125,
        'D1_test_rule': 'Only after all D1 terminal records/checkpoints sealed, independent D4 score once per successful checkpoint; old-task repeated diagnosis',
        'D2_test_evaluations': 0, 'new_search_GP_VAE_KG': 0,
        'failure': 'preserve logs; no automatic from-initialization retry; failed matrix remains missing; strict resume only with complete latest state',
        'stop': 'identity mismatch, test entering development, unrecoverable protocol, changed scientific recipe => stop affected work; no result-based stopping',
        'resources': 'single worker on available GPU with >=2048MiB; device locked; never interrupt others',
        'interruption_acceptance': 'first planned task pauses after saved epoch5, resumes in fresh process; controller stops after two completed tasks then resumes, preserving hashes',
        'analysis': comparison, 'scope_end': 'D5 report/package; no automatic next-stage experiments'}
    for name, value in [('dataset_manifest.json', dataset), ('candidate_manifest.json', {'candidates': candidates}),
                        ('seed_manifest.json', seeds), ('split_manifest.json', splits), ('comparison_manifest.json', comparison),
                        ('training_recipe.json', recipe), ('task_manifest.json', {'tasks': tasks}), ('diagnostic_protocol.json', protocol),
                        ('configuration_reconstruction.json', reconstruction)]:
        write(root / name, value)
    csvwrite(root / 'OLD_RESULTS.csv', oldrows)
    methods = ['W-I', 'RAW16', 'Random']
    oldmeans = {m: {'validation': float(np.mean([r['validation_mean'] for r in ret['returns'] if r['method'] == m])),
                   'test': float(np.mean([oldtests['configurations'][r['candidate_id']]['mean'] for r in ret['returns'] if r['method'] == m]))} for m in methods}
    write(root / 'old_context_recomputed.json', {'methods': oldmeans,
        'W-I_minus_Random_selection_pp': 100 * (oldmeans['W-I']['validation'] - oldmeans['Random']['validation']),
        'W-I_minus_Random_test_pp': 100 * (oldmeans['W-I']['test'] - oldmeans['Random']['test']),
        'C0_C1_original_reversal': (candidates[0]['old_selection_mean'] - candidates[1]['old_selection_mean']) * (candidates[0]['old_test_mean'] - candidates[1]['old_test_mean']) < 0})
    source_names = ['README_RESULTS.md', 'GPT_ANALYSIS_REQUEST.md', 'performance_protocol.json', 'returned_configurations.json',
        'C_eligibility_manifest.json', 'precautionary_exclusion.json', 'P1_summary.json', 'QUALIFICATION_READING_NOTE.md',
        'P4_complete.json', 'P6_complete.json', 'actual_commands.md', 'performance_identity.json']
    inputs = {str(old / n): sha(old / n) for n in source_names}
    inputs[str(repo / DATA_MANIFEST)] = sha(repo / DATA_MANIFEST)
    inputs[str(old / 'private/C_validation_cache.json')] = sha(old / 'private/C_validation_cache.json')
    write(root / 'source_manifest.json', inputs)
    (root / 'original_evidence').mkdir(exist_ok=True)
    for name in source_names:
        shutil.copy2(old / name, root / 'original_evidence' / name)
    shutil.copy2(prompt, root / 'PROMPT.txt')
    plan_matches = [str(p) for p in Path('/home/fangxy').glob('**/GNN_NAS_test提升实验方案.txt') if p.is_file()]
    environment = {'python': sys.version, 'platform': platform.platform(),
        'versions': {n: importlib.metadata.version(n) for n in ['torch', 'torch-geometric', 'numpy', 'scipy', 'pytest']},
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'plan_text_paths': plan_matches, 'prompt_authorizes_explicit_design_when_plan_text_absent': True,
        'frozen_dependency_differences': differences}
    write(root / 'environment.json', environment)
    (root / 'git_status_before.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True))
    (root / 'workspace_diff_before.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=repo, text=True))
    core = {str(p.relative_to(repo)): sha(p) for p in sorted((repo / 'analyse/rank_reversal_diagnostic').glob('*.py'))}
    for n in ['eval_utils.py', 'dataset_utils.py', 'deterministic_runtime.py', 'hp_modes.py', 'evaluation_errors.py',
              'analyse/structured_kernel_followup/validation_train.py', 'analyse/structured_kernel_followup/common.py']:
        core[n] = sha(repo / n)
    write(root / 'code_identity.json', core)
    for n in core:
        target = root / 'frozen_code' / n
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / n, target)
    d0 = f'''# D0 核对

服务器权威输入：{old}。当前 HEAD {environment['head']}；相关冻结代码差异为零。五个正式返回配置由封存清单去重，C0/C1 从全池原验证成绩的前两名确定，未按 test 选候选。原资格结论沿用 P1_summary/C_eligibility_manifest；配置内历史 PROVENANCE_UNRESOLVED 不作为现行结论。

数据为 Cora / PyG {environment['versions']['torch-geometric']}，2708 节点、10556 有向存储边（无向图）、1433 特征、7 类。全图结构和 NormalizeFeatures 后无标签特征参与传播，是 transductive full-batch；开发标签仅原 train∪validation 的640节点。内容身份 {dataset['dataset_content_fingerprint']}；原 masks 身份 {dataset['split_fingerprint']}。

原 train=140（各类20），early-stop=selection=500。精确每类数量、索引和哈希见 split_manifest；test=1000，仅记录索引/数量/哈希，不计算类别分布。数据重建复用 PyG read_file/edge_index_from_dict/NormalizeFeatures，逐张量核对 x/edge/masks 与既有数据 manifest 完全相同。D0 仅解码 allx、tx、graph、test.index、ally、y；未打开 ty 或 processed/data.pt 的反序列化入口；它们仅进行字节哈希。ally 非开发行不取 argmax、不提供给训练。训练进程只持有其余标签为-100的 development_data.pt，且原 Cora 路径访问被审计钩子拒绝。

实际调用链：c_independent_validation.performance.train_one → structured_kernel_followup.validation_train.train_validation_only → eval_utils.DynamicGNN；epoch_core 原样用于本轮。Adam，配置的 lr/l2/dropout/hidden，cosine T_max=300、eta_min=lr×0.01，float32、strict deterministic、baseline/ReLU、无 LayerNorm、max300/patience80。初始 best=0，严格 val>best 才更新，最小改进0，精确同分保留首个epoch，恢复 best_state 后验证准确率必须完全相等。配置解析与原15个checkpoint严格装载检查见 configuration_reconstruction.json。

原搜索标签是 seed0/1/2 各自最佳验证准确率的平均，恢复后评分等于最佳验证值；SE=std(ddof=1)/sqrt(3)。本轮不涉及 GP。冻结 C 训练器没有逐epoch test 计算；生产 eval_utils 的可选 track_test 分支只写日志/best_test，并不影响 loss、val比较或patience；上一轮实际训练调用未调用该分支，test在返回封存后的独立入口进行。因此本轮没有改变训练控制。

训练更改仅为显式注入 masks、每完整epoch保存最新训练状态、故意中断/完整恢复及 selection logits 存档。D1 50任务、D2 75任务；D2保持各类train=20及原validation各类配额。默认seed从100起依身份排除，实际列表{seeds['D1']}；D2使用其中前三个。随机数与重叠明细已冻结。原始三seed结果分开保存。

原上下文数字从封存结果复算见 old_context_recomputed.json，不强制匹配文字近似值。补充方案文本路径：{plan_matches or '服务器未找到；按本Prompt明示规格执行，这不是原始数据缺失'}。

停止条件：身份/原配方不符、test进入开发、科学目标改变时停止受影响组。失败不换seed，不以结果方向增减样本。正常总初始化上限125，D2 test次数必须为0。全部输入与科学统计规则先封存，之后才能训练。
'''
    (root / 'D0_AUDIT.md').write_text(d0)
    frozen = {'inputs': {n: sha(root / n) for n in FROZEN}, 'code': core, 'time': now()}
    frozen['identity'] = digest(frozen)
    write(root / 'freeze.json', frozen)
    write(root / 'status.json', {'D0': 'COMPLETE', 'D1': 'PENDING', 'D2': 'PENDING', 'D4': 'PENDING', 'D5': 'PENDING', 'updated_at': now()})
    print(json.dumps({'D0': 'COMPLETE', 'tasks': len(tasks), 'seeds': seeds['D1'], 'identity': frozen['identity']}), flush=True)
