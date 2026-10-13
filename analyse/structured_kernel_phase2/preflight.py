"""Record the observed scientific stop before revealing B or running experiments.

This entry point implements only the blocked preflight reached by the approved
plan. It deliberately does not provide a bypass, label loader, or replay runner.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

PREP = 'results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2'
PARTITION = 'results/posthoc/fuzzy-metric-stage2-20260910-v1/partition.json'
PHASE1 = 'results/structured_kernel_phase1_20261002T142841Z'
PINNED = {
    PARTITION: 'dba3ef021d1d13417b4eedbf83730a8862f4bc7de8a8b831ccb58cd115ab5643',
    PREP + '/data_use.json': '6f7c4e0fd536cbd875ce71b7b931cec57a6a47e38f3ed9321b0ad71c735d5019',
    PREP + '/candidates.json': 'f686a5c4600c511bd27be81d1f3bef0267139affde4adffece1e7c58e81ff728',
    PREP + '/manifest.json': '338a32a367e5901a16990c85a609f5cc7afd550a81324f8e1cd65c634dd55296',
    PREP + '/provenance.json': '0d4bb619ef8856383f3a5822ce5c2cb1d28fe773e8cd29e2b9b7cb9a59f19b81',
    PREP + '/base_encoding.json': 'fe0610eec6be45e69074d03d12fea3f712f6feecdaf261d02a56c5bfa3156a5e',
}
REASON = 'B_HISTORICAL_DEVELOPMENT_CONFLICTS_WITH_INDEPENDENT_CONFIRMATION'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def assess_data_use(partition, data_use):
    """Classify metadata, without accepting performance records as input."""
    declarations = {'partition_data_use': partition.get('data_use'),
                    'independent_confirmation': data_use.get('independent_confirmation'),
                    'use': data_use.get('use'), 'reason': data_use.get('reason')}
    missing = any(declarations[k] is None for k in declarations)
    historical = (declarations['use'] == 'development_only' or
                  'not independent confirmation' in str(declarations['partition_data_use']).lower())
    return {'state': 'BLOCKED' if missing or historical else 'REQUIRES_FULL_PREFLIGHT',
            'reason_code': 'DATA_USE_METADATA_MISSING' if missing else REASON if historical else None,
            'declarations': declarations,
            'B_revealed': False, 'interpretation': 'A scientific eligibility decision, not a performance test.'}


def validate_resume(root, identity):
    status = read(root / 'status.json')
    if status['run_identity'] != identity:
        raise ValueError('Resume identity changed; do not reuse this run or reveal B.')
    if status['state'] != 'BLOCKED' or status['B_revealed']:
        raise ValueError('This entry point can resume only its unrevealed blocked preflight.')
    return status


def source_audit(repo):
    files = {}
    for relative, expected in PINNED.items():
        actual = sha(repo / relative)
        if actual != expected:
            raise ValueError('Pinned metadata changed; repeat scientific audit: ' + relative)
        files[relative] = actual
    phase1 = repo / PHASE1
    frozen = read(phase1 / 'frozen_manifest.json')
    for relative, expected in frozen['code_files'].items():
        if sha(repo / relative) != expected:
            raise ValueError('Phase1 frozen source changed: ' + relative)
    for relative, expected in frozen['input_files'].items():
        if sha(phase1 / relative) != expected:
            raise ValueError('Phase1 frozen input changed: ' + relative)
    protected = read(phase1 / 'baseline.json')['protected_files']
    if any(sha(repo / name) != expected for name, expected in protected.items()):
        raise ValueError('Protected production files changed.')
    partition, use = read(repo / PARTITION), read(repo / PREP / 'data_use.json')
    eligibility = assess_data_use(partition, use)
    if eligibility['reason_code'] != REASON:
        raise ValueError('Eligibility evidence differs from the approved blocked plan; new scientific audit required.')
    candidates = read(repo / PREP / 'candidates.json')
    from analyse.structured_kernel_phase1.data import CANDIDATE_KEYS
    from analyse.structured_kernel_phase1.features import build_operator_graph, graph_hash, isomorphic_operator_graph
    if any(set(row) != CANDIDATE_KEYS for row in candidates):
        raise ValueError('Unexpected fields in unlabelled configuration source.')
    indexed = {row['candidate_id']: row for row in candidates}
    a_ids, b_ids = partition['A_ids'], partition['B_ids']
    if len(indexed) != len(candidates) or set(a_ids) & set(b_ids) or set(indexed) != set(a_ids) | set(b_ids):
        raise ValueError('Candidate partition identity mismatch.')
    groups = {row['architecture_fingerprint']: build_operator_graph(row) for row in candidates}
    a_arch = sorted({indexed[c]['architecture_fingerprint'] for c in a_ids})
    b_arch = sorted({indexed[c]['architecture_fingerprint'] for c in b_ids})
    cross = [(a, b) for a in a_arch for b in b_arch if isomorphic_operator_graph(groups[a], groups[b])]
    aliases = [(a, b) for i, a in enumerate(b_arch) for b in b_arch[i + 1:] if isomorphic_operator_graph(groups[a], groups[b])]
    if cross or aliases:
        raise ValueError('Architecture relations changed from the audited plan.')
    b_folds = {a: int(graph_hash(groups[a]), 16) % 5 for a in b_arch}
    split = {'status': 'UNLABELLED_ASSIGNMENT_ONLY', 'not_confirmation_eligible': True,
             'rule': 'exact isomorphism class minimum graph_hash, int(hash,16)%5; each observed class is a singleton',
             'architecture_fold': b_folds, 'architecture_graph_hash': {a: graph_hash(groups[a]) for a in b_arch},
             'fold_ids': {str(f): sorted(c for c in b_ids if b_folds[indexed[c]['architecture_fingerprint']] == f) for f in range(5)}}
    split['candidate_counts'] = [len(split['fold_ids'][str(f)]) for f in range(5)]
    split['architecture_counts'] = [sum(v == f for v in b_folds.values()) for f in range(5)]
    if not all(split['candidate_counts']):
        raise ValueError('Empty B fold.')
    seal = read(phase1 / 'prediction_seal.json')
    models, training = {}, {}
    for draw in (20260922, 20260923):
        for fold in range(5):
            subset = f's{draw}_f{fold}_n50'
            training[subset] = sha(phase1 / 'training' / (subset + '.json'))
            for method in ('W-I', 'RAW16'):
                setting = subset + '_' + method
                path = phase1 / 'settings' / setting
                model_hash = sha(path / 'model.pt')
                if model_hash != seal['settings'][setting]['model_sha256']:
                    raise ValueError('Selected checkpoint hash changed: ' + setting)
                models[setting] = {'model_sha256': model_hash, 'setting_identity': read(path / 'identity.json')['identity']}
    report = read(phase1 / 'report_status.json')
    if report['counts']['COMPLETE'] != 100 or report['starts_successful'] != 300 or report['selected_method'] != 'W-I':
        raise ValueError('Phase1 eligibility changed.')
    source = {'files': files, 'phase1_frozen_identity': frozen['identity'], 'protected_files': protected,
              'A_candidates': len(a_ids), 'A_architectures': len(a_arch), 'B_candidates': len(b_ids),
              'B_architectures': len(b_arch), 'B_id_hash': digest(sorted(b_ids)),
              'A_B_exact_graph_matches': len(cross), 'B_graph_aliases': len(aliases),
              'training_files': training, 'models': models,
              'recipe_metadata': read(repo / PREP / 'provenance.json')['training_protocol_identity'],
              'label_source_opened': False, 'B_labels_decoded': 0, 'B_SE_decoded': 0, 'test_scores_decoded': 0}
    return source, eligibility, split


def make_package(root):
    excluded = {'evidence_package.tar.gz', 'package_manifest.json', 'package_result.json'}
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and p.name not in excluded]
    if any(p.is_symlink() or p.suffix in ('.pt', '.pth') for p in files):
        raise ValueError('Invalid share-package member.')
    manifest = {str(p.relative_to(root)): sha(p) for p in files}
    write(root / 'package_manifest.json', {'files': manifest, 'scope': 'Unlabelled preflight and scientific stop; no B labels/SE or test.'})
    files.append(root / 'package_manifest.json')
    output = root / 'evidence_package.tar.gz'
    with tarfile.open(output, 'w:gz') as archive:
        for p in files:
            archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
    write(root / 'package_result.json', {'path': str(output), 'sha256': sha(output), 'bytes': output.stat().st_size, 'files': len(files)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo, root = args.repo.resolve(), args.run_dir.resolve()
    if root.parent != repo / 'results' or not root.name.startswith('structured_kernel_phase2_'):
        raise ValueError('Use a new structured_kernel_phase2_* directory directly under repo/results.')
    source, eligibility, split = source_audit(repo)
    code_paths = [repo / 'analyse/structured_kernel_phase2/__init__.py', Path(__file__).resolve(),
                  repo / 'tests/structured_kernel_phase2/test_preflight.py']
    code = {str(p.relative_to(repo)): sha(p) for p in code_paths}
    prompt_hash = sha(args.prompt_file)
    identity = digest({'source': source, 'code': code, 'prompt': prompt_hash, 'eligibility': eligibility, 'split': split})
    command = shlex.join([sys.executable, '-B', '-m', 'analyse.structured_kernel_phase2.preflight', *sys.argv[1:]])
    if args.resume:
        validate_resume(root, identity)
        with (root / 'actual_commands.md').open('a') as stream:
            stream.write(f'\nResume checked at {now()}; unchanged BLOCKED identity, zero label access.\n\n```sh\n{command}\n```\n')
        make_package(root)
        print(json.dumps({'state': 'BLOCKED', 'resumed': True, 'B_revealed': False, 'run_dir': str(root)}))
        return 2
    if root.exists():
        raise ValueError('Run directory exists; use --resume with the exact same identity.')
    root.mkdir()
    (root / 'evidence').mkdir()
    (root / 'code').mkdir()
    write(root / 'source_manifest.json', source)
    write(root / 'B_split_manifest.json', split)
    write(root / 'evidence/data_use_declarations.json', eligibility['declarations'])
    for relative in (PARTITION, PREP + '/data_use.json'):
        (root / 'evidence' / Path(relative).name).write_bytes((repo / relative).read_bytes())
    (root / 'PROMPT.txt').write_bytes(args.prompt_file.read_bytes())
    (root / 'PLAN.md').write_text('''# 第二轮执行计划及当前停止点

沿用已核对的第二轮计划与 PROMPT.txt 完整科学规格：仅 W-I/RAW16，n50；先无标签预检，封存全部B预测，再一次评分；B门槛通过才开展10组同qLogEI配对缓存搜索。

当前执行在第一项科学资格预检停止。数据用途证据没有改变：B 为 historical development/development_only，不能直接作为计划要求的独立确认集。此处不把执行授权解释为修改科学资格，不设置跳过开关。

已实现并执行的范围：无标签来源及原模型身份复核、停止规则、BLOCKED状态、精确身份恢复和证据包。完整实验执行器、WL迁移、采集适配、B预测/评分、A重放未在本次阻塞运行中实现或执行。此前只读规划中的可行性检查不计为本次实验。

恢复命令只复核相同来源/代码/协议身份，保持BLOCKED且不读取标签。解决科学用途冲突必须先取得明确的科学裁决或新的有效证据，并保留本目录；不得通过编辑历史用途记录、改阈值、改划分或反复打开B解除阻塞。
''')
    p1_protocol = read(repo / PHASE1 / 'protocol.json')
    write(root / 'protocol.json', {'version': 1, 'status': 'BLOCKED_BEFORE_REVEAL', 'complete_specification': 'PROMPT.txt',
          'specification_sha256': prompt_hash, 'phase1_protocol': p1_protocol,
          'methods': ['W-I', 'RAW16'], 'n': 50, 'workers': 1, 'threads': 1, 'device': 'cpu', 'dtype': 'float64',
          'B_prediction_settings': 20, 'search_seeds': list(range(2026100200, 2026100210)),
          'replay_fit_settings': 420, 'replay_optimization_starts': 1260, 'queries_per_trajectory': 71,
          'B_bootstrap': {'replicates': 2000, 'seed': 2026100231, 'cluster': 'architecture within fold', 'interval': [0.025, 0.975]},
          'replay_bootstrap': {'replicates': 2000, 'seed': 2026100232, 'cluster': 'paired search seed', 'interval': [0.025, 0.975]},
          'B_gate': {'mae_gain_pp': 0.3, 'positive_folds': 4, 'nlpd_max_worsening': 0.1, 'is90_max_worsening_fraction': 0.05, 'ci_lower_gt': 0},
          'replay_gate': {'mean_gain_pp': 0.3, 'strict_positive_pairs': 7, 'ci_lower_gt': 0, 'all_20_trajectories_valid': True},
          'planned_acquisition': {'class': 'surrogate.accuracy_gp.StableQLogExpectedImprovement', 'sampler': 'SobolQMCNormalSampler',
             'samples': 512, 'q': 1, 'fat': True, 'tau_max': 0.01, 'tau_relu': 1e-6, 'eta': 0.001, 'chunk_size': 64,
             'mc_seed_rule': 'SeedSequence([search_seed,step,1]).generate_state(1,dtype=uint32)[0]',
             'best_f': '(max(observed_mean)-T_y_mean)/T_y_scale', 'posterior': 'standardized latent function only'},
          'implementation_qualified_for_B_reveal': False})
    tasks = [{'task_id': f'B_s{d}_f{f}_{m}', 'state': 'BLOCKED', 'reason_code': REASON} for d in (20260922, 20260923) for f in range(5) for m in ('W-I', 'RAW16')]
    tasks += [{'task_id': f'A_s{s}_{m}_n{n}', 'state': 'BLOCKED', 'reason_code': REASON} for s in range(2026100200, 2026100210) for m in ('W-I', 'RAW16') for n in range(50, 71)]
    write(root / 'settings.json', tasks)
    status = {'state': 'BLOCKED', 'reason_code': REASON, 'blocked_at': now(), 'run_identity': identity, 'B_revealed': False,
              'counts': {'B_predictions_completed': 0, 'B_scores_completed': 0, 'GP_fit_settings_completed': 0,
                         'optimizer_starts_completed': 0, 'GP_trajectories_completed': 0, 'oracle_observations': 0,
                         'new_GNN_trainings': 0, 'KG_runs': 0}, 'source_files': source['files'], 'code_files': code,
              'next_action': 'Resolve scientific data-use conflict; no automatic bypass or label reveal.'}
    write(root / 'status.json', status)
    write(root / 'B_confirmation.json', {**eligibility, 'performance_evaluated': False, 'engineering_gate_passed': None, 'bootstrap_interval': None})
    write(root / 'B_prediction_seal.json', {'status': 'NOT_CREATED', 'reason_code': REASON, 'B_revealed': False, 'prediction_count': 0})
    write(root / 'WL_extension_manifest.json', {'status': 'NOT_EXECUTED_IN_THIS_RUN', 'reason_code': REASON, 'prior_readonly_plan_only': True})
    write(root / 'new_seed_confirmation_manifest.json', {'status': 'NOT_EXECUTED', 'reason_code': REASON, 'candidates': [],
          'candidate_count': 0, 'proposed_training_count': 0, 'future_seeds': [3, 4, 5], 'selection_not_performed': True})
    for name, columns in {'B_predictions.csv': ['candidate_id', 'method', 'draw', 'fold', 'mean_raw', 'latent_variance_raw'],
                          'B_metrics_by_fold.csv': ['fold', 'method', 'mae_pp', 'nlpd', 'is90_pp'],
                          'replay_trajectories.csv': ['seed', 'method', 'slot', 'candidate_id'],
                          'replay_summary.csv': ['seed', 'method', 'terminal_best_val_mean'],
                          'paired_comparisons.csv': ['seed', 'terminal_gain_pp']}.items():
        with (root / name).open('w', newline='') as stream:
            csv.writer(stream).writerow(columns)
    (root / 'oracle_access_log.jsonl').write_text('')
    (root / 'B_unlabelled_audit.md').write_text(f'''# B 无标签科学预检

状态：BLOCKED。B {source['B_candidates']} 个联合候选/{source['B_architectures']} 架构；A/B精确有向类型同构匹配0，B内部同构别名0。五折候选数 {split['candidate_counts']}，架构数 {split['architecture_counts']}。这些无标签结构事实不消除历史开发用途冲突。

partition.json:540 声明 `{eligibility['declarations']['partition_data_use']}`。
data_use.json:196–198 声明 independent_confirmation=[]、use=development_only；理由为 `{eligibility['declarations']['reason']}`。

当前仍有10个原n50训练投影、20个选中模型，其哈希与第一轮封存记录一致。原代码和输入哈希通过复核。B标签文件未打开，B mean/SE未校验，因此没有B性能结论。
''')
    (root / 'acquisition_audit.md').write_text('''# 采集执行状态

NOT_EXECUTED：在科学资格预检停止。本轮没有构造、评分或测试qLogEI适配器，不宣称已经完成同qLogEI搜索比较。原只读计划已定位StableQLogExpectedImprovement；拟采用的固定参数保留在protocol.json，但不等于实现验收通过。
''')
    (root / 'README_RESULTS.md').write_text(f'''# 第二轮执行结果：BLOCKED

本次按Prompt B及已确认计划执行预检，合法停止在B预测封存/揭示之前。**第二轮性能实验未完成，也未失败于性能门槛。**

阻塞原因：原分区明确注明B属于历史开发数据、不是独立确认；prepared_v2/data_use.json同时注明development_only且独立确认集合为空。来源文本、原文件和SHA256见evidence/及source_manifest.json。最新执行授权没有更改这一科学规格。

1. B实际为{source['B_candidates']}候选/{source['B_architectures']}架构，配方元数据沿用Cora/public/NormalizeFeatures/full_batch/global4/No-LN、300/80。A/B没有精确同构架构重合。这说明B结构不在本轮A集合内，不能据此把历史开发池变为独立保留集。
2. B预测、评分和bootstrap均未执行；B_revealed=false。不能判断W-I在B上通过或未通过。
3. qLogEI适配及其正式验收未在此阻塞运行中实施，未声称两方法已有采集比较结果。
4. 完成GP轨迹0/20，拟合设置0/420，优化起点0/1260，缓存查询0，新GNN训练0、KG运行0。
5. 当前不能判断第一轮预测收益是否转化为有限池搜索收益。
6. 第一轮W-I收益仍属于A上的探索性结果；不升级为B独立确认、搜索成功或test结论。
7. 没有生成新seed复核候选，当前不建议据本次阻塞运行启动新GNN训练。

已实现的代码只负责科学预检、来源身份、阻塞状态、幂等恢复与证据打包；没有交付一个假装完整可运行的实验执行器。CSV只有表头表示未执行，绝不把空值或0当作性能成绩。oracle日志为空表示零次标签释放。

需先明确解决科学用途冲突或提供新的有效来源证据；不得修改原历史声明、换阈值、换方法、重划分或反复读取B来解除阻塞。当前目录可做身份一致的恢复核验，不能自动进入后续实验。
''')
    (root / 'GPT_ANALYSIS_REQUEST.md').write_text('''# GPT 分析请求

请分析本次第二轮BLOCKED证据包。先核对status.json、B_confirmation.json及evidence中的原始用途声明，明确区分“B与A结构无重合”和“B从未参与历史模型开发”。结合source_manifest.json的来源哈希引用具体证据。

本次B标签/SE未揭示，预测/评分/搜索/训练均为零；表头CSV不是性能结果。不得计算不存在的B指标，不得将阻塞解释为W-I被证明无效或未通过性能门槛，也不得把第一轮A探索收益升级为独立确认。

请指出科学用途冲突需要何种明确裁决或来源证据；不要建议偷偷降级确认含义、改阈值、换W-A或重划B，不要直接执行后续实验。区分事实、解释和假说。
''')
    (root / 'actual_commands.md').write_text(f'# 真实执行命令\n\n工作目录：`{repo}`；记录时间：{now()}。退出码2表示已保存的科学BLOCKED，不是优化数值失败。\n\n```sh\n{command}\n```\n')
    patch = []
    for path in code_paths:
        relative = str(path.relative_to(repo))
        target = root / 'code' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        patch.extend(difflib.unified_diff([], path.read_text().splitlines(keepends=True), fromfile='/dev/null', tofile='b/' + relative))
    (root / 'code_diff.patch').write_text(''.join(patch))
    write(root / 'baseline.json', {'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip(),
          'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'protected_files': source['protected_files']})
    write(root / 'resume_manifest.json', {'run_identity': identity, 'B_revealed': False, 'completed_fit_states': [],
          'sampler_states': [], 'revealed_ids': [], 'logical_budget_consumed': 0})
    make_package(root)
    print(json.dumps({'state': 'BLOCKED', 'reason_code': REASON, 'B_revealed': False, 'run_dir': str(root)}))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
