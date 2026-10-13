"""Independent retrospective scoring after every trajectory is sealed."""
import csv
import difflib
import json
import sqlite3
import tarfile
import time
from pathlib import Path
import numpy as np
from analyse.structured_kernel_phase1.io import read_json,write_json,write_csv,sha256,now
from analyse.structured_kernel_phase1.data import read_A_labels
from .protocol import METHODS,SEEDS
from .learner import validate_complete
from .oracle import access_guard,Ledger


def bootstrap(gains):
    values=np.asarray(gains,dtype=float)
    rng=np.random.default_rng(2026100232)
    samples=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
    return np.quantile(samples,[.025,.975]).tolist()


def gate(gains,valid):
    interval=bootstrap(gains)
    checks={'20_complete_trajectories':valid==20,'mean_n71_gain_ge_0.3pp':float(np.mean(gains))>=.3,
        'at_least_7_strictly_positive_seeds':sum(v>0 for v in gains)>=7,'bootstrap_lower_gt_0':interval[0]>0}
    return ('DEVELOPMENT_REPLAY_GAIN_SUPPORTED' if all(checks.values()) else 'DEVELOPMENT_REPLAY_NO_STABLE_GAIN'),checks,interval


def score(root,repo):
    if (root/'summary.json').exists():
        return read_json(root/'summary.json')
    for seed in SEEDS:
        for method in METHODS:
            validate_complete(root/'trajectories'/f'{seed}_{method}')
    counts={'physical_A_label_file_opens':0}
    access_guard(root,repo,'scorer',counts)
    labels=read_A_labels(repo)
    configs=read_json(root/'public_configurations.json')
    protocol=read_json(root/'protocol.json')
    optimum=max(r['mean'] for r in labels.values())
    optids=sorted(c for c,r in labels.items() if r['mean']==optimum)
    trajectories=[];curves=[];fits=[];recommendations={}; bykey={}
    for seed in SEEDS:
        for method in METHODS:
            tr=f'{seed}_{method}';folder=root/'trajectories'/tr
            obs=read_json(folder/'observations.json')
            if any({k:o[k] for k in ('mean','se','seeds')}!=labels[o['candidate_id']] for o in obs):
                raise ValueError('Oracle values differ from frozen A')
            bests=[]
            for query in range(22):
                best=max(o['mean'] for o in obs[:50+query]);bests.append(best)
                curves.append(dict(seed=seed,method=method,new_queries=query,kind='initial' if query==0 else 'greedy' if query==21 else 'qlogei',
                    best_validation_mean=best,best_pp=100*best,cached_simple_regret_pp=100*(optimum-best)))
            local_fits=[read_json(folder/'fits'/f'n{n:03d}'/'fit.json') for n in range(50,71)]
            decisions=[read_json(folder/'decisions'/f'{q:02d}.json') for q in range(1,22)]
            if [d['kind'] for d in decisions]!=['qlogei']*20+['greedy']:
                raise ValueError('Decision-kind budget mismatch')
            for fit in local_fits:
                for start in fit['starts']:
                    fits.append(dict(trajectory=tr,n=fit['n'],selected=start['start_id']==fit['selected_start'],**start))
            row=dict(seed=seed,method=method,initial_best_pp=100*bests[0],headroom_pp=100*(optimum-bests[0]),
                initial_contains_optimum=any(o['candidate_id'] in optids for o in obs[:50]),
                n70_best_pp=100*bests[20],n71_best_pp=100*bests[21],greedy_increment_pp=100*(bests[21]-bests[20]),
                regret_integral_q0_20=np.trapezoid(100*(optimum-np.asarray(bests[:21])))/20,
                fit_seconds=sum(f['fit_seconds'] for f in local_fits),acquisition_seconds=sum(d['acquisition_seconds'] for d in decisions),
                fits=21,starts=63,convergence_warnings=sum(r['convergence_warning'] for f in local_fits for r in f['starts']),failures=0,valid=True)
            trajectories.append(row);bykey[(seed,method)]=row
            chosen=min(obs,key=lambda o:(-o['mean'],o['candidate_id']))
            recommendations[tr]=dict(candidate_id=chosen['candidate_id'],validation_mean=chosen['mean'],validation_se=chosen['se'],
                configuration=configs[chosen['candidate_id']],recipe=protocol['recipe'],selection='best actually observed among 71')
    paired=[]
    for seed in SEEDS:
        w,r=bykey[(seed,'W-I')],bykey[(seed,'RAW16')]
        paired.append(dict(seed=seed,initial_best_pp=w['initial_best_pp'],headroom_pp=w['headroom_pp'],
            initial_contains_optimum=w['initial_contains_optimum'],n70_WI_pp=w['n70_best_pp'],n70_RAW16_pp=r['n70_best_pp'],
            n71_WI_pp=w['n71_best_pp'],n71_RAW16_pp=r['n71_best_pp'],
            n70_gain_pp=w['n70_best_pp']-r['n70_best_pp'],n71_gain_pp=w['n71_best_pp']-r['n71_best_pp'],
            WI_greedy_increment_pp=w['greedy_increment_pp'],RAW16_greedy_increment_pp=r['greedy_increment_pp'],
            WI_regret_integral=w['regret_integral_q0_20'],RAW16_regret_integral=r['regret_integral_q0_20']))
    gains=[r['n71_gain_pp'] for r in paired];gains70=[r['n70_gain_pp'] for r in paired]
    state,checks,ci=gate(gains,20)
    ledger=Ledger(root,{}, {})
    with ledger.db:
        ledger.db.execute('INSERT INTO sessions(role,opens,time) VALUES(?,?,?)',('scorer',counts['physical_A_label_file_opens'],now()))
    access=ledger.export();ledger.db.close()
    summary=dict(state=state,purpose='development_replay',independent_confirmation=False,pool='A179',
        n71_mean_paired_gain_pp=float(np.mean(gains)),n71_bootstrap_95_percent_ci_pp=ci,
        n70_mean_paired_gain_pp=float(np.mean(gains70)),n70_descriptive_bootstrap_ci_pp=bootstrap(gains70),
        positive_n71_seeds=sum(v>0 for v in gains),initial_sets_containing_optimum=sum(p['initial_contains_optimum'] for p in paired),
        pool_optimum_validation_mean=optimum,pool_optimum_ids=optids,gate=checks,
        greedy_increment_mean_pp={m:float(np.mean([r['greedy_increment_pp'] for r in trajectories if r['method']==m])) for m in METHODS},
        regret_integral_mean={m:float(np.mean([r['regret_integral_q0_20'] for r in trajectories if r['method']==m])) for m in METHODS},
        counts=dict(trajectories=20,fits=len(fits)//3,starts=len(fits),qlogei_queries=400,terminal_greedy_queries=20,
            initial_observations=1000,convergence_warnings=sum(r['convergence_warning'] for r in fits),
            adam_final50_warnings=sum(r['adam_final50_improvement']>1e-4 for r in fits),
            lbfgs_budget_warnings=sum(r['end_reason'] in ('max_iter','max_eval') for r in fits),
            failed_starts=sum(not r['success'] for r in fits),**access),
        fit_seconds=sum(r['fit_seconds'] for r in trajectories),acquisition_seconds=sum(r['acquisition_seconds'] for r in trajectories),
        bootstrap_interpretation='Conditional fixed A pool and already selected methods; random initial sets, not independent confirmation',
        completed_at=now())
    if summary['counts']['logical_releases']!=1420 or summary['counts']['fits']!=420 or summary['counts']['starts']!=1260:
        raise ValueError('Incomplete actual budget')
    write_csv(root/'trajectories.csv',trajectories)
    write_csv(root/'paired_comparisons.csv',paired)
    write_csv(root/'curves.csv',curves)
    write_csv(root/'optimization_summary.csv',fits)
    write_json(root/'returned_candidates.json',recommendations)
    write_json(root/'summary.json',summary)
    plot(root,curves,paired)
    return summary


def plot(root,curves,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for method in METHODS:
        values=np.array([[r['cached_simple_regret_pp'] for r in curves if r['method']==method and r['seed']==s] for s in SEEDS])
        ax[0].plot(range(22),values.mean(0),marker='.',label=method)
    ax[0].axvspan(20,21,alpha=.12,color='gray',label='Terminal greedy')
    ax[0].set(xlabel='New oracle queries (21 = greedy)',ylabel='Mean cached simple regret (pp)',title='A development replay; no independent confirmation')
    ax[0].legend()
    x=np.arange(10)
    ax[1].bar(x-.18,[p['n70_gain_pp'] for p in paired],width=.36,label='n70')
    ax[1].bar(x+.18,[p['n71_gain_pp'] for p in paired],width=.36,label='n71')
    ax[1].axhline(0,color='black',lw=.7);ax[1].set(xlabel='Paired seed suffix',ylabel='W-I minus RAW16 (pp)',xticks=x,xticklabels=range(200,210))
    ax[1].legend();fig.tight_layout();fig.savefig(root/'development_replay.png',dpi=160);fig.savefig(root/'development_replay.pdf');plt.close(fig)


def write_reports(root,repo):
    s=read_json(root/'summary.json');c=s['counts'];f=read_json(root/'run_identity.json')
    (root/'README_RESULTS.md').write_text(f'''# A 开发回放；B 未使用；无独立确认

状态：**{s['state']}**。purpose=development_replay；independent_confirmation=false。

本轮研究第一轮已在 A 选出的 W-I，能否在同一 A179 缓存开发池、相同初始 50 条观测及相同 20 次 qLogEI + 1 次终端 greedy 预算下，改善最佳已观测验证成绩。

## 实际结果

- 完成 {c['trajectories']}/20 轨迹、{c['fits']}/420 次拟合、{c['starts']}/1260 个冷起点。
- n70 平均配对收益：{s['n70_mean_paired_gain_pp']:.9f} pp；描述性 95% 配对 bootstrap 区间 {s['n70_descriptive_bootstrap_ci_pp']} pp。
- n71 主终点平均配对收益：{s['n71_mean_paired_gain_pp']:.9f} pp；95% 配对 bootstrap 区间 {s['n71_bootstrap_95_percent_ci_pp']} pp；严格正收益 {s['positive_n71_seeds']}/10。
- 终端 greedy 平均增量：{s['greedy_increment_mean_pp']} pp。n70 与 n71 分开报告，终端推荐不能归因于第 21 次 qLogEI。
- 初始集合已经包含池内最优：{s['initial_sets_containing_optimum']}/10；逐 seed headroom 见 paired_comparisons.csv。
- q0..20 缓存 simple regret 梯形积分/20：{s['regret_integral_mean']}。
- 工程门槛：{json.dumps(s['gate'],ensure_ascii=False)}。

## 完整性与限制

qLogEI 查询 {c['qlogei_queries']}；greedy {c['terminal_greedy_queries']}；初始逻辑观测 {c['initial_observations']}；逻辑释放 {c['logical_releases']}；唯一候选 {c['unique_candidates']}；oracle release RPC 调用 {c['oracle_release_calls']}，其中幂等重试 {c['idempotent_replays']}。
物理 A 标签文件打开 {c['physical_A_label_file_opens']} 次（包括每次加载前哈希校验及 JSON 解码；按 oracle/scorer 会话分项见 oracle_summary.json）。这不是 GNN 训练数。
失败起点 {c['failed_starts']}；收敛告警 {c['convergence_warnings']}（Adam 末 50 步 {c['adam_final50_warnings']}，LBFGS 预算 {c['lbfgs_budget_warnings']}，两项可重叠）。有限且预算完成不等于证明收敛。总拟合耗时 {s['fit_seconds']:.1f}s，采集耗时 {s['acquisition_seconds']:.1f}s。
B 标签/SE 读取=0；节点 test 读取=0；新 GNN 训练=0；KG=0。生产文件、冻结第一轮、旧 BLOCKED 任务保持不变。

事实：以上数值是固定开发池缓存回放成绩。解释：配对初始化减少初始观测差异；运行时隔离保证决策不读取未来缓存成绩。限制：第一轮方法选择及本轮评价均使用 A；bootstrap 仅描述固定池与已选方法下随机初始化的条件差异，不支持独立确认、未知架构泛化、真实在线训练收益或 test 提升。未满足门槛时，不换种子、模型或指标救场。

逐 seed 结果、曲线及其单位分别见 paired_comparisons.csv、curves.csv 与 development_replay.png。最终返回的是 71 个已观测候选中的最佳者；完整配置和配方见 returned_candidates.json。

## 恢复与证据

run_identity：`{f['run_identity']}`。
实际命令见 actual_commands.md；恢复验收见 resume_verification.json；代码、数据来源、测试见 code_diff.patch、source_manifest.json、test_results.txt。
模型、优化器、观测顺序、采样器和不可变决策封存均保留服务器。分享包省略模型/优化器大文件与 SQLite，保留审计、决策分数、观测和 SHA 清单。预算和主终点完整后本轮结束。
''')
    (root/'NEXT_CONFIRMATION_PLAN.md').write_text('''# 后续确认计划（本轮不执行）

1. 同一已选配置采用预先冻结的新 GNN seeds 重训，可检验训练随机性下的可重复性；仍不是未知架构独立确认。先冻结需复核的配置去重列表、现有配方、seed 数量、主终点与分析。预算=配置数×新 seed 数；按既有同配方单次训练实测中位数/分位数估算时间和资源，不在本轮新增训练。
2. 真正独立候选池 C 的性能结果此前不得用于方法设计、筛选或调参。先审计所有历史实验、标签访问及实际用途日志；用 exact directed typed graph 同构核查与开发架构关系。同架构新 HP、全新架构、新数据集分别对应不同问题。未知架构泛化须与历史开发架构隔离。
3. 若考虑原 768 候选剩余部分，先只用无标签来源追溯候选与全部历史使用记录。缺失访问日志或历史成员哈希时记录无法核实，不把 A/B 外候选、重新划分或改名视为天然独立。独立性无法建立时另行构建来源清楚的池。
4. 在读取 C 表现前冻结来源、候选选择规则、方法、采集预算、主终点、分析及训练配方。预算规则：候选缓存数×训练 seed 数；或方法数×搜索 seeds×每轨迹初始与新增查询数，明确共享/去重的实际训练预算。CPU GP 预算按本轮实测各 n/method 时间估计；GNN GPU 预算需实测来源支持，当前不猜固定小时数。
5. 只有开发收益与上述来源审计共同支持后，才值得考虑昂贵确认。若本轮未过门槛，先审阅证据和机制，不自动扩大训练投入。新 seed 复核不能包装成独立 C 确认。本轮不训练、不读节点 test、不启用 KG。
''')
    (root/'GPT_ANALYSIS_REQUEST.md').write_text(f'''请分析此 A-only 开发回放证据。independent_confirmation=false；B 未使用。旧 B 独立确认仍 BLOCKED。

实际状态 {s['state']}；完成 {c['trajectories']} 轨迹、{c['fits']} 拟合、{c['starts']} 起点。请以 README_RESULTS.md、protocol.json、source_manifest.json、paired_comparisons.csv、curves.csv、optimization_summary.csv、oracle_summary.json 和测试/恢复记录为证据。

1. 核查 n71 主终点与冻结门槛，不用 n70 或曲线替代失败主终点。
2. 对照 n70、n71 和 greedy 增量，区分 qLogEI 过程与最终推荐收益；关注初始 headroom 和包含池内最优的 seed。
3. 解释 10 个配对初始化 bootstrap 的固定开发池条件含义，不扩展为独立确认或未知架构泛化。
4. 核查实际预算、访问隔离、三个冷起点训练 NLL 选择、数值告警、完整性及恢复零新增证据。
5. 区分事实、解释、假说；据当前证据讨论是否值得投入新训练/真正独立 C，列出来源缺口，禁止将新 seeds 重训当作独立架构确认。
''')
    (root/'acquisition_audit.md').write_text('''# 采集实现审计

生产链：bo_phase4.optimize_acq → predictor.make_logei → surrogate.accuracy_gp.StableQLogExpectedImprovement。直接实例化同一 Stable 类；不调用会调整生产噪声下限/重拟合的 make_logei 包装器。

FrozenAdapter 使用冻结第一轮 StaticExactGP 的原模型状态。CPU float64，M×1×D，输出 posterior M×1×1；每次条件化在第一轮 min_fixed_noise=0、jitter=1e-8→1e-7→1e-6、单次 Cholesky retry policy 内完成并物化密集协方差。外层生产安全上下文未改写训练噪声。q=1 的一维 posterior 根分解使用 sqrt，不引入额外观测噪声。保留 autograd；合成测试直接对照 posterior mean/covariance 及梯度。

Sobol 512，共享每 seed/step 的 base samples，fat=true,tau_max=.01,tau_relu=1e-6,eta=.001。step=1..20。升序未观测 ID、chunk64，完整枚举而非 q=M 或连续优化。best_f 来自已观测 T 并以当前 y 标准化。未知点不读取 SE、不追加观测噪声。并列绝对容差 1e-12；greedy 在 raw accuracy 均值上使用同一容差。无 KG、decoder 或默认 SingleTaskGP。

每次决策记录全部候选分数、选中 ID、模型哈希、已观测顺序、best_f、MC seed、sampler 哈希、耗时及数值恢复事件。模型/优化器及 base samples 在服务器保留。详见测试对分块与完整枚举、潜在均值协方差、噪声契约及未来标签投毒的验证。
''')
    patch=[]
    for relative in f['code']:
        content=(repo/relative).read_text().splitlines(keepends=True)
        patch.extend(difflib.unified_diff([],content,fromfile='/dev/null',tofile='b/'+relative))
    (root/'code_diff.patch').write_text(''.join(patch))


def package(root):
    excluded={'.pt','.sqlite','.sqlite-wal','.sqlite-shm','.tmp','.tar.gz'}
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix not in excluded
        and p.name not in ('SHA256SUMS','package_sha256.txt') and not p.name.endswith('.tar.gz')]
    (root/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(root)}\n' for p in files))
    destination=root.with_name(root.name+'_evidence.tar.gz')
    with tarfile.open(destination,'w:gz') as tar:
        for p in files+[root/'SHA256SUMS']:
            tar.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    write_json(root/'package_sha256.txt',{'path':str(destination),'sha256':sha256(destination),'members':len(files)+1})
    return destination
