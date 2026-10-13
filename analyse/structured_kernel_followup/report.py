import csv
import difflib
import json
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from .common import *


def rows(root,name):
    p=root/name
    return list(csv.DictReader(p.open())) if p.exists() else []


def table(data,columns):
    if not data:return '暂无有效结果。\n'
    def fmt(v):
        if isinstance(v,float):return f'{v:.6f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(c,'')) for c in columns)+' |' for r in data)+'\n'


def write(root,name,text):
    (root/name).write_text(text,encoding='utf-8')


def commands(root):
    entries=[json.loads(l) for l in (root/'commands.jsonl').read_text().splitlines()]
    text='# 实际执行命令\n\n服务器 ismc36-ts，项目 /home/fangxy/D-VAE。以下来自执行入口实时记录；训练任务各为单独进程、单 GPU worker。\n\n'
    for e in entries:
        text+=f"{e['time']}\n\n```bash\ncd {shlex.quote(e['cwd'])}\n{shlex.join(e['argv'])}\n```\n\n"
    text+='环境变量及完整时间见 commands.jsonl；预执行 pytest 的真实命令、完整日志和 JUnit 结果见 acceptance/。\n'
    write(root,'actual_commands.md',text)


def report(root,repo):
    state=read_json(root/'module_status.json');summaries={m:read_json(root/f'{m}_summary.json') if (root/f'{m}_summary.json').exists() else state['modules'][m] for m in ('M1','M2','M3','M4')}
    m1,m2,m3,m4=[summaries[m] for m in ('M1','M2','M3','M4')]
    trace=rows(root,'champion_trace.csv');comp=rows(root,'kernel_mean_contributions.csv');points=rows(root,'training_point_contributions.csv')
    selected=[r for r in trace if r['method']=='W-I' and int(r['n']) in (50,60,69,70)]
    text='# M1：冻结 GP 的冠军低估诊断\n\n本模块只重建既有后验，没有优化或改选点。真值只用于事后诊断。\n\n'
    text+=f"验证模型 {m1.get('verified_models',0)}/86，均值分解 {m1.get('decompositions',0)}/12；状态 {m1['state']}。训练 NLL 与 n70 剩余候选均值分别按原尺度绝对误差 1e-8 验收，逐模型证据见 posterior_reconstruction_validation.json。\n\n"
    text+=table(selected,['seed','n','champion_observed','champion_mu_percent','latent_sd_pp','prospective_sd_pp','bias_pp','remaining_mean_rank','sealed_qlogei_rank','sealed_greedy_mean_rank'])
    text+='\n直接现象：\n\n'
    facts=[]
    for s in M1_SEEDS:
        r=[x for x in trace if int(x['seed'])==s and x['method']=='W-I'];by={int(x['n']):x for x in r}
        if 50 not in by or 70 not in by:continue
        a,b=by[50],by[70];delta=float(b['champion_mu_percent'])-float(a['champion_mu_percent'])
        obs=read_json(repo/REPLAY/'trajectories'/f'{s}_W-I'/'observations.json')
        slots=[i-49 for i,o in enumerate(obs) if o['candidate_id']==CHAMPION]
        hit='初始化已包含' if slots and slots[0]<=0 else (f'第 {slots[0]} 次查询命中' if slots else '71 个已观测点仍未命中')
        last=by.get(69,{})
        text+=f"- {s}：n50 均值 {float(a['champion_mu_percent']):.6f}%，n70 {float(b['champion_mu_percent']):.6f}%，变化 {delta:+.6f} pp；{hit}。最后 qLogEI 排名 {last.get('sealed_qlogei_rank') or 'NA'}（n69），终端均值排名 {b['remaining_mean_rank'] or 'NA'}（n70）。\n"
        facts.append({'seed':s,'initial_mu_percent':float(a['champion_mu_percent']),'n70_mu_percent':float(b['champion_mu_percent']),'change_pp':delta,'champion_query':slots[0] if slots else None,'last_qlogei_rank':last.get('sealed_qlogei_rank'),'n70_mean_rank':b['remaining_mean_rank']})
    text+='\nRAW16 相同初始历史对照：\n\n'+table([r for r in trace if r['method']=='RAW16'],['seed','champion_mu_percent','latent_sd_pp','prospective_sd_pp','bias_pp','remaining_mean_rank','sealed_qlogei_rank'])
    text+='\n冻结参数下的代数均值分解（基准为百分数，各项为 pp）：\n\n'+table(comp,['seed','n','baseline_percent','architecture_contribution_pp','hp_contribution_pp','interaction_contribution_pp','sum_percent','raw_abs_error','actual_diagonal_jitter'])
    text+='\n候选贡献使用原候选坐标下 β=C⁻¹k_sᵀ 与残差的乘积。所有正负贡献、每分量最大/最小各5个候选、原观测值、相似度和权重均在 training_point_contributions.csv，未做贡献百分比或因果归因。\n\n失败轨迹 n70 每个分量最强负贡献：\n\n'
    negative=[r for r in points if int(r['seed']) in M1_SEEDS[:2] and int(r['n'])==70 and r['top5_negative_rank']=='1']
    text+=table(negative,['seed','component','candidate_id','observed_validation_percent','contribution_pp','architecture_similarity','hp_similarity','weight_beta'])
    text+='\n解释边界：后验均值由当前观测、标准化、长度尺度、常量均值及三项核共同确定。上述表直接定位了哪些核关联和残差组合压低冠军；低均值与不确定性共同影响实际封存的 qLogEI 排名。成功轨迹命中后，冠军自己的观测会改变后验。这里没有干预核分量、交换观测历史或重跑搜索，因此不能证明某个分量是唯一失败原因；成功/失败历史的差异仍是关联证据。主不确定性始终来自完整 GP，未加总分量条件方差。\n'
    write(root,'M1_diagnosis.md',text);write_json(root/'M1_direct_findings.json',facts)
    text='# M2：同预算纯随机缓存参考\n\nA 上新增开发分析，非独立确认。随机顺序先于标签释放冻结，10 组均复用原初始化；两组初始化已有冠军也保留。W-I/RAW16 为旧20 qLogEI+1 greedy完整策略，Random为21次无放回随机。\n\n'
    text+=f"状态 {m2['state']}；{m2.get('trajectories',0)}/10 轨迹，新增查询 {m2.get('new_random_queries',0)}/210。逻辑观测 {m2.get('logical_releases',0)}，唯一候选 {m2.get('unique_candidates',0)}，物理 A 标签文件打开 {m2.get('physical_A_label_file_opens',0)} 次，oracle调用 {m2.get('oracle_calls',0)}，幂等重试 {m2.get('idempotent_replays',0)}。不代表新 GNN 训练。\n\n"
    text+=table([{'comparison':k,**v} for k,v in m2.get('comparisons',{}).items()],['comparison','mean_pp','CI95_pp','wins','ties','losses'])
    text+='\n全部10组共同重抽索引2000次，seed=2026100531；区间是给定A及旧缓存成绩的配对开发分析条件区间。主终点WI−Random n71；不设新门槛、不增加随机重复。\n\n'
    text+=table(rows(root,'random_paired_comparisons.csv'),['seed','initial_best_pp','initial_contains_champion','WI_n70_pp','RAW16_n70_pp','Random_n70_pp','WI_n71_pp','RAW16_n71_pp','Random_n71_pp','WI_minus_Random_n71_pp','RAW16_minus_Random_n71_pp'])
    text+=f"\n池内冠军命中数/10：{m2.get('champion_hits')}。0..20缓存regret梯形积分/20的组均值：{m2.get('regret_integral_mean')}。第21槽在图中另标，Random没有免费模型推荐。\n"
    write(root,'M2_random_reference.md',text)
    text='# M3：固定返回配置的训练种子复核\n\n只复核原20条返回决定对应的5个配置。配置×training seeds3/4/5共15个正式任务，未重新搜索、增加配置或训练种子。\n\n'
    text+=f"状态 {m3['state']}；有效任务 {m3.get('valid_tasks',0)}/15，训练尝试 {m3.get('training_attempts',0)}，同任务恢复尝试 {m3.get('same_task_resumed_attempts',0)}。实际训练累计 {m3.get('total_measured_training_seconds',0):.3f} 秒。\n\n"
    text+='原 DynamicGNN + Adam + cosine，Cora/public/NormalizeFeatures/full-batch/global4/No-LN，float32，300/80；val严格改善才更新best，tie计入patience。原函数返回最大val但不恢复best；本入口保存并恢复对应best状态，重新验证同一val，不改变选择规则。逐轮数值与原函数、模拟中断恢复的合成验收见 acceptance。\n\n'
    text+='数据加载器载入全图标签并核对数据身份。训练视图保留train/val标签，其他标签替换为-100且不提供test_mask；全图特征和边按原transductive配方使用。test_metric_computations=0，test_score_reads=0。不能把全图标签加载说成“从未载入test标签数组”。\n\n'
    text+=table(rows(root,'fresh_configuration_summary.csv'),['candidate_id','seed3_percent','seed4_percent','seed5_percent','mean_percent','SE_pp'])
    text+='\nSE=std(ddof=1)/sqrt(3)，单位pp。每训练seed按原全部10个搜索起点的固定返回映射求D_r：\n\n'+table(m3.get('D_by_training_seed',[]),['training_seed','D_pp'])
    if m3.get('state')=='COMPLETE':text+=f"\n三个D平均 {m3['D_mean_pp']:+.6f} pp，范围 {m3['D_range_pp']}，旧缓存0/1/2平均差 +0.600000 pp。\n"
    text+='\n03/06/08共享同配置同测量，其差严格为0；05原来不同配置均值恰同，方向可变。fresh_return_pair_results.csv的30行是共享15个真实测量的映射，不是30个独立配对样本。仅三个training seeds，报告逐seed及描述性平均/范围，不用旧起点bootstrap代替新训练不确定性。\n\n'
    if (root/'historical_same_seed_audit.json').exists():text+=f"历史用途核验：{read_json(root/'historical_same_seed_audit.json')['count']}/15相同配置/配方/seed任务已存在于完成的历史stabilization。本轮仍是真实重新训练，并非从旧成绩复制；不能称这些是从未出现的独立测量。未读取历史同seed成绩。\n\n"
    text+='这些结果仅支持固定已选配置的可重复性判断，不是新seed搜索复验；不能据5个新成绩计算A的新冠军/regret/命中率，也不能和M2拼接成“新seed下W-I超过Random”。\n'
    if m3.get('remaining_task_ids'):text+='\n未完成任务及实际错误：\n\n```json\n'+json.dumps({'remaining':m3['remaining_task_ids'],'errors':m3['errors']},ensure_ascii=False,indent=2)+'\n```\n'
    write(root,'M3_selected_config_reproducibility.md',text)
    text='# M4：独立C来源审计\n\n只读原始无标签配置和历史身份/用途元数据，没有C/B性能揭示或新训练。\n\n'
    text+=f"审计状态 {m4['state']}；科学结论 {m4.get('scientific_state','未完成')}。原候选 {m4.get('candidate_count',0)}，架构 {m4.get('architecture_count',0)}；分类 {m4.get('classifications')}；配置完整 {m4.get('configuration_completeness',0)}。可核实合格候选 {m4.get('eligible_candidates_with_evidence',0)}，合格未知架构 {m4.get('eligible_unknown_architectures',0)}。\n\n"
    text+='192个原架构逐一经真实有向、带类型图精确同构核验，都与已确认开发用途架构重叠；哈希只用于加速候选匹配，不替代精确判断。336配置有明确开发成员证据；其余432配置的完整历史性能用途未能核实，标PROVENANCE_UNRESOLVED，不能把未知当未用。即使同架构换HP，也不能回答未知架构泛化。\n\n'
    text+='historical_use_registry.json列出已核实的原始开发、第一轮方法选择、第二轮开发回放、历史目录清单与证据hash；这不是所有历史文件的穷尽使用证明。无标签结构或VAE无监督暴露本身不等同于性能泄漏。未为补历史而打开混合seed_results或其他B/C成绩缓存。\n'
    write(root,'C_source_audit.md',text)
    benchmark=read_json(root/'first_formal_task_benchmark.json') if (root/'first_formal_task_benchmark.json').exists() else {}
    nextplan='# 下一轮独立确认准备（本轮不执行）\n\n当前原768中没有可用于未知架构确认的已证实独立池。先建立候选来源与历史用途证据，不能从旧池改名、重切分或读取成绩挑“独立”候选。\n\n建议先生成并冻结新的无标签候选、完整图/HP身份与入选规则；对全部可核实开发架构做精确有向带类型同构排除，未解决来源不纳入。性能揭示前锁定W-I与RAW16、当前核和优化预算、WL对新图的编码规则、scaler训练边界及qLogEI接口，并明确任何无标签池暴露。\n\n若沿用当前策略，计划预先冻结同初始化50、20qLogEI+1终端greedy、10组配对和所有随机种子。主终点为n71返回配置验证成绩的配对差；辅助终点/区间方法/停止规则必须在C标签取得前预注册。科学设置变更需新计划，不能看C表现后调参反复确认。仍用Cora/public/NormalizeFeatures/global4/No-LN/300/80；这检验未知架构，不自动变成跨数据集或节点test确认。\n\n预算公式：对去重后U个需要揭示的完整配置、R个预注册GNN训练seed，正式训练任务为U×R；若先取得完整缓存则U=|C|。模型拟合预算为方法数×搜索组数×每组拟合状态数×每拟合优化起点数；如复用本轮回放结构则2×10×21×3=1260优化起点。两方法/不同组共享相同配置评估时必须去重。单GPU串行时间估算为Σ_config,seed实测或可比估计耗时，另计代理与I/O；不以逻辑查询次数冒充训练次数。\n\n'
    nextplan+=f"本轮第一个正式任务的实测证据：{benchmark}。这是该指定配置的设备/停止epoch/耗时，不保证新C不同架构同耗时；不杜撰额外GPU或固定总小时数。来源问题解决之前不启动C标签获取/训练，本报告未安排购买资源。\n"
    write(root,'NEXT_INDEPENDENT_CONFIRMATION_PLAN.md',nextplan)
    checks={k:v for k,v in read_json(root/'run_identity.json')['sources'].items() if k in ('bo_phase4.py','eval_utils.py')}
    baseline=read_json(root/'baseline.json')
    # Verify all frozen sources again, including existing user-edited evaluator.
    verify(root,repo)
    write_json(root/'final_source_audit.json',{'frozen_sources_unchanged':True,'user_baseline':baseline,'source_checks':checks,
        'old_replay_state':read_json(repo/REPLAY/'summary.json')['state'],'old_confirmation_state':read_json(repo/BLOCKED/'status.json')['state'],'time':now()})
    count={'old_GP_refits':0,'M1_verified_states':m1.get('verified_models',0),'M1_expected_states':86,
        'M2_random_trajectories':m2.get('trajectories',0),'M2_new_queries':m2.get('new_random_queries',0),
        'M3_valid_tasks':m3.get('valid_tasks',0),'M3_unique_configurations':m3.get('unique_configurations',0),
        'M3_training_attempts':m3.get('training_attempts',0),'B_performance_reads':0,'C_performance_reveals':0,
        'test_metric_computations':0,'test_score_reads':0,'KG_runs':0,'full_graph_label_loader_sessions':m3.get('training_attempts',0)}
    write_json(root/'summary.json',{'purpose':read_json(root/'protocol.json')['purpose'],'independent_confirmation':False,'modules':summaries,'counts':count})
    primary=m2.get('comparisons',{}).get('WI_minus_Random_n71',{})
    md='# 第三轮：开发诊断与固定返回配置复核\n\n**不是独立C确认。旧DEVELOPMENT_REPLAY_NO_STABLE_GAIN和旧B确认BLOCKED保持原样。**\n\n'
    md+='目的：解释已有冠军低估，增加同预算随机缓存参考，复核五个既定配置的训练seed变化，并核实独立C的来源。四模块独立执行，不以收益方向作为完成门槛。\n\n'
    md+=table([{'module':m,'state':s['state']} for m,s in summaries.items()],['module','state'])
    md+=f"\nM1：{m1.get('verified_models',0)}/86既有后验通过复核。失败历史和成功历史的n50→n70、实际qLogEI排名以及12组代数分解见 [M1报告](M1_diagnosis.md)。\n\n"
    for f in facts:
        if f['seed'] in M1_SEEDS[:2]:md+=f"- {f['seed']}：冠军预测从 {f['initial_mu_percent']:.3f}% 到 {f['n70_mu_percent']:.3f}%，终端均值排名 {f['n70_mean_rank']}，最后qLogEI排名 {f['last_qlogei_rank']}。\n"
    md+=f"\nM2：完整W-I策略减同预算Random的n71平均差为 {primary.get('mean_pp')} pp，配对条件95%区间 {primary.get('CI95_pp')}，胜/平/负 {primary.get('wins')}/{primary.get('ties')}/{primary.get('losses')}。这是旧A缓存上的开发比较，不代表新训练收益。\n\n"
    md+=f"M3：实际有效任务 {m3.get('valid_tasks',0)}/15；D_3/D_4/D_5={m3.get('D_by_training_seed')}，平均 {m3.get('D_mean_pp')} pp，范围 {m3.get('D_range_pp')}，对照旧+0.600 pp。这15个配置×seed测量通过固定映射共享，不能把30映射行当独立样本。历史12/15同配置同seed任务已存在，本轮是真实重训，非首次独立样本。\n\n"
    md+=f"M4：{m4.get('scientific_state')}，合格未知架构为0。原192架构全部已有精确开发架构匹配；432配置仍有完整使用历史缺口，不能推断未用过。\n\n"
    md+='直接证据是封存后验、随机缓存比较、15次新运行的validation结果和无标签来源关系。代数负贡献支持关联解释，不是核失败因果证明；仅3个新训练seed且固定了原返回配置，未复验新seed下整个搜索。是否值得独立实验，应先解决真正新架构的来源和历史隔离，再按 [下一轮准备](NEXT_INDEPENDENT_CONFIRMATION_PLAN.md) 预注册；本轮没有启动C。\n\n'
    md+='实际计数：\n\n'+table([{'counter':k,'value':v} for k,v in count.items()],['counter','value'])
    md+='\n全图标签由原Cora加载器载入以核对身份，训练视图只保留train/val标签；节点test评分和score读取均为0。模型、optimizer、RNG、恢复状态留在服务器 fresh_training/。资源、错误/重试、测试及恢复证据均在目录内。\n\n'
    md+=f"恢复命令（已完成任务按哈希校验后跳过）：\n\n```bash\nssh ismc36-ts\ncd /home/fangxy/D-VAE\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/fangxy/.conda/envs/nas/bin/python -B -m analyse.structured_kernel_followup run --run-dir {root.relative_to(repo)} --resume\n```\n\n"
    md+='交付索引：[协议](protocol.json)、[M1](M1_diagnosis.md)、[M2](M2_random_reference.md)、[M3](M3_selected_config_reproducibility.md)、[C审计](C_source_audit.md)、[实际命令](actual_commands.md)、[代码diff](code_diff.patch)、[复核请求](GPT_ANALYSIS_REQUEST.md)、[恢复验收](resume_verification.json)。\n'
    write(root,'README_RESULTS.md',md)
    request='# GPT分析请求\n\n请完整阅读本包README_RESULTS、protocol、source_manifest、run_identity及四模块报告，按以下边界独立评估，而不是寻找通过旧门槛的解释。\n\n1. M1的86模型NLL/后验误差是否合格？12组分量和、C⁻¹k候选贡献是否正确？失败201/205的初始低估、后续变化、不确定性和实际qLogEI/greedy排名支持什么直接结论？哪些因果解释尚不成立？\n2. M2的21Random与完整20qLogEI+1greedy是否同预算同初始化？十组共享2000bootstrap索引是否正确？报告主n71差值、条件区间、胜平负及冠军命中率；不要重命名终点或剔除初始化冠军。\n3. M3原配方是否忠实、validation-only边界是否成立？根据15任务的真实3/4/5结果计算每seed D及平均范围。原03/06/08应严格0，05可改变方向。评价旧+.6pp是否保留，同时承认15中12同seed历史任务存在、仅3个训练seed、30映射行非独立。不得推断整个A新冠军或新seed搜索收益，也不得与M2合成为新seed下W-I超过Random。\n4. M4是否充分区分明确开发使用、无标签暴露和历史未解决？原192架构已有开发匹配是否排除未知架构C？指出建立新C必须补的来源，不把未知当未用。\n5. 核对零旧GP重训、86/10/210/15/5预算、零B/C性能/节点test评分/KG、测试/恢复/源码保护和完整交付。区分事实、解释、假说；本轮purpose为开发诊断和固定配置可重复性，independent_confirmation=false。\n\n请给出科学与工程问题、证据路径及严重程度，建议是否值得在真正新C上独立确认。不得要求看C成绩后再改方法、增种子救显著性或重切旧B伪造独立性。\n'
    write(root,'GPT_ANALYSIS_REQUEST.md',request)
    code=read_json(root/'run_identity.json')['code'];patch=[]
    for rel in code:
        lines=(repo/rel).read_text().splitlines(keepends=True)
        patch.extend(difflib.unified_diff([],lines,fromfile='/dev/null',tofile='b/'+rel))
        dest=root/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/rel,dest)
    write(root,'code_diff.patch',''.join(patch));commands(root)


def package(root,repo):
    verify(root,repo);commands(root)
    manifest={}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix in ('.pt','.sqlite','.tmp') or p.name in ('SHA256SUMS','PACKAGE_MANIFEST.json','controller.lock'):continue
        manifest[str(p.relative_to(root))]=sha256(p)
    write_json(root/'PACKAGE_MANIFEST.json',{'files':manifest,'omitted':'Large model/optimizer/RNG checkpoints and oracle SQLite stay on server; all selected results, curves, ledger and checkpoint hashes included','root':str(root)})
    manifest['PACKAGE_MANIFEST.json']=sha256(root/'PACKAGE_MANIFEST.json')
    write(root,'SHA256SUMS',''.join(f'{h}  {p}\n' for p,h in manifest.items()))
    archive=root.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for rel in [*manifest,'SHA256SUMS']:tar.add(root/rel,arcname=root.name+'/'+rel,recursive=False)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha256(archive)+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'sha256':sha256(archive),'bytes':archive.stat().st_size,'files':len(manifest)+1}),flush=True)
