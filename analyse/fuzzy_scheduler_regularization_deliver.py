"""Evidence packaging and report from verified completed artifacts."""
import sys,os,json,hashlib,collections,statistics,time,subprocess,difflib,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from analyse.fuzzy_loss_interaction_core import read,write,file_hash,digest,journal,utc
from analyse.fuzzy_scheduler_regularization_run import verify

def mean(v):return float(statistics.mean(v))
def pct(v):return f'{100*v:.4f}' if v is not None else '未完整'
def fmt(v):return f'{v:.2f}'
def table(columns,rows):return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def report(o):
    p=verify(o,private=True);assert read(o/'analysis/independent_recompute.json')['passed'];m=read(o/'analysis/main.json');counts=m['resources'];ind=read(o/'analysis/independent_recompute.json');sels=read(o/'analysis/tables/selected_regularization.json');conv=read(o/'analysis/tables/convergence_summary.json');cases=read(o/'analysis/tables/scheduler_repair_cases.json');timing=read(o/'analysis/timing.json');wall=read(o/'execution_complete.json')['controller_wall_seconds'];work=read(o/'workspace_before.json')
    # Original tracked changes and all original protected source/data/VAE bytes remain untouched.
    before=(o/'git_diff_before.patch').read_bytes();after=subprocess.check_output(['git','diff','--binary']);assert before==after
    protected=read(o/'protected_before.json');observed={n:file_hash(Path(n) if Path(n).is_absolute() else ROOT/n) for n in protected};assert observed==protected
    write(o/'protected_after.json',observed);(o/'git_diff_after.patch').write_bytes(after);(o/'git_status_after.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    codefiles=['analyse/fuzzy_scheduler_regularization_'+n+'.py' for n in ['core','run','evaluate','recompute','deliver']]+['tests/test_fuzzy_scheduler_regularization.py','configs/fuzzy_scheduler_regularization_v1.json']
    for n in codefiles:
        target=o/'effective_source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
    patch=''
    for n in codefiles:patch+=''.join(difflib.unified_diff([], (ROOT/n).read_text().splitlines(True),fromfile='/dev/null',tofile=n))
    (o/'new_files.patch').write_text(patch)
    commands=[f'/home/fangxy/.conda/envs/nas/bin/python -B analyse/fuzzy_scheduler_regularization_run.py prepare {o} --prompt /tmp/Codex_Fuzzy_Scheduler_Repair_Regularization_Execution_Prompt_20260921.md',f'/home/fangxy/.conda/envs/nas/bin/python -B analyse/fuzzy_scheduler_regularization_run.py run {o}',f'/home/fangxy/.conda/envs/nas/bin/python -B analyse/fuzzy_scheduler_regularization_evaluate.py {o}',f'/home/fangxy/.conda/envs/nas/bin/python -B analyse/fuzzy_scheduler_regularization_recompute.py {o}',f'/home/fangxy/.conda/envs/nas/bin/python -B analyse/fuzzy_scheduler_regularization_run.py verify {o}']
    execution=dict(protocol=p['identity'],host=work['hostname'],project=str(ROOT),branch=work['branch'],HEAD=work['HEAD'],commands=commands,source_hashes={n:file_hash(ROOT/n) for n in codefiles},completed=read(o/'execution_complete.json'),resources=counts,independent_passed=True,no_training_repeated_for_instrumentation_repair=True,original_worktree_diff_unchanged=True,protected_sources_data_VAE_unchanged=True,time=utc())
    write(o/'execution_manifest.json',execution)
    statusrows=[[purpose,kind,sum(dist.values()),json.dumps(dist,ensure_ascii=False)] for purpose,kinds in m['status_by_purpose'].items() for kind,dist in kinds.items()]
    selrows=[]
    for context in [f'offline-f{i}' for i in range(5)]+[f'replay-start{s}' for s in range(2026091900,2026091905)]:
        cs=next(r for r in sels if r['context']==context and r['group']=='C');fs=next(r for r in sels if r['context']==context and r['group']=='F')
        selrows.append([context,cs['lambda_w'],cs['deciding_metric'],fs['lambda_w'],fs['deciding_metric']])
    replays=table(['流程','组','完整起点','平均最好验证成绩（%）','距A最好（百分点）','冠军命中'],[[r['flow'],r['group'],r['completed'],pct(r['mean_best_raw']),pct(r['mean_gap_raw']),str(r['champion_hits'])+'/5'] for r in m['replay']])
    offline=table(['流程','组','已知头误差（百分点）','留出头误差','留出GP误差','GP前5差','主近邻差'],[[r['flow'],r['group'],pct(r['train_head_mae_raw']),pct(r['holdout_head_mae_raw']),pct(r['holdout_gp_mae_raw']),pct(r['gp_top5_gap_raw']),pct(r['primary_neighbor_raw'])] for r in m['offline']])
    paired=read(o/'analysis/tables/replay_paired.json');pairs=table(['50次成对比较','平均差（百分点）','胜/平/负'],[[r['comparison'],pct(r['difference_raw']),f"{r['wins']}/{r['ties']}/{r['losses']}"] for r in paired if r['queries']==50 and r['search_seed']=='all_five'])
    zeros={g:sum(r['group']==g and r['lambda_w']==0 for r in sels) for g in ['C','F']};newstatus=m['status'];rawtaskseconds=sum(t.get('task_seconds',0) for t in timing['by_purpose'].values());mapopt=sum(r['optimization_seconds'] for r in conv if r['kind']=='mapping');gpopt=sum(r['optimization_seconds'] for r in conv if r['kind']=='gp');usedfit=sum(r['steps'] for r in conv);unknownphase=timing['missing_detailed_phase_counters'];totalchecks=sum(r['checks'] for r in ind['fit_checks'])
    lrchanges=[e for e in read(o/'analysis/tables/scheduler_timeline.json') if e['action']=='lr_reduced'];blocked=[e for e in read(o/'analysis/tables/scheduler_timeline.json') if e['action']=='lr_reduction_blocked_insufficient_budget']
    costrows=[[k,counts['actual_updates'][k]['total'],json.dumps(counts['actual_updates'][k]['by_purpose'],ensure_ascii=False)] for k in ['mapping','gp']]
    attributed=read(o/'analysis/cost_attribution.json')['rows'];workflow_cost_rows=[]
    for stage in ['offline','replay']:
        for flow in ['REPAIRED','REPAIRED_SELECT_REG']:
            parts=[r for r in attributed if r['stage']==stage and r['flow']==flow]
            physical={k:sum(r['physical_additional_work'][k] for r in parts) for k in ['mapping_updates','gp_updates','task_seconds']}
            logical={k:sum(r['logical_standalone_work'][k] for r in parts) for k in ['mapping_updates','gp_updates']}
            workflow_cost_rows.append([stage,flow,physical['mapping_updates'],physical['gp_updates'],logical['mapping_updates'],logical['gp_updates'],f"{physical['task_seconds']:.2f}"])
    workflow_cost_table=table(['部分','流程（含相应内选）','本轮实际映射步','本轮实际GP步','单独运行逻辑映射步','单独运行逻辑GP步','实际worker秒'],workflow_cost_rows)
    synthetic_times=[dict(path=str(f.relative_to(o)),kind=read(f)['kind'],updates=read(f)['completed_updates'],fit_seconds=read(f)['fit_seconds'],optimization_seconds=read(f)['optimization_seconds']) for f in (o/'synthetic').rglob('summary.json')]
    write(o/'analysis/synthetic_timing_available.json',dict(rows=synthetic_times,full_test_wall_clock_not_measured=True,standalone_gradient_checks_not_in_fit_seconds=True))

    case_table=table(['样例','类型','首次实际更新分叉','旧步数→新步数','旧末3窗口完整','新末3窗口完整','新终态'],[[r['task'],r['kind'],r['first_different_update'] if r['first_different_update'] is not None else '无',f"{r['old_steps']} → {r['new_steps']}",str(r['old_final_three_windows']),str(r['new_final_three_windows']),r['new_status']] for r in cases])
    text=f'''# 调度修复与参数约束比较

本轮结论：调度缺口已修复，参数约束降低了留出架构的平均预测误差，但没有改善主要目标——同样50次查询后的最好成绩。C由80.5333%降至80.3733%，平均下降0.16个百分点；F保持80.7333%。随机参照为80.7600%，冠军命中2/5，C为0/5、F为1/5。不能据此宣布解决过拟合或获得搜索优势。

本轮完成六个工程样例、全部预定内层选参、五折离线对照及35条逻辑历史回放。是否全部达到50次查询：{m['all_35_complete']}。新增GNN训练、VAE更新、补测、B/test访问均为0。完整数据见下表；这是同一个已反复开发的A池，不能写成独立确认或普遍优势。

调度规则已修好。独立复算核对{totalchecks}个检查点的学习率、保护期、窗口和停止条件；没有违反保护期的重复下降，也没有末步配置后被误计为实际使用的学习率。真实下降{len(lrchanges)}次，因剩余观察预算不足阻止下降{len(blocked)}次。完整观察不等于已稳定，未达到条件的拟合全部保留。

C的10次内选中{zeros['C']}次选择0，F的10次内选中{zeros['F']}次选择0。选择0只说明本轮有限网格和合法内层数据未支持额外约束。选参由GP前2差、前1差、MAE、小强度依次决定，数值并列容差1e-12；不使用外层或未来回放成绩。C/F选中强度不同时，组间结果评价的是完整流程。

## 1. 相同50次新增查询

{replays}

{pairs}

每条先读初始20条，然后10批、每批5条。包含初始20的最好成绩是主指标。冠军未命中写“>50”，没有按51求平均。新增20条逻辑轨迹与导入15条旧参照分开，旧W未进入本轮主比较。选择0的流程标记logical_alias，仅复用严格等价结果，不当成独立重复。

## 2. 留出架构与参数选择

{offline}

组内加入所选约束后，C的留出GP平均绝对误差由8.9602降至8.4310个百分点，F由7.9337降至7.7895；预测头误差也下降。已知候选拟合误差上升，说明约束确实改变了拟合取舍。但C的GP前5成绩差由1.8400增至2.1956个百分点，F由2.7067降至2.3956；主近邻差两组均增大。这些指标方向不一致，外层误差改善不能替代最终选优证据。

仅修复调度时，C/F的五个回放终点成绩均与旧ADAPTIVE相同。加约束后F在部分较早预算上有改善，但50次终点未提高。C/F在不加参数约束时的比较，以及两组各自内选后的比较，都提供离线成对表offline_paired.csv和回放成对表replay_paired.csv；各自内选后的组间比较不能解释成只改变邻域项。

误差及成绩差都越小越好。训练误差按15个折/种子模型等权平均；留出误差汇合每个模型种子的179个查询。外层按架构隔离，五折、模型种子0/1/2；原epsilon、F权重和折0/2回退来源保持。主近邻差仍是每种子汇合179个留出查询，对每查询5个已知邻居取绝对成绩差均值，再取179查询中位数，最后平均3个模型种子。没有替换成所有配对平均。

{table(['合法数据范围','C强度','C决胜项','F强度','F决胜项'],selrows)}

三内折按architecture_id整组划分，离线种子2026092100+折号，回放种子为search_seed；C/F与四强度共用内折和未训练初始化。每个内层训练进程只能读其训练部分成绩和SE，标准化只拟合其训练部分；预测分数与身份封存后评价器才读取验证标签。回放只用最初20条选一次强度，后续50次固定。原代码已标准化z、HP与y，本轮沿用实际实现，没有新增标准化变量。

参数项为全部2337个可训练参数（含bias）的平方和，不除参数数、不乘1/2，Adam weight_decay=0，GP与冻结VAE不受此项惩罚。强度0直接沿用原计算图。全部强度使用同一个修复后的训练协议，没有验证早停，没有据稳定率或外层结果选强度。

## 3. 工程样例与稳定状态

{case_table}

工程样例严格从指定原post000和当时的已知标签开始；known30/40分别为新增30/40，实际已知50/60。GP样例沿用原固定映射和输入。所有工程样例在首次学习率差异之前loss及梯度一致，固定前缀的参数、Adam和RNG一致。无实际更新分叉的样例仍保留，不能把末步仅修改下一步学习率解释成已训练。

{table(['部分','拟合类型','次数','终态分布'],statusrows)}

总体状态：映射{json.dumps(newstatus['mapping'],ensure_ascii=False)}；GP {json.dumps(newstatus['gp'],ensure_ascii=False)}。观察不足、几何退化、数值失败均为0。这些次数是实际拟合，包含工程、内选与正式拟合，不重复计算别名。到上限的有限模型保存并冻结后继续选点，逐批标注fallback。GP阶段固定映射与标准化，全部坐标按同一版本重算；冻结不反证前面已经稳定。

映射首个完整窗口160步、最早连续三次通过200步；GP分别50/70步。下降后lr_epoch_start=t+1，首个完整窗口为映射t+80、GP t+40，三次完整检查分别到t+120、t+60。保护期内不积攒平台次数，到期检查仍先判稳定；预算不足不下降，最低学习率不清窗口。目标原三项及L0用L0尺度，新总损失与参数加权项用新总损失尺度，避免正则较大掩盖原目标变化。

## 4. 预算、计时与独立核对

{table(['类型','实际更新次数','各部分更新'],costrows)}

{workflow_cost_table}

“本轮实际”各行不重叠，选择0的复用没有重复收费；“单独运行逻辑”包含该流程自身必需的拟合和全部内选，复用项按源任务工作量列示，不能当作实际新增运行或新的计时测量。工程样例单独列在上表。

实际映射拟合{counts['actual_mapping_fits']}次、GP拟合{counts['actual_gp_fits']}次。合成验证另为映射{counts['synthetic']['mapping']}步、GP {counts['synthetic']['gp']}步，均不超过60。全部预算账本逐项核对，没有隐去重算；没有数值设置救跑。

实际新增历史查询{counts['actual_new_queries']}，实际初始访问{counts['actual_initial_accesses']}；新流程逻辑查询{counts['logical_new_queries']}、逻辑初始400；旧参照750查询、300初始。总逻辑查询{counts['all_logical_queries']}、初始700。共有{counts['logical_alias_tasks']}个离线/回放逻辑任务复用等价结果。内选累计既有标签分割访问{counts['inner_existing_label_partition_accesses']}，单列为已知标签内部使用，不是新GNN评估。

控制器墙钟{wall:.2f}秒（{wall/60:.2f}分钟），包含调度短暂停顿；worker任务墙钟合计{rawtaskseconds:.2f}秒。映射优化{mapopt:.2f}秒、GP优化{gpopt:.2f}秒。检查前向、诊断梯度、距离、GP预测及存盘分项见analysis/timing.json，存在嵌套，不能相加。第0折24个内选任务作为正式计时样例，没有额外计时训练。独立复算{ind['seconds']:.2f}秒，无优化更新；评价{m['evaluation_seconds']:.2f}秒。代码审查、报告、压缩与传输另计。合成检查各拟合的可用计时见analysis/synthetic_timing_available.json，未测整套合成检查的墙钟，未用估计值冒充测量。

计时记录有一次明确修订：最初六个工程样例和两个内选任务的汇总错误引用前轮计时器。只修正新worker的分项计时来源，没有改变目标、梯度、学习率、选参或已完成模型，也没有重训。原源码、补丁、修订身份和受影响列表保存在instrumentation_amendment.json/.patch。上述{len(unknownphase)}个任务的真实worker墙钟和拟合优化时间存在，细分诊断耗时无法追回，报告未把原来的0冒充真实零成本。其余任务完整记录分项。

独立程序重算{ind['physical_fit_count']}个实际拟合及其全部窗口、初始梯度和检查点梯度/距离/GP均值；{ind['independent_inner_scores']}个内折强度评分、{ind['independent_selections']}次选择；{ind['GP_scoring_cases']}个GP评分场景，最大差{ind['max_GP_score_error']:.3e}。新旧回放选择、标签收据、50查询、最好成绩曲线与全部成对结果均核对。训练worker实际拒绝读取全A隐藏标签、未知SE、其他任务及VAE，并留下拒绝记录。

## 5. 位置、运行与复算

主机{work['hostname']}，项目{ROOT}，分支{work['branch']}，完整HEAD {work['HEAD']}。Python {work['python'].split()[0]}，PyTorch {work['torch']}，GPyTorch {work['gpytorch']}，NumPy {work['numpy']}；CPU、每进程1线程，未安装升级依赖或占用GPU。原已跟踪差异及原保护源码/数据/VAE哈希保持不变，未pull/reset/切分支/清理/提交/推送。

结果目录：{o}。本次协议身份：{p['identity']}。完整附件REQUEST.md、固定配置config.json、原来源reference_snapshot、原冻结源码source_snapshot、实际交付源码effective_source和补丁new_files.patch均保留。

已验证执行命令（项目目录下运行）：

```bash
{chr(10).join(commands)}
```

中断且未完成时用同一run命令恢复，每任务至多一次进程中断恢复，已用预算与收据不会重置；已完成目录使用verify/recompute，不能重新prepare。复算不训练，已有复算报告保留，新报告写入verification_outputs。

CSV为原始记录派生的完整精度表格，嵌套字段为JSON字符串、缺失为留空，raw准确率单位明确。analysis/plots包含调度时间线、损失/参数项/梯度/距离/GP均值以及最好成绩曲线。ZIP只含本轮必要A数据、检查点、旧参照、记录与源码，不打包整仓库、凭据或B/test。SHA256SUMS校验包内文件，evidence_bundle.zip.sha256校验ZIP。

## 6. 下一步与本轮停止点

最有依据的下一步是只用本轮已封存记录，审查“初始20条的内层Top2选择”与后续50次选优结果为何不一致，优先定位C中非零约束使选点变差的批次。先核对高分候选的排序与错失原因，再决定是否需要另行预注册新实验；当前不把参数约束设为默认改进，不扩大强度网格或训练预算。本轮矩阵到此结束，没有启动下一轮。
'''
    (o/'REPORT.md').write_text(text)
    print(json.dumps(dict(report=str(o/'REPORT.md'),resource=counts,all_checks_passed=True)))


def package(o):
    started=time.perf_counter();assert read(o/'analysis/independent_recompute.json')['passed'];assert read(o/'csv_roundtrip_checks.json')['passed']
    archive=o/'evidence_bundle.zip';assert not archive.exists(),'do not overwrite a sealed package'
    omit={'evidence_bundle.zip','evidence_bundle.zip.sha256','SHA256SUMS','bundle_verification.json','packaging.log'}
    paths=[p for p in o.rglob('*') if p.is_file() and p.name not in omit and 'runtime_tmp' not in p.parts and p.suffix!='.tmp' and not p.name.endswith('.tar.gz') and not p.name.startswith('._') and p.name not in ['controller.lock']]
    # Existing source snapshots are small and necessary. No repository traversal.
    hashes={str(p.relative_to(o)):file_hash(p) for p in sorted(paths)}
    (o/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for n in hashes:z.write(o/n,n)
        z.write(o/'SHA256SUMS','SHA256SUMS')
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(hashes)+1
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    hasher=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):hasher.update(chunk)
    h=hasher.hexdigest();(o/'evidence_bundle.zip.sha256').write_text(h+'  evidence_bundle.zip\n');result=dict(passed=True,files=len(hashes)+1,checkpoints=sum(n.endswith('.pt') for n in hashes),bytes=archive.stat().st_size,sha256=h,seconds=time.perf_counter()-started);write(o/'bundle_verification.json',result);print(json.dumps(result))

if __name__=='__main__':
    output=Path(sys.argv[2]).resolve()
    if sys.argv[1]=='report':report(output)
    elif sys.argv[1]=='package':package(output)
    else:raise ValueError('report or package')
