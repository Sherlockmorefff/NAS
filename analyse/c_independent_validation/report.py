import shutil,subprocess,tarfile
from .common import *

def report(root,repo):
 from analyse.c_provenance_evidence_closure.projector import Projector
 class CostProjection(Projector):
  def field_value(self,key,parent):
   if key=='elapsed_seconds':return self.value('literal',key)
   self.skip();return None
 status=read(root/'status.json');p1=read(root/'P1_summary.json');U=p1['qualified_configurations'];cost={'GP_fits':0,'GP_starts':0,'GP_seconds':0.,'GNN_complete_tasks':0,'GNN_task_seconds':0.,'GNN_attempts':0,'test_evaluations':0}
 for folder in [root/'A_only_models',root/'trajectories']:
  for p in folder.rglob('fit.json') if folder.exists() else []:
   v=read(p);cost['GP_fits']+=1;cost['GP_starts']+=len(v['starts']);cost['GP_seconds']+=v['fit_seconds']
 cost.update({'GP_attempted_starts':0,'GP_failed_starts':0,'GP_resume_events':0,'GP_logged_Adam_updates':0,'GP_logged_LBFGS_closures':0,'GP_start_recorded_seconds':0.,'acquisition_seconds':0.})
 for folder in [root/'A_only_models',root/'trajectories']:
  if not folder.exists():continue
  for p in folder.rglob('start*.jsonl'):
   cost['GP_attempted_starts']+=1
   for line in p.read_text().splitlines():
    event=json.loads(line)
    cost['GP_resume_events']+=int(event.get('event')=='resume')
    cost['GP_logged_Adam_updates']+=int(event.get('stage')=='adam')
    cost['GP_logged_LBFGS_closures']+=int(event.get('stage')=='lbfgs')
  for p in folder.rglob('start*.json'):
   v=read(p);cost['GP_failed_starts']+=int(not v['success']);cost['GP_start_recorded_seconds']+=v['elapsed_seconds']
  for p in folder.glob('*/decisions/*.json'):
   cost['acquisition_seconds']+=read(p).get('acquisition_seconds',0.)
 cost['interrupted_uncheckpointed_time']='Not reconstructible exactly; append-only optimizer/epoch events retain recomputed work; reported saved durations are lower bounds if interrupted.'
 for p in (root/'training').glob('*/result.json') if (root/'training').exists() else []:
  with p.open('rb') as f:v=CostProjection(f).all()
  cost['GNN_complete_tasks']+=1;cost['GNN_task_seconds']+=v['elapsed_seconds']
 attempts=root/'training_attempts.jsonl'
 if attempts.exists():cost['GNN_attempts']=len(attempts.read_text().splitlines())
 for p in (root/'test_evaluations').glob('*.json') if (root/'test_evaluations').exists() else []:cost['test_evaluations']+=read(p)['test_evaluations']
 cost['GNN_repeated_attempts']=max(0,cost['GNN_attempts']-cost['GNN_complete_tasks'])
 cost.update({'logical_releases':read(root/'oracle_summary.json')['logical_releases'] if (root/'oracle_summary.json').exists() else 0,
  'nominal_budget_per_trajectory':'71 configurations × 3 training seeds','full_cache_budget':U*3,'not_online_end_to_end_acceleration':True})
 write(root/'actual_costs.json',cost)
 lines=['# C 资格修正与独立验证','',f'实际阶段：{json.dumps({k:status[k] for k in ("P0","P1","P2","P3","P4","P5","P6")},ensure_ascii=False)}。',
  f'P1：合格 {U//4} 架构 / {U} 配置；疑似重合的一图四配置按来源理由保守排除，未宣称已确认泄漏。',
  '本轮通过受控身份投影修正旧访问限制。273 个文件的身份字段已提取；性能值解码为 0。161 项旧覆盖义务已逐来源处理，细节见 evidence_delta 与成员证明。',
  '四份汇总 registry 的 1,667 个不透明旧身份由其实际绑定的 144 个历史文件完整结构超集覆盖；不要求逐身份恢复旧图，也不假定旧字符串采用当前 hash 规则。',
  f'冻结范围内的资格结论为 {p1["qualification"]}。截止时间 {p1["cutoff"]}。原有 W-I、历史 BLOCKED/NO_QUALIFIED_C 结果及生产修改均保留。',
  f'原搜索每条观察 71/{U} = {71/U:.2%}。固定池内初始化差异的区间不能推广成跨数据集、跨候选池保证；接近穷举的平局不证明方法等效。' if U else '没有可进入性能协议的合格候选。','',
  f'实际成本：已完成 GNN {cost["GNN_complete_tasks"]}/{U*3} 任务，累计任务时间 {cost["GNN_task_seconds"]:.1f} 秒；GP {cost["GP_fits"]} fits / {cost["GP_starts"]} starts；test {cost["test_evaluations"]} 个去重 checkpoint。',
  f'GNN 启动尝试 {cost["GNN_attempts"]}，未闭合或重复尝试差额 {cost["GNN_repeated_attempts"]}；具体恢复位置见 training_attempts、每任务 attempts 和 status。',
  '完整 C 缓存成本单独披露；30 条缓存回放不是重复训练 30×213 次，也不构成在线端到端加速验证。','']
 if (root/'prediction_metrics.json').exists():
  metrics=read(root/'prediction_metrics.json');lines+=['## A-only 独立预测','']
  for m,v in metrics.items():
   x=v['metrics'];lines.append(f'{m}：MAE {x["MAE_pp"]:.4f}、RMSE {x["RMSE_pp"]:.4f}、偏差 {x["signed_bias_pp"]:.4f} 个百分点；prospective 90% 覆盖 {x["prospective_coverage90"]:.2%}，宽度 {x["prospective_width90_pp"]:.4f}、IS90 {x["prospective_IS90_pp"]:.4f} 个百分点。')
  lines+=['预测在任何 C 标签前封存；prospective 方差使用 A 噪声，C-SE 辅助结果标为 retrospective。预测区间按架构整组重采样。','']
 else:lines+=['独立预测指标尚未计算：必须先完成全部正式轨迹；阶段状态不等于预测有效性。','']
 if (root/'search_metrics.json').exists():
  lines+=['## 同预算 C 搜索','']
  for name,v in read(root/'search_metrics.json')['comparisons'].items():lines.append(f'{name}：n71 验证均差 {v["mean_pp"]:.4f} 个百分点，95% 配对区间 {v["CI95_pp"]}，胜/平/负 {v["wins"]}/{v["ties"]}/{v["losses"]}。')
  lines+=['这两项比较共享冻结 bootstrap 索引；主比较是 W-I−Random，无事后成功门槛。','']
 else:lines+=['搜索结论尚未产生；不根据已完成的局部轨迹做性能判断。','']
 if (root/'test_results.json').exists():
  lines+=['## 返回配置最终节点 test','']
  for name,v in read(root/'test_results.json')['comparisons'].items():lines.append(f'{name}：test 均差 {v["mean_pp"]:.4f} 个百分点，95% 配对区间 {v["CI95_pp"]}，胜/平/负 {v["wins"]}/{v["ties"]}/{v["losses"]}。')
  lines+=['test 仅评价预先封存返回项的 best-validation checkpoints，不额外训练，不回看旧 B/test 作开发对照。','']
 else:lines+=['最终节点 test 尚未完成；没有用 test 选择方法、配置或 epoch。','']
 lines+=['资格成立、预测价值、搜索收益与 test 收益是不同结论。W-I 与 RAW16 的核和表示都不同，不能单独归因于显式结构表示；本轮不新增 L-I 或其他方法。',
  '若任何配对区间不支持收益，应相应收缩该项结论；流程完成本身不等于独立验证成功。C 此后不得继续用于调核后再宣称独立。',
  '模型/optimizer/checkpoint 保留服务器原路径；紧凑包包含相应身份、完整正式决策、指标及必要证据，不包含大权重或原历史混合成绩缓存。']
 (root/'README_RESULTS.md').write_text('\n\n'.join(lines)+'\n')
 complete=status.get('P6')=='COMPLETE'
 (root/'NEXT_REQUIRED_STEP.md').write_text('# 下一步\n\n'+('本轮已结束，不继续调核或追加候选。按预测、搜索及 test 各自的差值与区间审查结论；任何未来按 C 修改方法都使 C 转为开发数据，需要新的独立确认。' if complete else '仅恢复当前未完成阶段，使用本目录运行身份与精确 checkpoint；不重新初始化已完成任务、fit 或评分。若存在技术失败，先查 status 与对应 worker 日志。不得用替换候选、缩小预算或方法修改掩盖未完成。')+'\n')
 (root/'GPT_ANALYSIS_REQUEST.md').write_text('''# 请复核本轮独立验证

先读 README_RESULTS、status、P1_summary、performance_protocol（若已冻结）与 actual_costs。严格区分资格证据、计算完成与各项性能结论。

审查身份投影是否真正修正旧访问限制且没有成绩值解码；一图四配置是否在成绩前保守排除；实际历史成员超集是否绑定完整来源，尤其四份 registry 的 144 个历史文件与 1,667 个不透明身份。未知且可能重合不能放行，可靠完整负匹配也不能机械阻断。

审查 A-only 预测是否在任何 C 标签前封存；与 C 内逐轮拟合分离；prospective 噪声来自 A，搜索标准化、噪声和 best_f 仅来自已释放 T。未查询 mean/SE 和全池统计是否被隔离，三方法是否共同初始化且均止于71，无额外 n71 fit。

审查三 seed 均值/SE、训练等价与 best-state 恢复；30条正式返回是否先封存，test 是否只评价去重返回项的原三 seed checkpoint。成本须包括整个 U×3 验证缓存、GP/采集、失败和恢复，不得包装成在线端到端加速。

按独立预测、W-I−Random 主比较、W-I−RAW16 参照及最终 test 分别报告证据强弱。区间只反映固定池初始化波动；不能把约40%池覆盖下的平局宣称等效或跨大空间无效，也不能把 W-I/RAW16 的差异单独归因于表示。未启动或未完成阶段不得伪造空结果为完成。
''')
 commands=[]
 for path in [root/'actual_commands.jsonl',root/'controller_commands.jsonl']:
  if path.exists():commands.extend(json.loads(line) for line in path.read_text().splitlines())
 (root/'actual_commands.md').write_text('# 实际运行命令\n\n工作目录 `/home/fangxy/D-VAE`，环境 `/home/fangxy/.conda/envs/nas/bin/python`。\n\n```bash\n'+'\n'.join(x['command'] for x in commands)+'\n```\n\n恢复：同环境执行 `-B -m analyse.c_independent_validation all --run-dir '+str(root)+' --resume`。\n')
 codefiles=list((repo/'analyse/c_independent_validation').glob('*.py'))+list((repo/'tests').glob('test_c_independent_*.py'))
 patches=[]
 for p in sorted(codefiles):
  target=root/'code'/p.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  d=subprocess.run(['git','diff','--no-index','--','/dev/null',str(p.relative_to(repo))],cwd=repo,stdout=subprocess.PIPE);assert d.returncode in (0,1);patches.append(d.stdout)
 (root/'code_diff.patch').write_bytes(b''.join(patches))
 # Code baseline user modifications must remain exactly the same.
 frozen=read(root/'run_identity.json');checks={p:sha(repo/p)==frozen['initial_code'][p] for p in ('eval_utils.py','bo_phase4.py')};assert all(checks.values())
 write(root/'protected_production_verification.json',checks)

def package(root,repo):
 report(root,repo)
 excluded_parts={'P1_iterations'};files=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.suffix in ('.pt','.pth','.ckpt','.sqlite') or p.name in ('SHA256SUMS','package_receipt.json') or any(x in excluded_parts for x in p.relative_to(root).parts):continue
  files.append(p)
 (root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in files))
 archive=root.with_suffix('.tar.gz')
 with tarfile.open(archive,'w:gz') as t:
  for p in files+[root/'SHA256SUMS']:t.add(p,arcname=root.name+'/'+str(p.relative_to(root)),recursive=False)
 receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'manifest_entries':len(files),'time':now()};write(root/'package_receipt.json',receipt);print(json.dumps(receipt),flush=True)
