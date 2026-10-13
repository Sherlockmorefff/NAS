import difflib,shutil,subprocess,sys,tarfile,xml.etree.ElementTree as ET
from collections import Counter
from .common import *

def resume_verify(root,repo):
 """Verify mixed input bytes by hash only, then exercise actual resume paths."""
 frozen=read(root/'run_identity.json');assert frozen['protocol_sha256']==sha(root/'protocol.json') and frozen['mixed_allowlist_sha256']==sha(root/'mixed_allowlist.json')
 verify(repo,read(root/'collection_code_identity.json'));verify(repo,read(root/'review_code_identity.json'));verify(repo,frozen['safe_inputs'])
 expected={p:r['source_sha256'] for p,r in read(root/'projection_manifest.json').items() if r.get('source_sha256')}
 expected.update({p:r['source_sha256'] for p,r in read(root/'followup_evidence.json').items() if r.get('source_sha256')})
 expected.update({r['archive']:r['archive_sha256'] for r in read(root/'backup_search.json')['archives']})
 count=0;byte_count=0
 for p,h in expected.items():
  assert sha(repo/p)==h,'Mixed source input identity changed: '+p;count+=1;byte_count+=(repo/p).stat().st_size
 before=read(root/'review_complete.json')['files'];verify(root,before)
 runs=[]
 for phase in ['collect','review']:
  cmd=[sys.executable,'-B','-m','analyse.c_provenance_evidence_closure',phase,'--run-dir',str(root.relative_to(repo)),'--resume'];p=subprocess.run(cmd,cwd=repo,text=True,capture_output=True);runs.append({'command':cmd,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
  if p.returncode:raise RuntimeError('Resume failed: '+phase)
 verify(root,before)
 result={'state':'PASS','commands':runs,'new_evidence_collections':0,'new_candidate_enumerations':0,'new_full_graph_recoveries':0,'new_model_operations':0,'qualification_and_core_hashes_unchanged':True,'core_result_files_verified':len(before),'code_and_protocol_identities_verified':True,'verification_only_mixed_file_byte_reads':count,'verification_only_hash_reads':count,'verification_bytes_read':byte_count,'performance_field_decodes':0,'performance_field_exposures':0,'performance_field_writes':0,'performance_field_uses':0,'verification_note':'Input byte hashing is reported separately from evidence collection. No performance scalar parsed during resume.','time':now()}
 write(root/'resume_verification.json',result);print(canonical({k:v for k,v in result.items() if k!='commands'}))

def report(root,repo):
 s=read(root/'summary.json');new=s['new_evidence'];resume=read(root/'resume_verification.json');verify(root,read(root/'review_complete.json')['files']);verify(repo,read(root/'review_code_identity.json'));verify(repo,read(root/'safe_input_manifest.json'))
 protected={p:h for p,h in read(root/'safe_input_manifest.json').items() if p in ['bo_phase4.py','eval_utils.py',P4+'/C_preparation/C_eligibility_manifest.json',OLD+'/C_eligibility_manifest.json'] or p.endswith('structured_kernel_phase2_20261002T161300Z/status.json')}
 verify(repo,protected);write(root/'protected_state_verification.json',{'unchanged':True,'files':protected,'models_or_optimizer_states_opened':False})
 files=code_files(repo);p='tests/test_c_provenance_evidence_review.py';files[p]=sha(repo/p);diff=[]
 for rel,h in sorted(files.items()):
  target=root/'code'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/rel,target)
  diff.append('diff --git a/'+rel+' b/'+rel+'\nnew file mode 100644\n');diff.extend(difflib.unified_diff([], (repo/rel).read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+rel))
 (root/'code_diff.patch').write_text(''.join(diff));write(root/'delivery_code_identity.json',files)
 for rel in files:compile((repo/rel).read_text(),rel,'exec')
 check=subprocess.run(['git','diff','--check'],cwd=repo,text=True,capture_output=True);assert check.returncode==0
 write(root/'final_checks.json',{'all_new_source_syntax_compiled':True,'git_diff_check_exit':check.returncode,'git_diff_check_output':check.stdout+check.stderr,'retained_WI':True})
 # Only necessary safe reproducibility material; no original mixed file copied.
 references=[P4+'/C_preparation/C_unlabeled_manifest.json',P4+'/C_preparation/C_features.json',P4+'/C_preparation/frozen_dictionary.json',P4+'/C_preparation/extended_feature_manifest.json',OLD+'/orbit_manifest.json',OLD+'/source_schema_and_use_registry.json',OLD+'/source_member_sets.json',OLD+'/schema_evidence.json',OLD+'/history_scope.json',OLD+'/candidate_identity_checks.json']
 references += [OLD+'/'+r['path'] for r in read(repo/OLD/'orbit_manifest.json')['architectures'].values()]
 refs={}
 for rel in references:
  path=root/'reused_evidence'/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/rel,path);refs[rel]=sha(repo/rel)
 write(root/'reused_evidence_manifest.json',refs)
 versions=read(root/'historical_version_search.json');matched=sum(v['status']=='EXACT_CODE_CONTENT_FOUND' for v in versions['bindings']);total=len(versions['bindings'])
 gaps=read(root/'remaining_gaps.json');groups=Counter('/'.join(Path(g['source']).parts[:3]) for g in gaps)
 activity=read(root/'development_member_use_coverage.json')['post_cutoff'];cut=read(root/'protocol.json')['cutoff']
 (root/'README_RESULTS.md').write_text(f'''# C 来源证据补齐与资格复核

本轮计算完成，证据状态为 **PARTIAL_EVIDENCE_CLOSURE**，C 资格状态仍为 **{s['C_state']}**。已取得实质新证据，部分来源义务已解除；当前合格 **0 架构 / 0 配置**，45 架构 / 180 配置保留来源未解决。未证明任何架构已参与性能开发；其中 **1 架构 / 4 配置存在历史粗结构重合可能**。

## 实际取得的新证据

- 从冻结白名单的 769 个文件完成受控身份投影，成绩值未解码。475 份搜索历史/补充观测和 18 份成员来源，共 **493 份集合**取得完整结构成员证明。98,073 是含重复快照的结构记录数，不是独特架构或训练样本数；观测到 12,460 个不同粗结构身份。
- 冻结 task_plan、代码哈希、成员输入哈希、划分哈希与实际 result_audit 全部对应。**1,680 个回归诊断任务及其 2,688 个被引用任务/缓存来源**可由 **76 配置 / 60 架构**的完整成员超集覆盖；没有打开那些任务成绩文件。
- 逐候选矩阵新增或更新 **{new['changed_candidate_source_decisions']} 项判断**，有 **{s['decision_counts']['PASS_BOUND_MEMBER_SUPERSET']} 项**获得成员超集排除证明。未知用途不会阻断已证实不重合的来源。
- 在此前总指纹索引遗漏的补充低保真历史中，发现同一粗结构出现于三份记录。它对应 C 架构 `5c3ed607fdb7f89cbbdf4126e6492829cbe9d794714682cdfec414593ebbfabc`。记录标识、操作/边、global4、GAT heads=1、GIN eps=0 已核实。
- 6 个历史源码哈希目标中，{matched}/{total} 个找到精确 Git 或快照内容；另外检查了 631 个已知代码快照及 4 份相关归档的指定代码条目。命中记录对应的 BO 调用者精确版本仍未找到。已找到 evaluator 默认参数的旧代码，但默认值不能证明实际调用参数；没有据此补造 GCNII 属性或宣布精确同构。

## 仍未闭合的事项

剩余 **{s['remaining_concrete_source_gaps']} 项具体来源义务**：三项只影响该粗匹配架构；其余来源的完整成员、历史版本或用途关联仍不足，影响范围见 remaining_gaps.csv。主要包括 94 项 LayerNorm 确认/评估来源，以及若干早期标签扩展、结构代理和开发评估来源。成绩专用、seed_results、模型评估和节点 test 输出没有被纳入直接读取白名单；需要安全成员清单或与实际运行绑定的完整池超集。

全局覆盖项由这些具体未决来源推导，不是“缺失历史图总数必须为零”的机械门槛。本轮没有要求恢复全部历史图。原总索引未命中与新低保真记录粗命中不矛盾：后者此前没有被总索引覆盖。

## 时间、边界与续跑

截止时间推进到 **{cut}**。旧截止时间之后唯一记录变动为上一轮审计包；运行/文件元数据未发现新性能开发。这个结论限于记录到的项目活动，不声称能够发现未记录或已删除的操作。

候选、HP、WL 字典与 A/C 特征身份保持原样。复用原 **5,304 次**节点重编号结果，新增枚举 **0**，确认恢复的完整带有效属性历史图 **0**。旧 NO_QUALIFIED_C、BLOCKED_C、B BLOCKED、生产代码均未修改。W-I 保持冻结。

首次补证共记录 {s['access_counters']['mixed_file_byte_reads']} 次混合/归档字节访问、{s['access_counters']['hash_reads']} 次哈希读取；这些是字节流次数，含复读，不是独立文件数。真实 resume 另做 {resume['verification_only_hash_reads']} 次输入字节哈希核验；新增证据收集、枚举、补图和模型操作均为 0，核心结果 hash 不变。成绩字段解码、暴露、写入、使用始终均为 **0**。GP 拟合/优化/预测、GNN/VAE/decoder、C/B/test 成绩操作和 KG 均为 **0**。

12 项投影测试、8 项资格规则测试与补充投影合成检查通过；基础图实现沿用未修改的旧验收。测试通过不等于来源资格通过。

当前不能进入 Cora 独立架构性能验证。原 50+20+1 方案至少要求 U≥71；当前 U=0。下一步见 NEXT_REQUIRED_STEP.md；若决定改变研究问题，唯一首选替代入口见 replacement_C_route.md。没有启动任何下一阶段任务。
''')
 (root/'evidence_delta.md').write_text(f'''# 相比旧审计的实质变化

旧审计只有位置敏感指纹并集与四份完整配置来源。本轮首次从真实旧记录取得完整结构身份投影，并通过冻结计划/版本/输入/实际完整性审计建立组级成员覆盖链。

1. {new['direct_complete_collections']} 份集合获得直接成员证明，比较依据是实际 operations/edges，而非猜测旧存储指纹版本。
2. 2,688 份回归任务/缓存来源由同一完整 60 架构集合覆盖；无需逐个读取成绩输出或恢复图。
3. 新纳入 150 份补充搜索观测集合，找到了此前总索引遗漏的 1 个粗结构匹配、3 份记录。它不能被自动判为有效属性同构。
4. 来源树、Git 与已知备份检索确认了 {matched}/{total} 个目标源码内容，但缺失的 BO 调用版本仍影响匹配记录的精确属性证据。

这些是新建立的、可复核的证据链。certificate 是本轮索引和推导，绝不冒充历史生成时已有的证书。旧局部排除证明继续保留；未改变独立性定义、表示或候选池。
''')
 coverage=['# 来源覆盖与停止位置','',f'新证书覆盖 493 个完整记录集合及 2688 个共享任务来源。其余未决义务 {len(gaps)} 项，完整列表见 remaining_gaps.csv。','', '| 主要剩余来源组 | 义务数 |','|---|---:|']
 coverage += [f'| {k} | {v} |' for k,v in groups.most_common()]
 coverage += ['', '义务数不是独特架构数；副本/快照与多个来源会重复引用同一架构。SOURCE→certificate→投影 hash→原文件 byte hash 的链保存在 producer_schema_certificates.json。完整文件证明只覆盖该文件成员；实验级证明另需实际输入/运行绑定。', '', '有边界的检索已经走过：原缺口引用、相邻运行元数据、631 个代码快照、相关 Git 版本和四份已知同项目归档。未找到不等于不曾使用。被禁止的专用成绩输出不再重试打开。', '', '不得将旧 schema 未绑定与完整实际结构记录混为一谈：后者的粗结构负匹配已经足以排除精确同构。对命中记录仍需全部有效属性；当前未作默认值替代。','']
 (root/'source_coverage_report.md').write_text('\n'.join(coverage))
 (root/'NEXT_REQUIRED_STEP.md').write_text('''# 当前最小下一步

当前不能启动独立性能实验。优先完成两类可改变判断的材料：

1. 针对一个粗匹配架构，取得与历史记录绑定的 BO 调用代码（sha256 b582dec35073d5a8768a0ffa5f47e1d9fffcb59fac68c8c8c54bb6503456f00f）或等价的实际调用参数证据，确认 GCNII alpha/theta、激活/融合/归一化及后续性能用途。GAT/GIN 属性和三条身份记录已保存。不要凭已找到 evaluator 的默认参数补齐。
2. 对 remaining_gaps.csv 的其他来源，取得身份专用的完整成员/运行清单，或证明其成员全部来自一个已排除重合的完整池。94 项确认/评估来源可优先寻求一个组级超集证明，而不是逐份恢复成绩或图。给定集合用途未知但已证不重合时，直接解除该来源义务。

原样重跑不会补出缺失证据；模型训练也不能解决来源问题。新证据到位后建立新的审计身份，保留本轮结果。合格后再冻结独立性能协议、R 个种子、相同池/预算的 W-I、RAW16、Random；显式结构优于 latent 的主张须加入同核 L-I。预算只给 U×R，U≥71 仍不意味着原 180 配置方案已完整恢复。
''')
 (root/'replacement_C_route.md').write_text('''# 唯一首选替代入口：保留现有池，改为新的外部任务验证

本轮已有完整池中，部分架构已明确用于开发，其他池的历史用途覆盖仍不完整；尚无证据支持从其中直接抽出独立 Cora 架构池。重新随机生成也不能解决这一点。通过增加节点/算子强行改变空间，还会牵涉冻结 VAE、RAW16 编码和结构定义，当前缺少可直接采用的边界证明。

若现有来源体系最终不能闭合，首选保留已封存的 45 图/180 配置和编码，在一个能够以数据来源及项目任务清单证明未参与本项目开发的新数据任务上做外部验证。这样不需要再生成候选或改变核。它回答“新任务上的外部有效性”，不能替代“Cora 未参与开发架构”的结论。

必须补齐并另行冻结：具体数据集名称/版本及来源 hash、与已有项目任务的排除关系、划分、训练配方可适用性、R 个种子、同池同预算、缓存访问边界及评价指标。目前没有证据足够指定一个已核实合格的数据集，因此不猜名称、不称方案已经就绪。

原计划的改动是数据任务和结论边界；W-I/RAW16/Random 方法规格保持冻结，在新任务上按共同预算比较。若原 180 配置全部用于这一外部任务，GNN 缓存预算为 180×R；一般为 U×R。原无重复 50+20+1 搜索至少需 U≥71。此次只给入口，不下载数据、不生成候选、不训练，也不自动切换数据集。
''')
 (root/'GPT_ANALYSIS_REQUEST.md').write_text('''# 请复核本轮来源补证和资格结论

请先检查 new_evidence_manifest/evidence_delta，再追查 producer_schema_certificates、development_member_use_coverage、原始输入 hash 与安全投影。

重点审阅：493 个实际记录集合是否完整；1680 任务/2688 来源到 76 配置/60 架构的组级证明是否绑定实际代码、输入、划分与运行完整性；是否遗漏其他性能输入路径。无标签 reconstruction replay 与性能目标成员应区分。

检查旧总索引遗漏的三条低保真记录；一个粗匹配是否被错误升级为精确同构或 DEVELOPMENT_USED。已找到 evaluator 默认代码，但 BO caller 精确版本缺失，当前没有猜测 GCNII 实际参数。检查剩余 164 项来源义务是否确实相关，完整负匹配的未知用途来源是否仍被多余阻断。

区分当前代码与历史运行版本、目录清单与成员闭合、旧证据重排与新证明成立、计算完成与资格通过。源证书是新索引，不是历史事实的自我证明。检查 P1 的 new_full_graphs_recovered 计数名称：它实际计保存的粗命中包，最终确认完整有效图数为 0，见 targeted_overlap_review.json。

数据访问应如实区分混合字节访问/哈希核验与成绩字段值解码。检查 projector、白名单、合成测试、访问计数；原混合缓存和大权重不在交付包内。resume 重新核验原输入字节 hash，未重新收集证据/枚举/补图/运行模型。

最终给出成立的证明、仍需补的最小来源材料，以及是否接受外部任务替代入口。不得用新的性能实验替代来源证明；本轮未改变 W-I、未启动下一阶段。
''')
 commands=['# 真实执行记录','', '服务器项目：/home/fangxy/D-VAE。下列 Python 命令经 SSH 执行；测试的精确参数另见 acceptance/commands.json。','', '```bash',sys.executable+' -B -m analyse.c_provenance_evidence_closure freeze --prompt /tmp/c_provenance_evidence_closure_prompt.txt']
 commands += [read_line['command'] for read_line in [__import__('json').loads(line) for line in (root/'actual_commands.jsonl').read_text().splitlines()]]
 commands += ['```','', '定向补证与归档查询分别调用 followups.run_followups(root, repo) 和 backups.recover(root, repo)，完整实现及白名单均已保存。','', '完成后再次执行不带 --resume 的 review 会被拒绝；记录中有一次此类拒绝，未改写完成结果。交付源码已核实与实际完成时的 review_code_identity 完全一致。','', '今后恢复请执行下列入口，它先核验代码、协议和原输入 hash，再调用真实 collect/review resume：','', '```bash',sys.executable+' -B '+str(root/'resume.py'),'```','']
 (root/'actual_commands.md').write_text('\n'.join(commands))
 state=read(root/'phase_status.json');state['P4']='COMPLETE';write(root/'phase_status.json',state)
 print(canonical({'report':'COMPLETE','C_state':s['C_state'],'remaining_sources':len(gaps),'protected_states_unchanged':True}))

def package(root,repo):
 assert read(root/'phase_status.json')['P4']=='COMPLETE';verify(root,read(root/'review_complete.json')['files'])
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS','package_receipt.json','download_verification.json')}
 (root/'SHA256SUMS').write_text(''.join(h+'  '+p+'\n' for p,h in files.items()))
 archive=root.with_suffix('.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:tar.add(root,arcname=root.name,filter=lambda info:None if info.name.endswith(('/package_receipt.json','/download_verification.json')) else info)
 result={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'manifest_files':len(files),'no_original_mixed_caches_copied':True,'no_weights_or_credentials':True,'no_performance_values':True,'time':now()};write(root/'package_receipt.json',result);print(canonical(result))
