import difflib,shutil,subprocess,tarfile
from .common import *
from .audit import code_files,verify_files

def report(root,repo):
 s=read(root/'summary.json');registry=read(root/'source_schema_and_use_registry.json');scope=read(root/'history_scope.json');gaps=read(root/'minimal_missing_evidence.json');resume=read(root/'resume_verification.json')
 verify_files(root,read(root/'audit_complete.json')['files'])
 protected=read(root/'baseline.json')['protected'];verify_files(repo,protected)
 write(root/'protected_sources_verification.json',{'unchanged':True,'files':protected,'old_C_and_B_states_preserved':True})
 identity=read(root/'run_identity.json');assert code_files(repo)==identity['code']
 diff=[]
 for p in sorted(identity['code']):
  target=root/'code'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/p,target)
  diff.append('diff --git a/'+p+' b/'+p+'\nnew file mode 100644\n')
  diff.extend(difflib.unified_diff([], (repo/p).read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+p))
 (root/'code_diff.patch').write_text(''.join(diff))
 # Exact whitelisted inputs only: contains source code, config and identity/use metadata.
 for p,h in read(root/'input_manifest.json').items():
  if sha(repo/p)!=h:raise ValueError('Input changed at delivery: '+p)
  out=root/'evidence'/p;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/p,out)
 lines=['# C 独立性最小充分核验','',f"计算已完成；资格状态为 **{s['eligibility_state']}**。保留 W-I，不进入下一轮性能实验。",'',
  '本轮只核实“架构及任意 HP/seed 的性能是否曾参与本项目性能开发”。仅无标签生成、VAE 无监督预训练或 WL 字典接触不自动取消资格。合格资格应称“未参与性能开发的架构”，不能称“模型从未见过该结构”。','',
  f"审计截止时间：{scope['cutoff_utc']}；代码 HEAD：{scope['head']}。范围按现有项目目录清单及来源引用封存，未扫描其他用户或整个服务器。",'',
  '| 项目 | 实测结果 |','|---|---|',
  f"| 来源条目 / 原历史文件 | {s['source_count']} / {s['original_historical_files']} |",
  f"| 已证实字节 schema / 具有 schema 版本绑定的来源或事件 | {s['verified_byte_schemas']} / {s['schema_bound_sources']} |",
  f"| 成员覆盖完整的文件或事件 | {s['complete_membership_sources']} |",
  f"| 架构 / 配置分类 | {s['architecture_counts']} / {s['configuration_counts']} |",
  f"| 枚举架构 / 粗结构 SHA256 次数 | {s['enumerated_architectures']} / {s['coarse_hash_evaluations']}（上限 5400） |",
  f"| 条件性历史粗指纹命中架构 / 指纹 | {s['conditional_coarse_hit_architectures']} / {s['unique_conditional_historical_hashes_hit']} |",
  f"| 已确认性能开发精确重合架构 / 新恢复历史图 | {s['exact_development_overlap_architectures']} / {s['new_graph_recoveries']} |",
  f"| 逐候选 × 来源判断 | {s['matrix_rows']} |",'',
  '来源数是审计文件、成员集合或操作事件数，不是独特架构数或训练样本数；来源组有重叠，不能直接相加。', '',
  '用途证据：准备集 336 个配置映射到 192 个架构，与已记录的性能开发成员完全一致；原 768 配置池也覆盖这 192 个架构，所以其架构资格共享开发用途。V1/V2 生成池的后续用途混合，整体保留 USE_UNRESOLVED；但完整图精确不重合足以通过对应来源。第四轮 C 生成与编码事件由封存代码、运行身份及完成清单共同证明只做结构工作，并单独披露；此事件不能覆盖其他历史出现。', '',
  '当前 architecture_diagnostics、BO 导入的 canonical_json_fingerprint 在合法有限图载荷上的字节规则一致；实际完整配置逐行复算通过。ORIGIN/V2 的预标签不可变清单绑定了配置、依赖和代码快照；V1 的已保存版本没有同等完整的导入依赖绑定，但可用完整图直接判断。历史合并 collector 会原样接受已有 fingerprint，不能将当前函数的一致性推广为所有旧记录的版本证明。', '',
  '有限枚举固定输入/输出，对全部内部双射复制节点有效属性（含 GCNII 层属性），不按新编号重新建图。完整图来源使用真实有向带属性同构；仅粗指纹命中不判开发泄漏。未绑定版本或覆盖未闭合来源的未命中只记为条件性排除，不能升级为资格通过。', '',
  f"逐来源决策计数：{s['decision_counts']}。完整来源的排除证明独立保留；并未使用历史缺图总数决定资格。",'',
  '剩余实质缺口是历史记录到原文件/producer 版本的绑定，以及相关开发实验截至截止时间的成员/后续用途闭合。H 清单保留了 3168 个原文件的路径、当时 hash 和计数，registry 只保留指纹到来源组；85 份安全搜索元数据没有绑定当时代码版本。现有必要条件不能证明这些缺口与某个 C 架构无关，因此影响范围如实保留到对应候选，不以期望合格数量调整范围。', '',
  f"合格配置 U={s['qualified_configurations']}。原 50+20+1 无重复方案至少需要 71 个配置，当前缺额 {s['missing_configurations_for_71']}；即使将来达到 71，也不等于原 180 配置方案完整就绪。后续 GNN 预算仅为 U×R，本轮未启动。", '',
  '冻结 WL 字典和 A/C 编码原样引用；45 图、180 配置身份、HP 合法性与唯一性、latent12/WL 维度和有限性通过。字典接触过 A 与原 45 个 C 结构，无表示重选。', '',
  f"所有新增模型/成绩操作为零：{s['operations']}。访问白名单拒绝历史混合成绩缓存；未为 hash 核验打开它们。原 C 的 BLOCKED_C、B 的 BLOCKED 和生产文件身份保持不变。",'',
  f"真实 resume 已通过：新增枚举、补图、模型操作均为 0；校验 {resume['verified_result_files']} 个结果文件，主要结果 hash 不变。程序测试通过与来源资格成立是两个独立结论。",'',
  '结论依据是逐候选证明矩阵与明确缺项；没有合格 C 是来源证据状态，不是 W-I 的性能失败。本轮到此结束。','']
 (root/'README_RESULTS.md').write_text('\n'.join(lines))
 coverage=['# 来源覆盖报告','',f"已冻结来源注册表 {sha(root/'source_schema_and_use_registry.json')}。",'',
  '完整配置文件的覆盖边界是各自文件中的全部记录，不冒充整个项目开发史。表中两份历史版本绑定通过的 schema 来自真实依赖与快照，C 事件也绑定实际代码；这些不证明旧 collector 接收到的每个既有字符串都来自同一版本。','',
  '| 来源 | 架构 | 配置 | 完整图复算 |','|---|---:|---:|---|']
 for p,v in registry['schema_evidence']['full_graph_sources'].items():coverage.append(f"| {p} | {v['architectures']} | {v['configurations']} | {v['regression_matches']} 行通过 |")
 coverage+=['','逐原文件记录在 source_schema_and_use_registry.json；完整已知关联在 record_source_associations.json，possible group superset 不伪装成已知逐文件出现。相同文件的副本只按现有清单已记录 hash 归组，不宣称整个目录相同。','',
  '范围覆盖限制：历史目录清单不是成员清单。缺失名称输入记录于 history_scope.json，不将不同名称路径当替代。相关历史输入可能含性能/test，按授权不打开；恢复入口只能是已有安全身份/图/用途证据。','']
 for g in gaps:coverage+=['## '+g['id'],'',g['missing'],f"影响架构数：{len(g['candidate_architecture_ids'])}。",g['minimal_sufficient_resolution'],'']
 (root/'source_coverage_report.md').write_text('\n'.join(coverage))
 (root/'NEXT_REQUIRED_STEP.md').write_text('''# 下一步所需的最小证据

当前不启动训练或独立性能实验。优先取得以下安全证据，只有新证据到位才新建冻结审计版本：

1. 将现有历史 registry 的成员绑定到原来源文件与当时 producer/代码 hash；区分沿用存储指纹和从 ops/edges 计算指纹的分支。证据位置是 H phase_a 的 inventory/registry、原实验代码快照、纯身份成员清单或已存在使用日志。无需打开成绩。
2. 对范围内仍可能相关的开发实验，提供截至截止时间的完整架构成员/用途清单，或证明它们只引用已排除重合的完整集合。成员清单应绑定协议/运行版本，不能仅给目录名或宣称未用。
3. 只有条件性粗指纹命中的记录才可能需要其无标签带有效属性图及来源绑定。无命中的完整已证 schema 集合不需要恢复历史图。具体命中见 targeted_graph_evidence.json。

若上述证据不可取得，保留 PROVENANCE_UNRESOLVED；不要求恢复全部 11823 项，不改变旧 BLOCKED 标志，不重选 WL、不新增候选。提供新证据属于新身份的审计；原版本可用 commands.md 中的 resume 命令只读复核。

资格闭合后才另行冻结独立性能协议、R 个训练种子与 W-I/RAW16/Random 等约定方法。若要主张显式结构优于 latent，应明确同核 L-I。原无重复方案要求 U≥71，接近最低数意味着过度覆盖，不能直接恢复 180 配置方案；GNN 预算仅为 U×R。本轮不执行这些实验。
''')
 (root/'GPT_ANALYSIS_REQUEST.md').write_text('''# 请复核 C 独立性资格证明

请先阅读 README_RESULTS.md、冻结 protocol/history_scope、source_schema_and_use_registry、source_coverage_report、minimal_missing_evidence，再检查 candidate_source_matrix 与 orbit/schema/身份材料。

重点区分：指纹未命中和历史版本/覆盖证明；粗结构命中和精确带属性同构；无标签暴露和性能开发；候选局部缺口和确实影响全部候选的缺口；独立资格和独立性能效果。

检查是否错误放行 USE_UNRESOLVED，或仍机械要求所有历史图齐全。检查源关联是否把组级可能集合写成逐文件确定成员，是否把不同 64 位身份类型混用，是否不当地将当前函数回归一致当历史版本证据。确认全部 m! 以及 GCNII 属性随节点移动，负匹配没有拓扑剪枝。

检查完整图来源的局部通过是否保留；未知用途已证不重合时是否不会阻断候选。检查剩余全体缺口是否有实际范围/版本证据支持，最小补证是否真能改变判断。合格数量不是程序验收门槛，也不能据此重新定义独立性。

本轮 GP/优化/预测、GNN/VAE/decoder、新候选、C/B/test 成绩及 KG 操作均为零。不得要求用性能实验解决来源证据缺口。输出可接受的证明、需纠正的推断和最小下一步证据。
''')
 commands=read(root/'actual_commands.json') if (root/'actual_commands.json').exists() else []
 cmd=['# 真实运行命令','', '工作目录：/home/fangxy/D-VAE；执行解释器 /home/fangxy/.conda/envs/nas/bin/python。','', '```bash',*commands,'```','', 'resume 会核验冻结输入、代码与完成证据；有新来源证据需新建版本，不覆盖旧证明。','']
 (root/'commands.md').write_text('\n'.join(cmd))
 status=read(root/'module_status.json');status['report']='COMPLETE';write(root/'module_status.json',status)
 print(canonical({'report':'COMPLETE','state':s['eligibility_state'],'protected_unchanged':True}))

def package(root,repo):
 assert read(root/'module_status.json')['report']=='COMPLETE'
 verify_files(root,read(root/'audit_complete.json')['files'])
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS','package_receipt.json')}
 (root/'SHA256SUMS').write_text(''.join(h+'  '+p+'\n' for p,h in files.items()))
 archive=root.with_suffix('.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:tar.add(root,arcname=root.name,filter=lambda info:None if info.name.endswith('/package_receipt.json') else info)
 receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'manifest_entries':len(files),'time':now(),'contains_large_weights_or_credentials':False,'contains_B_C_test_scores':False}
 write(root/'package_receipt.json',receipt);print(canonical(receipt))
