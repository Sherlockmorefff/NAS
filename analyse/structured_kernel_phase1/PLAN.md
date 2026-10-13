# 已批准的服务器 E0–E2 实施计划

本计划由用户批准，完整技术规格同时保存在 PROMPT.txt 的 Prompt B。只运行现有 A 标签上的 E0、E1、E2；不得访问 B/test、新训练 GNN/VAE、运行 KG、CLML、容量分支或参数扫描。不 commit/push，不切分支、不覆盖原生产修改或历史结果。

## 当前身份和来源

服务器 ismc36-ts，项目 /home/fangxy/D-VAE；分支 feature/accuracy-bottleneck-diagnostics；HEAD 64a44c14316f11debaf01cf8583426acfe1410e6。保留 bo_phase4.py、eval_utils.py 已有修改。实际版本和逐文件哈希执行时再次保存。

A特征和配置源为 results/posthoc/fuzzy-scheduler-regularization-v1-20260921T064826Z；唯一允许的标签文件为该目录 private/evaluation_A.json。原fold来自 results/posthoc/fuzzy-metric-stage2-20260910-v1/partition.json，冻结fold/subsets在 artifacts/dkl-stage2-v1-20260922T080157Z。配置和latent来源为 results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2，不得误用prepared。VAE为 checkpoints/joint_model_pipeline_global4_best.pth。

## E0 数据和访问边界

复核179候选、96架构、全部candidate/HP/architecture身份、配置、latent12、HP4、配方和来源哈希。Cora public、NormalizeFeatures、global4、No-LN、full batch、max_epochs300/patience80、seeds0/1/2；150/40不得混入。

SE已是三seed均值标准误；所有179项重新验证 mean 和 sample_std(ddof1)/sqrt3。known_var=SE²。保留原五折Q=[38,36,36,34,35]，验证架构隔离。同draw/fold的n20为n50有序前缀；draw20260922/20260923严格复用SeedSequence([draw,fold])对sorted training pool的permutation。

每个T独立计算ddof0输入与y尺度，常量x列scale1，零/非有限y尺度BLOCKED。HP已经取对数不再变换。known_var_std=SE²/y_scale²；extra_noise初值.01和下界1e-6为标准化方差，不重复换算。Q只提供无标签输入；fit、预处理、起点选择均不读Q成绩/SE。独立scorer在模型/预测封存后读取A内对应Q标签。完整candidate_manifest只用于审计/评分。

## E1 历史证据

只读复用 stage2、stage2-length、residual-constraint 三个已核实包：RAW16、FROZEN、REG、压缩DKL、残差DKL和长训。它们是constant mean、isotropic Matérn5/2、FixedNoise+extra噪声族，不与生产ARD RBF混合。核实监督预训练T边界、代码/source哈希及980个checkpoint x/y/known与T张量一致。

只恢复真实存在的0/10/50/100/300/1000点；10/50缺失按missing记录；压缩DKL中间模型被覆盖，不补造Q/几何轨迹；固定映射可从完整参数严格重建的点先核对endpoint容差atol1e-8/rtol1e-6。残差有post-update诊断，长训使用真实total_joint_step1000到4000。更新前NLL与更新后参数不能冒充同步状态。

输出train NLL、Q MAE/NLPD、先验核R在T及T∪Q上的非对角median/p95、r_eff=(trR)^2/tr(R²)、映射距离、ell/outputscale/mean/extra_noise、model seed预测分歧、真top10%及预测top5 mean(mu-y)。旧Q-SE评分标retrospective，并由冻结预测重新算prospective。不补训、不按Q选checkpoint。只有NLL改善、Q恶化、有效秩下降同时出现才支持过拟合/过度相关；距离缩小本身不证明坍缩。

## E2 冻结模型

五组L-A/L-I/W-A/W-I/RAW16。L用latent12，W用候选算子有向图WL H2（不是Cora图）。WL保留角色、操作、激活、sum融合和真实算子属性，GCNII序号作为语义属性保留；不包含hidden_dim/dropout/lr/l2，不制造边。每轮旧标签+排序入邻居+排序出邻居，h0/1/2分开字典。仅A179无标签池建字典，明确使用Q无标签结构。psi=[phi/L2norm(phi),n_nodes/n_max,n_edges/e_max]，不降维，不逐列标准化WL；正式特征manifest记录维数和碰撞。

各距离只除以当前T严格正两两距离中位数；无正距离取1并记录。RAW16为T标准化16维整体距离，不使用分块缩放后的拼接。

M(r)=(1+sqrt5*r+5*r²/3)exp(-sqrt5*r)。ka=M(da/ell_a)，kh=M(dh/ell_h)。K_A=sf²(ka+kh)/2；K_I=sf²[.25(ka+kh)+.5ka*kh]。eta固定.5。四组5标量，RAW16单整体ell共4标量。不ARD、不加先验/损失。

CPU float64、workers1、每进程threads1。三个起点ell=.5/1/2，sf²1、extra_noise.01、mean0；正下界ell1e-4/sf²1e-8/noise1e-6。每起点Adam lr.01 300步，继以PyTorch L-BFGS lr1 max_iter100 max_eval125 history100 strong_wolfe tolerance_grad1e-7 tolerance_change1e-9。MLL已除n，不二次除n；min_fixed_noise显式0。最终有限训练NLL选起点，同值按0/1/2；不ensemble。记录actual closure预算及结束原因。

仅Cholesky数值错误可按jitter1e-8/1e-7/1e-6重试同一参数处计算；超过上限FAILED，不增加起点/步数。jitter不是观测噪声。记录梯度、末段目标变化；Adam末50步改善>1e-4或LBFGS预算耗尽标warning，不自动延长。任一起点失败则setting不算完整成功。

## Pilot、验收与恢复

先合成数值验证：映射、隔离、嵌套、噪声单位、核对称/对角/Cholesky/独立NLL、posterior有限、WL重编号与连接、Q投毒、训练NLL选点、scorer不修改状态。再fold0/draw20260922/n20,n50/五方法pilot=10setting30起点。pilot基于工程完整性而不是Q好坏判断；通过后自动其余90setting270起点。总100setting300起点，两个噪声评分版本不新增拟合。

状态PENDING/RUNNING/COMPLETE/FAILED/BLOCKED；保留T/Q、scaler、feature/code/recipe/split、optimizer身份。仅身份相同且模型、预测、metrics、三起点日志齐全的COMPLETE可跳过。每25 Adam步保存状态；已完成起点复用；中断LBFGS从同一起点的已完成Adam边界恢复且记录实际重复工作。代码/科学配置变化新run版本，不混合旧pilot。

## 评分、判定与交付

主MAE单位百分点。prospective_var_std=latent_var_std+extra_noise_std+median_T(known_var_std)。NLPD用raw accuracy密度/nat；90%coverage/width/IS90均用此前瞻噪声。附latent interval、标明retrospective的Q-SE分解、Spearman、两偏差、regret@5，同分candidate_id。IS90=width+20*lower_miss+20*upper_miss。

旧模型先聚合seeds0/1/2，同fold/n再平均两个draw，再五fold等权。新静态GP只用选中起点。n50主、n20完整；全部三成对消融与四组vsRAW16。不得把重叠样本当独立重复做简单显著性。

所有100setting/300起点成功才可判通过：n50 MAE比RAW16降>=.3pp、>=4/5fold改善、NLPD恶化<=.1nat、IS90恶化<=5%、n20 MAE恶化<=.3pp。多个通过按n50 MAE，<.05pp并列，再NLPD，<.01nat并列再加法、latent。仅给唯一待第二轮确认方法，否则NO_STABLE_GAIN。

新run目录交付PLAN/protocol/candidate/split/feature/preflight/status/fit_log/predictions/context+fold metrics/mapping_diagnostics/legacy_missing/代码diff/实际命令/README_RESULTS/GPT_ANALYSIS_REQUEST/证据tar.gz。包排除凭据、B/test、大型不必要权重。报告区分事实/解释/假说，不把A探索收益当B/test或搜索成功，不直接执行下一轮。
