"""Render the finished experiment using saved statistics only; no model imports."""
import argparse
import csv
import datetime
import json
import statistics
from pathlib import Path

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def fmt(x,d=4):return '未定义' if x is None else f'{x:.{d}f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
def avg(v):return statistics.fmean(v) if v else None
def nested(rows,field):
    groups={}
    for r in rows:groups.setdefault((r['fold'],r['sample_seed']),[]).append(r[field])
    folds={}
    for (f,_),v in groups.items():folds.setdefault(f,[]).append(avg(v))
    return avg([avg(v) for v in folds.values()])

def main(root):
    gate=read(root/'diagnostic_gate.json');a=read(root/'scoring_A/summary.json');geom=read(root/'historical_geometry_summary.json')
    ledger=read(root/'resource_ledger.json');hist=read(root/'historical_manifest.json');status=read(root/'status.json')
    b=read(root/'scoring_B/summary.json') if (root/'scoring_B/summary.json').exists() else None
    lines=['# DKL错误相似性与受约束残差映射对照','']
    route=gate['route'];lines+=['A诊断完整覆盖60个原上下文、280份无查询真值的几何/预测记录。**分流为 '+route+'。** n50共同回归与旧DKL1000的最近邻性能误差相对原始16维特征分别增加 '+fmt(gate['comparisons']['n50_COMMON_REG']['mean_delta_E1_pp'])+' / '+fmt(gate['comparisons']['n50_OLD_DKL_1000']['mean_delta_E1_pp'])+' 个百分点，均为4/5折变差，满足预先规定的进入条件。因此执行两个样本量的完整残差对照。','']
    if b:
        for n in (20,50):
            v=b['per_n'][str(n)];p=v['paired_comparisons'];m=v['method_means'];c=m['R_DKL_C']
            lines += [f'**n={n}：{v["scientific_signal"]}；{v["constraint_signal"]}。** 相对REG的regret5改善 {fmt(p["R_REG"]["metrics"]["regret5_improvement_pp"]["mean"])} pp；相对无持续约束U改善 {fmt(p["R_DKL_U"]["metrics"]["regret5_improvement_pp"]["mean"])} pp。后续候选规则：{v["followup_candidate"]}。','']
        for n in (20,50):
            status[f'scientific_signal_n{n}']=b[f'scientific_signal_n{n}'];status[f'constraint_signal_n{n}']=b[f'constraint_signal_n{n}']
    lines += [f'执行状态 **{status["execution_status"]}**；完成 {status["formal_contexts_actual"]}/60 个共同上下文、{status["formal_endpoints_actual"]}/260 个终点及 **{status["formal_updates_actual"]:,}** 次正式更新。工程有效、训练稳定、预测质量与选优收益分别判断。A池此前已反复用于开发；本轮属于开发性离线对照，不能称独立确认、统计显著或已证明真实搜索收益。','',
      '## A：错误近邻与尺度变化','',
      'E1是查询点与当前所选20/50条中最近训练点的真实性能差，E5是前5近邻的平均差，均为百分点、越低越好。共同GP训练不改变映射，因此COMMON_REG与COMMON_GP不算两份独立近邻证据。','']
    ar=[]
    for n in (20,50):
        for m in ['RAW_X','COMMON_REG','COMMON_GP','OLD_DKL_1000','OLD_DKL_4000','OLD_RAW16']:
            v=a['per_n'][str(n)][m]
            ar.append([n,m,fmt(v['E1_pp']['mean']),fmt(v['E5_pp']['mean']),fmt(v['mae_pp']['mean']),fmt(v['coverage90_fraction']['mean']),fmt(v['interval90_width_pp']['mean']),fmt(v['nlpd_accuracy_units']['mean'])])
    lines += [table(['n','历史参照','E1↓','E5↓','MAE↓','90%覆盖','90%宽度(pp)','NLPD↓'],ar),'']
    gr=[]
    for name,g in gate['comparisons'].items():
        gr.append([name,*[fmt(g['fold_delta_E1_pp'][str(f)]) for f in range(5)],fmt(g['mean_delta_E1_pp']),g['positive_folds'],g['mapping_support']])
    lines += [table(['门槛比较','折0','折1','折2','折3','折4','平均ΔE1','正折数','通过'],gr),'',
       'ΔE1为映射减RAW_X，正值表示更差。只有COMMON_REG/OLD_DKL_1000的E1用于分流；4000步、E5、高相关比例及区间质量均未用于改变门槛。1e−9 pp只用于判数值同分。n20未通过，不隐去其混合方向；n50通过后按协议启动全部n20/n50矩阵。','']
    gtable=[]
    for r in geom['summary']:
        if r['representation']=='COMMON_REG':continue
        v=r['metrics'];get=lambda k:fmt(v[k]['fivefold_equal_mean'])
        gtable.append([r['n'],r['representation'],get('s_h'),get('ell_over_s_h'),get('QT_effective_distance_median'),get('QT_shape_change_vs_raw'),get('nearest_changed_vs_raw')])
    lines += [table(['n','参照','s_h','ell/s_h','QT中位d/ell','QT shape变化/RAW','最近邻更换/RAW'],gtable),'',
      '去整体尺度后的形状与近邻仍改变，同时相对长度尺度和有效核距离改变。各向同性长度尺度及整体缩放不能改变欧氏近邻次序，因此这支持对映射进行干预；但证据只能说明**映射与尺度共同变化**，不能证明映射是唯一原因。训练+查询几何、零距离比例、K/R变化及高相关但性能差的辅助比例均保存在historical/与scoring_A/中。COMMON_REG不产生科学GP预测；没有重建缺失的历史随机初始状态。','',
      '## B：固定终点选优与实际提升','',
      'C=R_DKL_C；U=R_DKL_U。regret5越低越好，gain5越高越好。以GP均值降序选5个，同分按ID；没有均值裁剪、多样性补丁、EI/KG或最佳checkpoint选择。先平均同折/抽样的3模型种子，再平均2抽样种子，最后5折等权；RAW16每子集只有一次实际拟合。','']
    if b:
        mr=[];pr=[];fr=[]
        for n in (20,50):
            v=b['per_n'][str(n)];m=v['method_means'];c=m['R_DKL_C']
            for method in ['R_FROZEN','R_REG','R_DKL_C','R_DKL_U','RAW16']:
                x=m[method];mr.append([n,method,fmt(x['regret5_pp']),fmt(x['gain5_pp']),fmt(x['break_best_rate']),fmt(x['has_headroom_fraction']),fmt(x['headroom_pp'])])
            mr.append([n,'精确随机5选',fmt(c['random_regret5_pp']),fmt(c['random_gain5_pp']),fmt(c['random_break_best_probability']),'同一查询折','同一训练最佳'])
            for ref in ['R_REG','R_DKL_U','R_FROZEN','RAW16']:
                x=v['paired_comparisons'][ref]
                pr.append([n,ref,fmt(x['metrics']['regret5_improvement_pp']['mean']),fmt(x['metrics']['gain5_improvement_pp']['mean']),fmt(x['metrics']['break_best_improvement']['mean']),str(x['fold_regret5_directions'])])
                fr.append([n,ref,*[fmt(x['fold_regret5_improvement_pp'].get(str(f))) for f in range(5)]])
            pr.append([n,'随机5选',fmt(c['random_regret5_pp']-c['regret5_pp']),fmt(c['gain5_pp']-c['random_gain5_pp']),fmt(c['break_best_rate']-c['random_break_best_probability']),'非新增独立重复'])
        lines += [table(['n','方法','regret5(pp)↓','gain5(pp)↑','破训练最佳比例','有headroom比例','平均headroom(pp)'],mr),'',
           'gain5=max(0,所选最好−训练已知最好)；headroom=max(0,查询最好−训练已知最好)。随机gain逐个次序统计量计算期望，没有用max(0,随机最好期望−训练最好)代替。带headroom子集的描述性表单独保留，正式主结论仍使用全部上下文。','',
           table(['n','C的比较基线','regret改善=基线−C','gain变化=C−基线','破纪录率变化','逐折改善/平/退'],pr),'',
           table(['n','基线−C regret(pp)','折0','折1','折2','折3','折4'],fr),'',
           '两个DKL与REG的分叉后映射/GP更新预算配平，但REG分步优化、DKL联合优化，不能称优化器调用数或算力相等。C/U同构、共享同一受约束共同起点，差异仅为分叉后是否继续投影；此比较不单独证明约束预训练必要。','',
           '## 泛化、区间与新近邻','']
        cr=[]
        for n in (20,50):
            for m,x in b['per_n'][str(n)]['method_means'].items():
                cr.append([n,m,fmt(x['E1_pp']),fmt(x['E5_pp']),fmt(x['mae_pp']),fmt(x['coverage90_fraction']),fmt(x['interval90_width_pp']),fmt(x['coverage95_fraction']),fmt(x['interval95_width_pp']),fmt(x['IS90_pp']),fmt(x['nlpd_accuracy_units'])])
        lines += [table(['n','方法','E1↓','E5↓','MAE(pp)↓','90%覆盖','90%宽度','95%覆盖','95%宽度','IS90(pp)↓','NLPD↓'],cr),'',
           '新终点近邻ID由无真值诊断器在封存前计算，E1/E5由封存后的独立评分器计算。观测方差=训练y尺度²×(max(原潜在标准化方差,0)+额外噪声方差)+查询SE²；extra已是方差，SE不再次除3。负舍入原值保留，低于−1e−8会停止。区间宽度与IS90单位为pp，NLPD用原accuracy单位。覆盖上升必须结合区间宽度、IS与NLPD；不能据此宣称EI/KG有效。','']
        for n in (20,50):
            m=b['per_n'][str(n)]['method_means'];c=m['R_DKL_C'];u=m['R_DKL_U'];raw=m['RAW16'];reg=m['R_REG']
            lines += [f'n={n}：持续约束使E1相对U降低{fmt(u["E1_pp"]-c["E1_pp"])} pp，但E5相对U增加{fmt(c["E5_pp"]-u["E5_pp"])} pp，不能把最近邻改善推广为所有邻域均改善。90%覆盖从U的{100*u["coverage90_fraction"]:.2f}%升至C的{100*c["coverage90_fraction"]:.2f}%，IS90从{fmt(u["IS90_pp"])}降至{fmt(c["IS90_pp"])} pp，NLPD也降低；但C的IS90仍高于RAW16的{fmt(raw["IS90_pp"])}，覆盖仍低于名义90%。区间改善没有同时形成一致的选优优势。C相对REG的gain变化为{fmt(c["gain5_pp"]-reg["gain5_pp"])} pp，相对随机为{fmt(c["gain5_pp"]-c["random_gain5_pp"])} pp，与regret分别解读。','']
        records=[]
        for entry in read(root/'predictions_manifest.json')['files']:
            pred=read(root/entry['path']);key=Path(pred['state_path']).stem
            block=read(root/'block_summaries'/f'{key}.json');diag=read(root/'diagnostics'/key/'j1000.json')
            row={k:pred[k] for k in ['n','method','fold','sample_seed','model_seed']};row.update(stability=block['stability'],spectral=diag.get('spectral',{}),projection=diag['last_update']['projection_step_fraction'])
            g=pred['geometry_row']['geometry'];comparison=pred['geometry_row'].get('comparisons_to_raw',{})
            row.update(s_h=g['s_h'],ell_over_s_h=g['ell_over_s_h'],QT_effective=g['effective_distance_stats']['QT']['median'],
              shape_QT=comparison.get('shape_change',{}).get('QT'),nearest_change=comparison.get('nearest_changed_fraction'))
            records.append(row)
        sr=[];geomrows=[]
        for n in (20,50):
            for m in ['R_FROZEN','R_REG','R_DKL_C','R_DKL_U','RAW16']:
                rows=[r for r in records if r['n']==n and r['method']==m]
                counts=[sum(bool(r['stability']['practical_stability']) for r in rows)]+[sum(bool(r['stability']['criteria'][k]) for r in rows) for k in ['objective_small','similarity_stable','mean_prediction_stable','uncertainty_stable']]
                sr.append([n,m,len(rows),*counts])
                geomrows.append([n,m,fmt(nested(rows,'s_h')),fmt(nested(rows,'ell_over_s_h')),fmt(nested(rows,'QT_effective')),fmt(nested(rows,'shape_QT')),fmt(nested(rows,'nearest_change')),fmt(nested(rows,'projection'))])
        lines += ['## 谱约束与实用稳定','',table(['n','方法','实际终点','四项全稳定','NLL稳定','K稳定','均值稳定','SD稳定'],sr),'',
          '稳定性要求700→800、800→900、900→1000三个连续窗口均通过：NLL相对变化≤0.001；训练核和训练+查询无噪声核变化≤1%；训练/查询均值最大变化≤0.001原accuracy；查询潜在SD与模型SD最大变化≤0.001。模型SD含extra，不含查询SE。逐窗口R漂移与outputscale变化另存。稳定不是数学收敛证明，也不能代替留出选优结果；未稳定没有触发加训。','',
          table(['n','方法','s_h','ell/s_h','QT中位d/ell','QT shape/RAW','最近邻更换/RAW','末块投影步比例'],geomrows),'',
          '残差映射为h(x)=x+W2 ReLU(W1x+b1)，264个映射参数及17个回归头参数。W1/W2每次映射更新后按精确SVD截断各奇异值到sqrt(0.5)，不整体缩放、不投影Adam动量；共同训练始终受约束，C继续约束，U分叉后关闭约束。双侧距离界仅针对当前子集scaler定义的16维输入空间；它不保证正确的性能近邻，GP长度尺度仍能使核高度相关。','']
        lines += ['两个n的C/U均为0/30达到全部四项实用稳定条件，因此未见完整稳定性改善。C的末段投影步比例为100%，约束边界持续活跃；非零原始梯度本身不是未收敛证明。C显著减少相对RAW的形状与邻居改变，E1也比RAW低，但这些几何改善不足以保证更好的top5选优。','']
        write(root/'stability_and_geometry_summary.json',{'rows':records,'stability_table':sr,'geometry_table':geomrows,'model_computation':False})
        lines += ['## 判读与限制','']
        for n in (20,50):
            v=b['per_n'][str(n)];means=v['method_means'];c=means['R_DKL_C']
            lines += [f'n={n}的选择规则通过={v["selection_signal"]}，持续约束机制规则通过={v["constraint_signal_pass"]}，后续候选={v["followup_candidate"]}。后续候选附加条件：{json.dumps(v["followup_conditions"],ensure_ascii=False)}；REG比较的模型/抽样种子均值：{json.dumps(v["REG_comparison_seed_means"],ensure_ascii=False)}。']
            if c['regret5_pp']>=means['RAW16']['regret5_pp']-1e-9:
                lines += [f'n={n}的C平均regret5未优于RAW16；这只限制当前固定结构、约束、训练及数据预算下的实用结论，不否定所有DKL。']
            if not v['constraint_signal_pass']:lines += [f'n={n}未获得持续约束较一致改善的证据；新结构或共同起点的整体效果不能全部归因于持续投影。']
            if not v['followup_candidate']:lines += [f'n={n}尚不列为较稳健的后续候选，保留任何局部正向效果及其真实幅度。']
            lines += ['']
    lines += ['5折训练集重叠，多个种子复用相同查询数据；只做描述性配对比较，没有把30条轨迹当作独立样本做t检验或朴素bootstrap。两个n分别判断，不能合并掩盖差异。新16维残差结构与旧8维映射不同，且新回归目标取消了旧embedding范数惩罚，新旧整体差异不单独识别谱约束的因果作用。本轮为DUE启发方案，不是完整论文复现。','',
      '历史结论保留：阶段01工程检查通过；阶段2完成200终点/344000更新，n20未见收益、n50混合；长度补充新增180000更新，1000→4000未改善主要选优及gain。旧G/S由冻结GP预测均值误差指导，本轮DKL联合优化纯数据负MLL，方法不同。','',
      '## 资源、验证与可复现性','',
      f'正式训练及投影 **{ledger["training_seconds"]:.6f}s**；A/B新模型诊断与预测 **{ledger["prediction_seconds"]:.6f}s**；验证 **{ledger["verification_seconds"]:.6f}s**；累计 **{ledger["seconds"]:.6f}/1200s**。合成更新{ledger["verification_updates"]}次另列，不计正式404000更新。失败训练尝试计数{ledger["failed_training_attempts"]}。历史A运行墙钟{hist["A_wall_seconds"]:.6f}s；正式B墙钟{status["formal_wall_seconds"]:.6f}s，合计{hist["A_wall_seconds"]+status["formal_wall_seconds"]:.6f}s。纯统计、代码审阅、传输和打包不算模型预算。', '',
      '每种方法独立使用成本包含其完整共同训练及诊断，再加自身分支训练及诊断；本实验实际账本仅共享共同状态一次。末步预测复用已经计费的诊断结果，未另做预测。成本表同时保留纯更新与模型诊断成本；首对正式n20/n50用于估时且保留在矩阵，没有真实标签试参。','',
      '新增CPU模型验证首轮通过：115次合成更新，覆盖初值恒等、264+17参数、奇异值截断/双侧界、四分支数学/模块隔离、C/U梯度及固定head、独立Cholesky NLL、已知噪声/方差单位、跨80%学习率切换的完整Adam恢复、诊断开关与查询毒化。16项评分/接口合成检查通过，语法和git差异检查通过。未重复运行历史全套测试或真实训练矩阵。','',
      '正式入口使用永久独占formal_run_once.lock，只启动一次fresh run。原始runner的直接--resume对进程强杀后未落盘的1–99步无法确认，因此本轮禁止使用该入口恢复；复现仅在新目录重新运行。已保存的完整checkpoint在合成测试中可精确恢复，这不等同于允许忽略强杀期间未记账成本。实际异常与缺项以status.json/failure.json为准；未出现问题时不会编造失败。','',
      '冻结协议与源码hash见protocol.json、code_manifest.json、frozen_manifest.json。A无真值诊断先封存后评分；B训练只读取每次已选子集和无标签query，以及不含分流效应大小的admission.json。全部B终点统一封存后才进入scoring_B。独立门槛/指标复算与正式状态/更新/预算审计保存在audit/。','',
      '本轮未读取B池/test成绩，未训练或前向运行GNN/VAE，未进行搜索回放、GPU计算、commit/push或阶段3；next_stage_authorized=false。后续GP尺度/先验、CLML、容量或其他约束方案仅可另行提出，不在本轮实施。','',
      '服务器目录：'+str(root.resolve())+'。复现命令见REPRODUCE.md；文件及证据包校验见FILES.sha256、evidence.tar.gz.sha256。']
    write(root/'status.json',status);(root/'README.md').write_text('\n'.join(lines)+'\n')
    start=datetime.datetime.fromisoformat(read(root/'preparation.json')['at_utc'])
    write(root/'report_wall_clock.json',{'prepared_at_utc':start.isoformat(),'report_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_from_prepared_to_report_seconds':(datetime.datetime.now(datetime.timezone.utc)-start).total_seconds(),
        'scope':'includes implementation coordination, approval waits, fitting, scoring, reporting; package/transfer subsequently measured separately',
        'A_plus_B_experiment_wall_seconds':hist['A_wall_seconds']+status['formal_wall_seconds']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
