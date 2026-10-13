"""Predeclared research decisions. Standard library only."""
import math
TOL=1e-12

def stage1_rule(metrics):
    c=metrics['C_GREEDY'];eligible=[m for m in ['E_MEAN','E_NOMINATE'] if metrics[m]['H']>=c['H']+1 and metrics[m]['R']<=c['R']+TOL]
    winner=None
    for m in eligible:
        if winner is None:winner=m;continue
        a,b=metrics[m],metrics[winner]
        if a['H']!=b['H']:
            if a['H']>b['H']:winner=m
        elif abs(a['R']-b['R'])>TOL:
            if a['R']<b['R']:winner=m
        elif abs(a['AUC_regret']-b['AUC_regret'])>TOL:
            if a['AUC_regret']<b['AUC_regret']:winner=m
    return dict(status='STAGE1_QUALIFIED' if winner else 'STAGE1_NOT_QUALIFIED',qualified=eligible,winner=winner,
        next_step='stage2_20_starts' if winner else 'ridge_50_solves',rule_version='H_increment1_R_nondecrease_then_H_R_AUC_E_MEAN_v1',
        metrics=metrics,threshold=dict(H_min=c['H']+1,R_max=c['R']+TOL),comparison_order=['H desc','R asc','AUC_regret asc','E_MEAN'])

def paired(m,c):
    b=sum(x['hit'] and not y['hit'] for x,y in zip(m,c));d=sum(not x['hit'] and y['hit'] for x,y in zip(m,c));n=b+d
    numerator=sum(math.comb(n,k) for k in range(b,n+1));denominator=2**n
    p=numerator/denominator if n else 1.
    return dict(b=b,c=d,p_numerator=numerator,p_denominator=denominator,p=p,p_adj=min(1.,2*p))

def stage2_rule(rows,winner):
    seeds=sorted({r['seed'] for r in rows});lookup={(r['method'],r['seed']):r for r in rows}
    assert len(seeds)==20 and len(rows)==60
    j=[s for s in seeds if not lookup[winner,s]['initial_contains_champion']]
    for s in seeds:assert len({lookup[m,s]['initial_contains_champion'] for m in [winner,'C_GREEDY','RANDOM']})==1
    metrics={m:dict(H=sum(lookup[m,s]['hit'] for s in j),R=math.fsum(lookup[m,s]['gap_accuracy'] for s in j)/len(j) if j else None) for m in [winner,'C_GREEDY','RANDOM']}
    tests={m:paired([lookup[winner,s] for s in j],[lookup[m,s] for s in j]) for m in ['C_GREEDY','RANDOM']}
    M,C,R=[metrics[m] for m in [winner,'C_GREEDY','RANDOM']]
    if len(j)<10:status='INSUFFICIENT_OPPORTUNITIES'
    elif M['H']>C['H'] and M['H']>R['H'] and all(t['p_adj']<=.05 for t in tests.values()) and M['R']<=C['R']+TOL:status='A_POOL_START_ROBUSTNESS_PASS'
    elif M['H']>C['H'] and M['R']<=C['R']+TOL:status='A_POOL_TREND_INCONCLUSIVE'
    else:status='A_POOL_PRIMARY_NOT_MET'
    return dict(status=status,J=j,opportunities=len(j),metrics=metrics,paired_tests=tests,winner=winner,
        ridge_required=status=='A_POOL_PRIMARY_NOT_MET',rule_version='paired_exact_one_sided_Bonferroni_two_v1')
