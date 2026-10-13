"""Frozen random orders followed by a trusted, A-only cache oracle/scorer."""
import csv
import json
import sqlite3
from pathlib import Path
import numpy as np
from .common import *


class RandomOracle:
    def __init__(self,root,labels,manifest):
        self.root,self.labels,self.manifest=root,labels,manifest
        self.db=sqlite3.connect(root/'random_oracle.sqlite')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS observations(seed INTEGER,slot INTEGER,cid TEXT,payload TEXT,PRIMARY KEY(seed,slot),UNIQUE(seed,cid))')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY,seed INTEGER,slot INTEGER,cid TEXT,replay INTEGER,time TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS loads(id INTEGER PRIMARY KEY,physical_opens INTEGER,time TEXT)')
        self.db.commit()

    def release(self,seed,slot,cid):
        order=self.manifest[str(seed)]['initial_ids']+self.manifest[str(seed)]['query_ids']
        if slot<0 or slot>=71 or cid!=order[slot]:raise ValueError('Oracle request violates sealed random order/budget')
        old=self.db.execute('SELECT cid,payload FROM observations WHERE seed=? AND slot=?',(seed,slot)).fetchone()
        if old and old[0]!=cid:raise ValueError('Changed idempotency token')
        if not old and self.db.execute('SELECT COUNT(*) FROM observations WHERE seed=?',(seed,)).fetchone()[0]!=slot:raise ValueError('Nonsequential random release')
        value=json.loads(old[1]) if old else {'candidate_id':cid,**self.labels[cid]}
        with self.db:
            if not old:self.db.execute('INSERT INTO observations VALUES(?,?,?,?)',(seed,slot,cid,json.dumps(value)))
            self.db.execute('INSERT INTO calls(seed,slot,cid,replay,time) VALUES(?,?,?,?,?)',(seed,slot,cid,int(old is not None),now()))
        return value


def run(root,repo):
    if (root/'M2_summary.json').exists() and read_json(root/'M2_summary.json')['state']=='COMPLETE':return read_json(root/'M2_summary.json')
    set_status(root,'M2','RUNNING')
    manifest=read_json(root/'random_query_manifest.json')
    frozen=read_json(root/'run_identity.json')
    if sha256(root/'random_query_manifest.json')!=frozen['inputs']['random_query_manifest.json']:raise ValueError('Random manifest changed before labels')
    # Only this trusted phase sees the complete A-only performance dictionary.
    from analyse.structured_kernel_phase1.data import read_A_labels,SOURCE
    import sys
    counts={'physical_A_label_file_opens':0}
    label=(repo/SOURCE/'private/evaluation_A.json').resolve()
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(args[0]).resolve()
            if path==label:counts['physical_A_label_file_opens']+=1
            if path.name=='evaluation_B.json' or ('seed_results' in path.name and path.is_relative_to(repo/'results')):raise PermissionError('Mixed/B performance forbidden')
    sys.addaudithook(audit)
    labels=read_A_labels(repo);oracle=RandomOracle(root,labels,manifest)
    with oracle.db:oracle.db.execute('INSERT INTO loads(physical_opens,time) VALUES(?,?)',(counts['physical_A_label_file_opens'],now()))
    optimum=max(v['mean'] for v in labels.values());curves=[];trajectories=[]
    for seed in SEEDS:
        plan=manifest[str(seed)];order=plan['initial_ids']+plan['query_ids']
        existing={s:json.loads(payload) for s,payload in oracle.db.execute('SELECT slot,payload FROM observations WHERE seed=?',(seed,))}
        obs=[]
        for slot,cid in enumerate(order):
            value=existing.get(slot) or oracle.release(seed,slot,cid)
            if value['candidate_id']!=cid:raise ValueError('Stored random order mismatch')
            obs.append(value)
        seal(root/'random_observations'/f'{seed}.json',obs)
        best=[]
        for query in range(22):
            score=max(o['mean'] for o in obs[:50+query]);best.append(score)
            curves.append({'seed':seed,'method':'Random','new_queries':query,'kind':'initial' if query==0 else 'random',
                'best_validation_mean':score,'best_pp':100*score,'cached_simple_regret_pp':100*(optimum-score)})
        winner=min(obs,key=lambda o:(-o['mean'],o['candidate_id']))
        trajectories.append({'seed':seed,'initial_best_pp':100*best[0],'n70_best_pp':100*best[20],'n71_best_pp':100*best[21],
            'slot21_increment_pp':100*(best[21]-best[20]),'champion_hit_n71':any(o['candidate_id']==CHAMPION for o in obs),
            'initial_contains_champion':CHAMPION in plan['initial_ids'],'returned_candidate_id':winner['candidate_id'],
            'regret_integral_q0_20':float(np.trapezoid(100*(optimum-np.asarray(best[:21])))/20)})
    old=list(csv.DictReader((repo/REPLAY/'paired_comparisons.csv').open()))
    paired=[]
    for random,row in zip(trajectories,old):
        seed=random['seed'];assert seed==int(row['seed'])
        p=dict(seed=seed,initial_best_pp=random['initial_best_pp'],initial_contains_champion=random['initial_contains_champion'],
            Random_n70_pp=random['n70_best_pp'],Random_n71_pp=random['n71_best_pp'],Random_champion_hit=random['champion_hit_n71'],
            Random_regret_integral=random['regret_integral_q0_20'])
        for method,key in [('W-I','WI'),('RAW16','RAW16')]:
            for n in (70,71):
                p[f'{key}_n{n}_pp']=float(row[f'n{n}_{key}_pp'])
                p[f'{key}_minus_Random_n{n}_pp']=p[f'{key}_n{n}_pp']-random[f'n{n}_best_pp']
            p[key+'_champion_hit']=abs(p[key+'_n71_pp']-100*optimum)<1e-12
            p[key+'_regret_integral']=float(row[key+'_regret_integral'])
        paired.append(p)
    oldcurves=list(csv.DictReader((repo/REPLAY/'curves.csv').open()))
    allcurves=[{'seed':int(r['seed']),'method':r['method'],'new_queries':int(r['new_queries']),'kind':r['kind'],
        'best_validation_mean':float(r['best_validation_mean']),'best_pp':float(r['best_pp']),'cached_simple_regret_pp':float(r['cached_simple_regret_pp'])} for r in oldcurves]+curves
    indices=np.random.default_rng(2026100531).integers(0,10,size=(2000,10))
    comparisons={}
    for method in ('WI','RAW16'):
        for n in (70,71):
            vals=np.array([p[f'{method}_minus_Random_n{n}_pp'] for p in paired])
            comparisons[f'{method}_minus_Random_n{n}']={'mean_pp':float(vals.mean()),'CI95_pp':np.quantile(vals[indices].mean(1),[.025,.975]).tolist(),
                'wins':int((vals>0).sum()),'ties':int((vals==0).sum()),'losses':int((vals<0).sum())}
    calls=[dict(call_id=i,seed=s,slot=slot,candidate_id=c,idempotent_replay=bool(replay),time=t) for i,s,slot,c,replay,t in oracle.db.execute('SELECT * FROM calls ORDER BY id')]
    with (root/'random_oracle_access_log.jsonl').open('w') as f:
        for call in calls:f.write(json.dumps(call)+'\n')
    loads=list(oracle.db.execute('SELECT physical_opens,time FROM loads ORDER BY id'))
    count=oracle.db.execute('SELECT COUNT(*),COUNT(DISTINCT cid) FROM observations').fetchone()
    assert count[0]==710
    summary={'state':'COMPLETE','trajectories':10,'initial_logical_observations':500,'new_random_queries':210,
        'logical_releases':count[0],'unique_candidates':count[1],'oracle_calls':len(calls),'idempotent_replays':sum(c['idempotent_replay'] for c in calls),
        'physical_A_label_file_opens':sum(x[0] for x in loads),'cache_load_sessions':len(loads),'comparisons':comparisons,
        'champion_hits':{m:sum(p[m+'_champion_hit'] for p in paired) for m in ('WI','RAW16','Random')},
        'regret_integral_mean':{m:float(np.mean([p[m+'_regret_integral'] for p in paired])) for m in ('WI','RAW16','Random')},
        'new_GP_fits':0,'new_GNN_tasks':0,'new_gate':None,'independent_confirmation':False}
    oracle.db.close()
    write_csv(root/'random_trajectories.csv',trajectories);write_csv(root/'random_paired_comparisons.csv',paired);write_csv(root/'random_curves.csv',allcurves)
    write_json(root/'M2_summary.json',summary);set_status(root,'M2','COMPLETE',trajectories=10,new_queries=210)
    plot(root,allcurves)
    return summary


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for method in ('W-I','RAW16','Random'):
        values=[[r['cached_simple_regret_pp'] for r in rows if r['method']==method and r['seed']==s] for s in SEEDS]
        ax.plot(range(22),np.mean(values,axis=0),marker='.',label=method)
    ax.axvspan(20,21,color='gray',alpha=.15);ax.set_ylim(bottom=0)
    ax.set(xlabel='New queries (slot21: GP greedy / Random random)',ylabel='Mean cached simple regret (pp)',title='Fixed A development pool; same initial50 and total71')
    ax.legend();fig.tight_layout();fig.savefig(root/'M2_random_reference.png',dpi=160);plt.close(fig)
