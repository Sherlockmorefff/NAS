"""Independent endpoint evaluation and forward-only integrity checks."""
import json,math
from pathlib import Path
from analyse.relative_metric_pilot import read,write,file_hash,digest,load,tensor_hash
from analyse.ensemble_conditional_rules import stage1_rule,stage2_rule

def evaluate_stage(o,stage):
    p=read(o/'protocol.json');labels=read(Path(p['A_labels_path']));assert file_hash(Path(p['A_labels_path']))==p['A_labels_sha']
    y={c:math.fsum(v['validation_seeds'])/3 for c,v in labels.items()};star=max(y.values());champ={c for c,v in y.items() if abs(v-star)<=1e-12}
    starts=read(o/f'starts_stage{stage}.json');paths=[]
    if stage==1:
        for key,r in read(o/'baseline_reuse_manifest.json')['trajectories'].items():
            seed,method=key.split('-',1);paths.append((int(seed),method,Path(r['directory']),True))
    for t in read(o/'tasks.json'):
        if t['stage']==stage:
            d=o/'trajectories'/t['name'];done=read(d/'completed.json');paths.append((t['seed'],done['method'],d,False))
    results=[];events=[];curves=[];minimal=[];fingerprints={p['A_labels_path']:p['A_labels_sha']}
    for seed,method,d,reused in sorted(paths,key=lambda r:(r[0],r[1])):
        initial=starts[str(seed)]['initial_ids'];known=set(initial);initial_hit=bool(known&champ);chosen=[];first=None;origin=None;batches=[];regrets=[]
        def point(q):
            best=max(y[c] for c in known);regret=star-best;regrets.append(regret)
            curves.append(dict(stage=stage,seed=seed,method=method,new_queries=q,best_validation=best,gap_accuracy=regret));return best
        firstbest=point(0);fingerprints[str(d/'completed.json')]=file_hash(d/'completed.json')
        for b in range(10):
            sp=d/'batches'/f'b{b:02}'/'selection.json';rp=sp.parent/'reveal.json';s=read(sp);r=read(rp)
            assert r['selection_sha']==file_hash(sp) and s['selected_at']<r['revealed_at']
            assert set(s['known_ids'])==known and set(r['labels'])==set(s['selected_ids']) and len(set(s['selected_ids']))==5 and not known&set(s['selected_ids'])
            assert len(s['rows'])==179-len(known) and {v['candidate_id'] for v in s['rows']}==set(y)-known
            sources=s.get('sources',['random' if method=='RANDOM' else 'member0']*5);lookup={v['candidate_id']:v for v in s['rows']}
            for j,(c,source) in enumerate(zip(s['selected_ids'],sources)):
                assert r['labels'][c]==labels[c];chosen.append(c);row=lookup[c]
                if first is None and c in champ:first=len(chosen);origin=source
                pred=row.get('mean',row.get('score')) if method!='RANDOM' else None
                events.append(dict(stage=stage,seed=seed,method=method,batch=b,query_index=5*b+j+1,paid_queries=5*(b+1),candidate_id=c,source=source,
                    prediction_before_reveal=pred,validation_mean=y[c],absolute_error=None if pred is None else abs(pred-y[c]),
                    member_scores=row.get('member_scores'),member_ranks=row.get('member_ranks'),sd=row.get('sd'),is_pool_champion=c in champ,
                    version=s['version'],selection_sha=file_hash(sp),selected_at=s['selected_at'],revealed_at=r['revealed_at'],reused=reused))
            fingerprints[str(sp)]=file_hash(sp);fingerprints[str(rp)]=file_hash(rp)
            batches.append(dict(selection=s,reveal=r,selection_sha=file_hash(sp),selection_path=str(sp)))
            known.update(s['selected_ids']);last=point(5*(b+1))
        assert len(chosen)==len(set(chosen))==50 and len(known)==70
        results.append(dict(stage=stage,seed=seed,method=method,reused=reused,initial_best=firstbest,initial_contains_champion=initial_hit,
            hit=not initial_hit and first is not None,queried_champion=first is not None,final_best=last,gap_accuracy=star-last,AUC_regret=math.fsum(regrets)/11,
            first_hit_query=first if first is not None else '>50',first_hit_batch=(first-1)//5 if first is not None else None,
            first_hit_paid_queries=5*((first-1)//5+1) if first is not None else None,first_hit_source=origin,censored=first is None,new_queries=50,total_known=70))
        minimal.append(dict(stage=stage,seed=seed,method=method,reused=reused,initial_ids=initial,batches=batches))
    metrics={}
    for method in sorted({r['method'] for r in results}):
        rows=[r for r in results if r['method']==method]
        metrics[method]=dict(H=sum(r['hit'] for r in rows),R=math.fsum(r['gap_accuracy'] for r in rows)/len(rows),AUC_regret=math.fsum(r['AUC_regret'] for r in rows)/len(rows),
            mean_final_best=math.fsum(r['final_best'] for r in rows)/len(rows),starts=len(rows))
    if stage==1:
        assert all(not r['initial_contains_champion'] for r in results) and len(results)==20
        assert metrics['C_GREEDY']['H']==1 and abs(metrics['C_GREEDY']['mean_final_best']-.8066666666666667)<1e-12
        assert metrics['RANDOM']['H']==2 and abs(metrics['RANDOM']['mean_final_best']-.8088)<1e-12
    else:assert len(results)==60
    return dict(results=results,events=events,curves=curves,metrics=metrics,trajectories=minimal,input_fingerprints=fingerprints,A_labels=labels,champion_ids=sorted(champ),pool_best=star)

def seal_stage1(o,a):
    assert read(o/'stage1_validation.json')['passed'];p=read(o/'protocol.json')
    r=stage1_rule(a['metrics']);r.update(per_start=a['results'],protocol=p['identity'],sources=p['sources'],config_sha=file_hash(o/'config.json'),input_fingerprints=a['input_fingerprints'])
    write(o/'stage1_decision.json',r);return r

def seal_final(o,one,two):
    if two is None:r=dict(status='STAGE1_NOT_QUALIFIED',winner=None,ridge_required=True,stage2_executed=False,reason='Neither ensemble meets H>=H_C+1 and R<=R_C+1e-12',opportunities=None,paired_tests={})
    else:
        assert read(o/'stage2_validation.json')['passed'];r=stage2_rule(two['results'],one['winner']);r.update(stage2_executed=True,per_start=two['results'],input_fingerprints=two['input_fingerprints'])
    r.update(stage1_decision_sha=file_hash(o/'stage1_decision.json'),protocol=read(o/'protocol.json')['identity'])
    write(o/'final_decision.json',r);return r

def export_results(o):
    one=evaluate_stage(o,1);stages=[one]
    if read(o/'final_decision.json')['stage2_executed']:stages.append(evaluate_stage(o,2))
    for n,key in [('trajectory_results','results'),('selection_events','events'),('best_so_far','curves')]:write(o/(n+'.json'),[v for a in stages for v in a[key]])
    write(o/'evaluation_minimal_inputs.json',dict(A_labels=one['A_labels'],trajectories=[v for a in stages for v in a['trajectories']]))
    write(o/'analysis_summary.json',dict(stages={str(i+1):a['metrics'] for i,a in enumerate(stages)},pool_best=one['pool_best'],champion_ids=one['champion_ids']))

def validate_stage(o,stage):
    import torch
    from surrogate.gp_guidance_replay_engine import scaler_from_initial,input_tensors
    from surrogate.ensemble_conditional import score_ensemble,choose
    from analyse.ensemble_conditional_run import protocol
    p=protocol(o);cfg=p['config'];features=read(o/'features.json');counts=dict(tasks=0,members_events=0,steps=0,snapshots=0,batches=0,initial_member0_matches_oldC=0)
    checkpoints=[]
    for t in read(o/'tasks.json'):
        if t['stage']!=stage:continue
        d=o/'trajectories'/t['name'];done=read(d/'completed.json');assert done['state_sha']==file_hash(d/'state.pt');st=load(d/'state.pt');method=done['method']
        initial=read(o/t['label_file']);scaler=scaler_from_initial(features,t['initial_ids'],initial);assert st['scaler']==scaler and len(st['known'])==70
        previous=[load(o/f'initial_weights/member{j}.pt')['model'] for j in range(t['members'])];versions={}
        for q in range(0,50,10) if t['members'] else []:
            event='initial' if q==0 else f'new{q}';steps=300 if q==0 else 200;ids=sorted(set(t['initial_ids'])|set(st['query_order'][:q]));states=[]
            for j in range(t['members']):
                f=d/'events'/event/f'member{j}';zero=load(f/'step000.pt');end=load(f/'latest.pt')
                assert not zero['optimizer']['state'] and zero['step']==0 and tensor_hash(zero['model'])==tensor_hash(previous[j])
                assert end['step']==steps and end['known_ids']==ids and end['scaler']==scaler and end['model_hash']==tensor_hash(end['model'])
                assert all(float(v['step'])==steps for v in end['optimizer']['state'].values())
                logs=[json.loads(s) for s in (f/'steps.jsonl').read_text().splitlines()]
                assert [r['completed_steps'] for r in logs]==list(range(1,steps+1))
                assert all(r['pre_completed_steps']==r['completed_steps']-1 and r['known_ids_hash']==digest(ids) and r['scaler_hash']==digest(scaler) and all(math.isfinite(r[k]) for k in ['total','regression','size']) for r in logs)
                for n in [0,50,100,steps]:
                    snap=load(f/f'step{n:03}.pt');assert snap['step']==n and snap['known_ids']==ids and snap['scaler']==scaler
                    assert snap['model_hash']==tensor_hash(snap['model']);assert snap['timing']==('event_start' if n==0 else 'post_update');counts['snapshots']+=1
                if stage==1 and q==0 and j==0:
                    old=read(o/'baseline_reuse_manifest.json')['trajectories'][f"{t['seed']}-C_GREEDY"]
                    assert tensor_hash(end['model'])==tensor_hash(load(Path(old['directory'])/'events/initial/latest.pt')['model']);counts['initial_member0_matches_oldC']+=1
                states.append(end['model']);counts['members_events']+=1;counts['steps']+=steps
            previous=states;versions[q//10]=states
        known=dict(initial);chosen=[];origin=[];prior=None
        for b in range(10):
            f=d/'batches'/f'b{b:02}';s=read(f/'selection.json');r=read(f/'reveal.json')
            assert s['known_ids']==sorted(known) and s['known_labels_hash']==digest(known) and r['selection_sha']==file_hash(f/'selection.json') and s['selected_at']<r['revealed_at']
            assert set(r['labels'])==set(s['selected_ids']) and not set(s['selected_ids'])&set(known)
            if t['members']:
                rows=score_ensemble(versions[b//2],input_tensors(features,scaler,known));assert rows==s['rows']
                assert s['model_hashes']==[tensor_hash(w) for w in versions[b//2]]
            else:rows=s['rows']
            choice=choose(rows,known,method,stage,t['seed'],b,cfg);assert all(s[k]==v for k,v in choice.items())
            if method in ['E_MEAN','C_GREEDY'] and b%2:assert prior['selected_ids']+s['selected_ids']==[v['candidate_id'] for v in prior['rows'][:10]]
            assert read(f/'aggregate.json')['query_count']==(b+1)*5
            prior=s;known.update(r['labels']);chosen.extend(s['selected_ids']);origin.extend(s['sources']);counts['batches']+=1
        assert known==st['known'] and chosen==st['query_order'] and len(set(chosen))==50
        if method=='E_NOMINATE':assert origin.count('mean')==40 and origin.count('nomination')==10
        assert all(st['member_done'])
        for f in d.rglob('*.pt'):checkpoints.append(dict(path=str(f),sha256=file_hash(f),bytes=f.stat().st_size))
        counts['tasks']+=1
    if stage==1:
        d=o/'trajectories'/f"s1-{cfg['stage1_seeds'][0]}-E_MEAN";f=d/'batches/b00';sha=file_hash(f/'selection.json')
        assert read(f/'pause-selected.json')['selection_sha']==read(f/'pause-revealed.json')['selection_sha']==read(f/'reveal.json')['selection_sha']==sha
        assert read(d/'events/initial/member1/pause-member.json')['completed_steps']==2
        # Member 0 was complete before member 1 paused, and was never physically repeated.
        physical=[json.loads(s) for s in (d/'events/initial/member0/physical.jsonl').read_text().splitlines()]
        assert sum(r['phase']=='computed' for r in physical)==300
    write(o/f'stage{stage}_validation.json',dict(passed=True,counts=counts,forward_only=True,training_updates=0,recovery_boundaries_checked=stage==1))
    write(o/f'checkpoint_manifest_stage{stage}.json',checkpoints)
