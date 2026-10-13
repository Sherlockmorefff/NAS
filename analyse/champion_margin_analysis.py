"""Independent evaluation: standard library only; scores/labels never sent to workers."""
import argparse,hashlib,json,math
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);content=json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if p.exists():
        assert p.read_text()==content,'immutable evaluation differs: '+str(p)
    else:
        import os
        tmp=p.with_suffix('.tmp');tmp.write_text(content);os.replace(tmp,p)


def evaluate(o,include_mixed=False):
    o=Path(o);p=read(o/'protocol.json');labels=read(p['A_labels_path']);assert sha(p['A_labels_path'])==p['A_labels_sha']
    y={c:math.fsum(v['validation_seeds'])/3 for c,v in labels.items()};best=max(y.values());champ={c for c,v in y.items() if abs(v-best)<=p['config']['tie_tolerance']}
    starts=read(o/'starts.json');baseline=read(o/'baseline_reuse_manifest.json');results=[];events=[];curves=[];fingerprints={str(Path(p['A_labels_path'])):sha(p['A_labels_path'])};bundles=[]
    for seed in p['config']['search_seeds']:
        paths=[(m,Path(baseline['trajectories'][f'{seed}-{m}']['directory']),True) for m in ['C_GREEDY','BR_FIXED_GREEDY','RANDOM']]
        paths.append(('BR_GAP_GREEDY',o/'trajectories'/f'stage1-start{seed}-s0',False))
        if include_mixed:paths.append(('SELECTED_MIXED',o/'trajectories'/f'stage2-start{seed}-s0',False))
        for method,d,reused in paths:
            done=d/'completed.json';assert done.exists(),str(done);fingerprints[str(done)]=sha(done)
            initial=starts[str(seed)]['initial_ids'];known=set(initial);chosen=[];first=None;firstsource=None;firstbatch=None;running=max(y[c] for c in known)
            curves.append(dict(method=method,seed=seed,new_queries=0,best_validation=running,gap_accuracy=best-running))
            snapshots=[]
            for b in range(p['config']['query_budget']//p['config']['batch_size']):
                sp=d/'batches'/f'b{b:02}'/'selection.json';rp=sp.parent/'reveal.json';s=read(sp);r=read(rp)
                fingerprints[str(sp)]=sha(sp);fingerprints[str(rp)]=sha(rp)
                assert r['selection_sha']==sha(sp) and s['selected_at']<r['revealed_at']
                assert set(s['known_ids'])==known and set(r['labels'])==set(s['selected_ids']) and not known&set(s['selected_ids'])
                assert len(s['selected_ids'])==len(set(s['selected_ids']))==5 and len(s['rows'])==179-len(known)
                assert set(x['candidate_id'] for x in s['rows'])==set(y)-known
                sources=s.get('sources',['random' if method=='RANDOM' else 'greedy']*5)
                for j,(cid,source) in enumerate(zip(s['selected_ids'],sources)):
                    assert r['labels'][cid]==labels[cid]
                    query=5*b+j+1
                    if first is None and cid in champ:first=query;firstsource=source;firstbatch=b
                    score=next(x['score'] for x in s['rows'] if x['candidate_id']==cid)
                    events.append(dict(method=method,seed=seed,batch=b,query_index=query,candidate_id=cid,selection_source=source,version=s['version'],
                        model_hash=s['model_hash'],known_before=len(known),remaining_before=len(s['rows']),remaining_budget=50-5*b,score=score,
                        validation_mean=y[cid],is_pool_champion=cid in champ,selection_sha=sha(sp),selected_at=s['selected_at'],revealed_at=r['revealed_at'],selection_path=str(sp),reused=reused))
                snapshots.append(dict(selection=s,reveal=r,selection_path=str(sp),selection_sha=sha(sp)))
                known.update(s['selected_ids']);chosen.extend(s['selected_ids']);running=max(y[c] for c in known)
                curves.append(dict(method=method,seed=seed,new_queries=5*(b+1),best_validation=running,gap_accuracy=best-running))
            assert len(chosen)==len(set(chosen))==50 and len(known)==70
            row=dict(method=method,seed=seed,reused=reused,initial_best=max(y[c] for c in initial),initial_contains_champion=bool(set(initial)&champ),
                hit_within_50=first is not None,final_best=running,gap_accuracy=best-running,first_hit_query=first if first is not None else '>50',
                first_hit_batch=firstbatch,first_hit_source=firstsource,censored=first is None,new_queries=50,total_known=70)
            results.append(row);bundles.append(dict(method=method,seed=seed,initial_ids=initial,batches=snapshots))
    summary={}
    for method in sorted({r['method'] for r in results}):
        rows=[r for r in results if r['method']==method];assert len(rows)==5
        summary[method]=dict(H=sum(r['hit_within_50'] for r in rows),R=math.fsum(r['gap_accuracy'] for r in rows)/5,mean_final_best=math.fsum(r['final_best'] for r in rows)/5)
    return dict(results=results,events=events,curves=curves,summary=summary,input_fingerprints=fingerprints,pool_best=best,champion_ids=sorted(champ),
        minimal_inputs=dict(A_labels=labels,trajectories=bundles))


def decide(o):
    o=Path(o);p=read(o/'protocol.json');a=evaluate(o,False);c=a['summary']['C_GREEDY'];g=a['summary']['BR_GAP_GREEDY'];tol=p['decision_rule']['accuracy_tie_tolerance']
    if g['H']!=c['H']:selected='gap' if g['H']>c['H'] else 'C';reason='H较大'
    elif abs(g['R']-c['R'])>tol:selected='gap' if g['R']<c['R'] else 'C';reason='H相同，R较小'
    else:selected='C';reason='H和R数值平局，按规则选C'
    result=dict(rule_version=p['decision_rule']['version'],protocol=p['identity'],input_fingerprints=a['input_fingerprints'],
        per_start=[r for r in a['results'] if r['method'] in ['C_GREEDY','BR_GAP_GREEDY']],H_R={m:a['summary'][m] for m in ['C_GREEDY','BR_GAP_GREEDY']},
        selected_loss=selected,selected_greedy='BR_GAP_GREEDY' if selected=='gap' else 'C_GREEDY',reason=reason)
    write(o/'stage1_decision.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();o=args.output
    a=evaluate(o,True);dec=decide(o)
    for name,key in [('trajectory_results','results'),('selection_events','events'),('best_so_far','curves')]:write(o/(name+'.json'),a[key])
    write(o/'analysis_summary.json',dict(summary=a['summary'],pool_best=a['pool_best'],champion_ids=a['champion_ids'],selected_greedy=dec['selected_greedy']))
    write(o/'evaluation_minimal_inputs.json',a['minimal_inputs']);write(o/'evaluation_input_fingerprints.json',a['input_fingerprints'])
    print(json.dumps(a['summary'],ensure_ascii=False))


if __name__=='__main__':main()
