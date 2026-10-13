"""Label-free ensemble scoring and deterministic nomination; no uncertainty acquisition."""
import hashlib,math
import numpy as np
from analyse.relative_metric_pilot import digest
from surrogate.gp_guidance_replay_engine import reload_model,score_candidates

def hash_order(ids,prefix):
    return sorted(ids,key=lambda c:(hashlib.sha256((prefix+c).encode('utf-8')).hexdigest(),c))

def score_ensemble(states,data):
    members=[]
    for state in states:
        rows,_,_=score_candidates(reload_model(state),data)
        members.append({r['candidate_id']:r['score'] for r in rows})
    return combine_scores(members)

def combine_scores(members):
    ids=sorted(members[0]);assert all(set(m)==set(ids) for m in members)
    ranks=[{c:i+1 for i,c in enumerate(sorted(ids,key=lambda c:(-m[c],c)))} for m in members]
    rows=[]
    for c in ids:
        scores=[m[c] for m in members];mean=math.fsum(scores)/len(scores)
        sd=math.sqrt(math.fsum((s-mean)**2 for s in scores)/(len(scores)-1)) if len(scores)>1 else None
        assert all(math.isfinite(x) for x in scores)
        rows.append(dict(candidate_id=c,member_scores=scores,mean=mean,sd=sd,member_ranks=[r[c] for r in ranks]))
    rows.sort(key=lambda r:(-r['mean'],r['candidate_id']))
    for i,r in enumerate(rows):r['mean_rank']=i+1
    return rows

def choose(rows,known,method,stage,seed,batch,cfg):
    ids=[r['candidate_id'] for r in rows]
    assert len(ids)==len(set(ids)) and not set(ids)&set(known)
    if method=='RANDOM':
        order=hash_order(ids,f"{cfg['random_prefix']}:{seed}:")
        return dict(selected_ids=order[:cfg['batch_size']],sources=['random']*5,nomination=None)
    assert rows==sorted(rows,key=lambda r:(-r['mean'],r['candidate_id']))
    if method in ['C_GREEDY','E_MEAN']:return dict(selected_ids=ids[:5],sources=['member0' if method=='C_GREEDY' else 'mean']*5,nomination=None)
    assert method=='E_NOMINATE' and all(len(r['member_scores'])==5 for r in rows)
    top=ids[:4];eligible=sorted(set(ids)-set(top));lookup={r['candidate_id']:r for r in rows}
    key=f"{cfg['nominate_prefix']}:{stage}:{seed}:{batch}";derived=int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest(),'big')
    member=int(np.random.Generator(np.random.PCG64(derived)).integers(0,5))
    cid=min(eligible,key=lambda c:(-lookup[c]['member_scores'][member],c))
    return dict(selected_ids=top+[cid],sources=['mean']*4+['nomination'],nomination=dict(key=key,derived_seed=str(derived),member=member,
        eligible_ids_sha256=digest(eligible),eligible_count=len(eligible),candidate_id=cid))
