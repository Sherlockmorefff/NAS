"""Root-only targeted cache/state inspection: no encode, prediction or updates."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from readout_core import Budget,budget_limit,cpu_setup
from readout_data import verify_cached_context

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root):
 out=root/'verification/v2_migration_checks.json'
 if out.exists():raise RuntimeError('Targeted migration verification already recorded')
 budget=Budget(root/'budget.json',limit=budget_limit(root));cpu_setup(0);before=budget.as_dict();start=time.monotonic();cases=[];error=None
 try:
  source=read(root/'source_manifest.json');cachem=read(root/'cache_manifest.json');work=read(root/'worklist.json')
  for rel,h in {**source['files'],**cachem['files']}.items():
   if sha(root/rel)!=h:raise RuntimeError('Migrated byte identity differs: '+rel)
  for c in work['contexts']:
   cid=c['context_id'];budget.check();cache=torch.load(root/'caches'/f'{cid}.pt',weights_only=False,map_location='cpu');state=torch.load(root/'source/common'/f'{cid}_COMMON_gp.pt',weights_only=False,map_location='cpu')
   tick=time.monotonic()
   try:
    verify_cached_context(cache);tr,q=cache['training'],cache['query']
    assert cache['identity']['context_id']==cid and len(tr['ids'])==c['n']
    assert torch.equal(tr['h0'],state['history_embedding']) and torch.equal(tr['y_standardized'],state['y']) and torch.equal(tr['known_variance_standardized'],state['known'])
    assert state['scheduler']['completed']==300 and state['scheduler']['block']==cid+'_COMMON_gp'
    assert all(v.dtype==torch.float64 for v in state['mapper'].values())
    assert state['mapper']['W1.weight'].shape==(8,16) and state['mapper']['W2.weight'].shape==(16,8)
    assert state['mapper']['head.weight'].shape==(1,16) and state['mapper']['head.bias'].shape==(1,)
    cases.append({'context_id':cid,'pass':True,'n':c['n'],'query_count':len(q['ids']),'cache_sha256':cache['identity']['cache_sha256'],'source_state_sha256':source['files'][f'source/common/{cid}_COMMON_gp.pt'],'no_reencoding_no_prediction':True})
   finally:budget.charge(time.monotonic()-tick,'verification')
  inherited=read(root/'validation_parent/source_cache_checks.json')
  assert inherited['contexts_passed']==60 and all(c['pass'] and c['const_zero']['pass'] for c in inherited['cases'])
  assert budget.formal_updates==0 and budget.synthetic_updates==182 and budget.synthetic_optimizer_calls==182
  budget.check()
 except Exception as e:error=repr(e)
 finally:budget.persist()
 result={'pass':error is None and len(cases)==60,'scope':'60 exact saved state and cached tensor shape/dtype/identity checks; no scientific recomputation needed after path/budget-only migration','cases':cases,'parent_zero_step_checks_reused':60,'new_gp_predictions':0,'new_optimizer_calls':0,'new_verification_seconds':budget.seconds-before['seconds'],'resources_before':before,'resources_after':budget.as_dict(),'wall_seconds':time.monotonic()-start,'failure':error,'new_budget_limit':budget.limit}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['pass','new_verification_seconds','new_gp_predictions','new_optimizer_calls','failure']}));return 0 if result['pass'] else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);raise SystemExit(main(p.parse_args().root))
