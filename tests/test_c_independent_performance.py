import json,math,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from analyse.c_independent_validation.performance import plans,validate_public,SEEDS,seal
from analyse.c_independent_validation.oracle import Ledger
from analyse.structured_kernel_development_replay.learner import build_training

def test_initial_and_random_shared_and_unique():
 ids=[f'candidate_{i:03}' for i in range(176)];initial,random=plans(ids)
 for seed in SEEDS:
  expected=np.random.default_rng(np.random.SeedSequence([seed,0])).permutation(sorted(ids)).tolist()[:50]
  assert initial[str(seed)]['ids']==random[str(seed)]['initial_ids']==expected
  assert len(set(expected+random[str(seed)]['query_ids']))==71
 with pytest.raises(ValueError):plans(ids[:71])

def test_oracle_budget_release_and_replay(tmp_path):
 ids=[f'c{i:03}' for i in range(176)];ini,rand=plans(ids);labels={c:{'mean':.5+i/1000,'se':.01} for i,c in enumerate(ids)};l=Ledger(tmp_path/'ledger.sqlite',labels,ini,rand);tr=f'{SEEDS[0]}_W-I'
 with pytest.raises(ValueError):l.release(tr,1,ini[str(SEEDS[0])]['ids'][1])
 for slot,cid in enumerate(ini[str(SEEDS[0])]['ids']):assert set(l.release(tr,slot,cid))=={'candidate_id','mean','se'}
 last=ini[str(SEEDS[0])]['ids'][49];assert l.release(tr,49,last)['candidate_id']==last
 with pytest.raises(ValueError):l.release(tr,50,last)
 with pytest.raises(ValueError):l.release(tr,71,ids[-1])
 assert l.summary()['logical_releases']==50 and l.summary()['repeated_requests']==1

def test_public_pool_rejects_unqueried_targets():
 p={'ids':['x'],'x_raw':[[0.]*16],'wl':[[1.,2.]],'architecture_ids':['a']};validate_public(p)
 for key in ('mean','se','champion','labels'):
  with pytest.raises(ValueError):validate_public({**p,key:[.9]})

def test_scaler_only_uses_released_T():
 p={'ids':['a','b','c'],'x_raw':[[1.]*16,[3.]*16,[10000.]*16],'wl':[[1.,0.],[0.,1.],[10000.,10000.]]}
 observations=[{'candidate_id':'a','mean':.6,'se':.01},{'candidate_id':'b','mean':.8,'se':.02}]
 t=build_training(p,observations,1,'W-I');assert t['scaler']['fit_ids']==['a','b'];assert np.allclose(t['scaler']['x_mean'],2.)
 assert np.isclose(t['scaler']['y_mean'],.7) and np.isclose(t['scaler']['y_scale'],.1)
 assert np.allclose(t['known_variance_standardized'],[.01,.04])

def test_three_seed_SE_is_mean_uncertainty():
 x=np.array([.7,.8,.9]);se=x.std(ddof=1)/math.sqrt(3)
 assert np.isclose(se**2,.01/3) and not np.isclose(se**2,.01/9)

def test_seal_recovery_preserves_first_timestamp(tmp_path):
 p=tmp_path/'sealed.json';seal(p,{'values':[1,2],'sealed_at':'first'});before=p.read_bytes();seal(p,{'values':[1,2],'sealed_at':'second'});assert p.read_bytes()==before
 with pytest.raises(ValueError):seal(p,{'values':[2,3],'sealed_at':'third'})

def test_learner_cannot_open_full_cache_or_A(tmp_path):
 root=tmp_path/'results/run';root.mkdir(parents=True);(root/'private').mkdir();(root/'private/C_validation_cache.json').write_text('{}')
 script='from pathlib import Path\nfrom analyse.c_independent_validation.oracle import learner_guard\nr=Path('+repr(str(root))+')\nlearner_guard(r,Path('+repr(str(tmp_path))+'))\ntry:\n (r/"private/C_validation_cache.json").read_text()\nexcept PermissionError:\n print("DENIED")\nelse:\n raise AssertionError("leak")\n'
 result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True);assert result.returncode==0 and 'DENIED' in result.stdout

def test_guard_allows_atomic_progress_but_not_cache_tmp(tmp_path):
 root=tmp_path/'results/run';root.mkdir(parents=True)
 script='from pathlib import Path\nfrom analyse.c_independent_validation.oracle import learner_guard\nfrom analyse.c_independent_validation.common import write\nr=Path('+repr(str(root))+')\nlearner_guard(r,Path('+repr(str(tmp_path))+'))\nwrite(r/"P5_progress.json",{"n":51})\ntry:\n write(r/"private/C_validation_cache.json",{})\nexcept PermissionError:\n print("DENIED")\nelse:\n raise AssertionError("leak")\n'
 result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
 assert result.returncode==0 and 'DENIED' in result.stdout and (root/'P5_progress.json').is_file()
