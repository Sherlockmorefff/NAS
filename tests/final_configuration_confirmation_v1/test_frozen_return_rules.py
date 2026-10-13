import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import numpy as np
import pytest
from analyse.final_configuration_confirmation_v1.common import digest,exact_mean,ordered_observations,write,read,sha
from analyse.final_configuration_confirmation_v1.scheduler import Queue
from analyse.final_configuration_confirmation_v1.analysis import bootstrap,selected_rule,score_gate
from analyse.final_configuration_confirmation_v1.monitor import due,recheck,estimate
from analyse.final_configuration_confirmation_v1.search import QueryLedger

def test_exact_ties_and_own_top5():
    configs={str(i):{'normalized_config_id':str(9-i)} for i in range(8)}
    observations=[{'candidate_id':str(i),'query_correct_sum':1000+i,'query_denominator':1500} for i in range(6)]
    assert set(o['candidate_id'] for o in ordered_observations(observations,configs)[:5])==set('12345')
    observations[0]['query_correct_sum']=1005
    assert ordered_observations(observations,configs)[0]['candidate_id']=='5'
    assert exact_mean([{'correct':401,'denominator':500},{'correct':402,'denominator':500}])==Fraction(803,1000)

def test_oracle_budget_sequential_and_replay(tmp_path):
    labels={str(i):{'candidate_id':str(i),'mean':.5,'se':.01} for i in range(100)}
    initial={'0':[str(i) for i in range(50)]}
    ledger=QueryLedger(tmp_path/'q.sqlite',labels,initial)
    with pytest.raises(ValueError):ledger.release('g00_U',1,'1')
    for i in range(71):ledger.release('g00_U',i,str(i))
    assert ledger.release('g00_U',70,'70')['candidate_id']=='70'
    with pytest.raises(ValueError):ledger.release('g00_U',71,'71')
    with pytest.raises(ValueError):ledger.release('g00_SM',61,'61')
    with pytest.raises(ValueError):ledger.release('g00_U',70,'69')
    assert ledger.summary()['replays']==1

def test_two_claimants_cannot_duplicate_task(tmp_path):
    path=tmp_path/'queue.sqlite';q=Queue(path);q.add({'task_id':'synthetic','seed':1})
    def claim(number):
        local=Queue(path); result=local.claim(number,{'uuid':'gpu'+str(number),'index':number-1});local.db.close();return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(claim,[1,2]))
    assert sum(r is not None for r in results)==1
    assert sum(r['reserved'] for r in q.rows())==1
    q.add({'task_id':'synthetic','seed':1})
    assert len(q.rows())==1
    with pytest.raises(ValueError):q.add({'task_id':'synthetic','seed':2})

def test_role_seed_mask_config_identity():
    base={'config':{'ops':['GCN'],'lr':.01},'seed':1,'role':'rerank','mask':'public','code':'x'}
    for key,value in [('seed',2),('role','evaluation'),('mask','new'),('code','y'),('config',{'ops':['GAT'],'lr':.01})]:
        assert digest(dict(base,**{key:value}))!=digest(base)

def test_access_isolation_subprocess(tmp_path):
    repo=tmp_path/'repo';root=repo/'results'/'new';root.mkdir(parents=True)
    (repo/'Cora').mkdir();(repo/'Cora'/'hidden').write_text('synthetic sentinel')
    (root/'private').mkdir();(root/'private'/'unreleased').write_text('synthetic sentinel')
    (root/'PUBLIC_C.json').write_text('{}')
    for role in ['search','train']:
        code=f'''from pathlib import Path
from analyse.final_configuration_confirmation_v1.common import access_guard
repo=Path({str(repo)!r});root=Path({str(root)!r})
access_guard(root,repo,{role!r})
for p in [repo/'Cora'/'hidden',root/'private'/'unreleased']:
 try:p.read_text()
 except PermissionError:continue
 raise AssertionError('hidden source accessible')
assert (root/'PUBLIC_C.json').read_text()=='{{}}'
'''
        subprocess.run([sys.executable,'-B','-c',code],check=True)

def test_budget_and_thresholds():
    assert 20*(71*3+10+61*3+30+10+61*3+30+10)==13380
    comparisons={k:{'selection_CI':[0,1]} for k in ['S-U','M-U','M-S']}
    assert selected_rule(comparisons)=='U'
    comparisons['S-U']['selection_CI']=[.1,1]
    assert selected_rule(comparisons)=='S'
    comparisons['M-U']['selection_CI']=[.1,1]
    assert selected_rule(comparisons)=='S'
    comparisons['M-S']['selection_CI']=[.01,1]
    assert selected_rule(comparisons)=='M'
    assert selected_rule(comparisons,False)=='S'

def test_bootstrap_preserves_identical_returns():
    counts=np.arange(200).reshape(20,10)%20+390
    same=np.stack([counts,counts,counts],axis=-1)
    result=bootstrap(same)
    assert result['selected_rule']=='U'
    for c in result['comparisons'].values():
        assert c['selection_CI']==[0,0] and c['ties']==20
    assert result['comparisons']['S-U']['selection_confidence']==1-.05/3
    with pytest.raises(ValueError):bootstrap(np.zeros((19,10,3)))

def test_seal_gate(tmp_path):
    with pytest.raises(PermissionError):score_gate(tmp_path)

def test_persistent_timer(tmp_path,monkeypatch):
    from analyse.final_configuration_confirmation_v1 import monitor
    q=Queue(tmp_path/'q.sqlite');ids=[]
    for i in range(6):
        t={'task_id':str(i),'role':'rerank_seed' if i%2 else 'rerank_mask','parameter_count':10+i,'seed':i,'split':'original'}
        q.add(t);a=q.claim(i%2+1,{'uuid':str(i%2),'index':i%2})
        e={'task_id':a['task']['task_id'],'token':a['token'],'worker':i%2+1,'completed':True,
            'started':i//2*10,'ended':i//2*10+10,'wall_seconds':10,'strict_resumes':0}
        q.finish(e);ids.append(a['task']['task_id'])
        write(tmp_path/'training'/a['task']['task_id']/'result.json',{'worker_seconds':9,'gpu_event_seconds':8,'stopped_epoch':100})
    write(tmp_path/'PILOT_MANIFEST.json',{'task_ids':ids})
    monkeypatch.setattr(monitor.time,'time',lambda:1000)
    eta=estimate(tmp_path,q)
    first=eta['first_check_unix'];assert first==eta['conservative_completion_unix']+900
    assert not due(eta,first-1) and due(eta,first)
    updated=recheck(eta,first)
    assert updated['next_check_unix']==first+300
    write(tmp_path/'ETA.json',updated)
    monkeypatch.setattr(monitor.time,'time',lambda:2000)
    again=estimate(tmp_path,q)
    assert again['first_check_unix']==first and again['next_check_unix']==first+300

def test_real_CPU_checkpoint_resume_and_complete_reentry(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    from deterministic_runtime import configure_torch_determinism
    configure_torch_determinism(torch);torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
    import analyse.rank_reversal_diagnostic.train as trainer
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(3,2);self.drop=torch.nn.Dropout(.2)
        def forward(self,x,edges):
            return self.linear(self.drop(x))
    monkeypatch.setattr(trainer,'make_model',lambda *args:Tiny())
    x=torch.arange(36,dtype=torch.float32).reshape(12,3)/36
    y=torch.tensor([0,1]*4+[-100]*4)
    data=SimpleNamespace(x=x,edge_index=torch.empty((2,0),dtype=torch.long),y=y,
        train_mask=torch.arange(12)<4,val_mask=(torch.arange(12)>=4)&(torch.arange(12)<8))
    config={'lr':.01,'l2':1e-5};calls=[]
    baseline=trainer.train(config,data,3,2,71,torch.device('cpu'),tmp_path/'baseline','id',max_epochs=10,patience=80)
    with pytest.raises(trainer.PlannedPause):
        trainer.train(config,data,3,2,71,torch.device('cpu'),tmp_path/'resume','id',max_epochs=10,patience=80,
            pause_after=5,on_initialization=lambda:calls.append(1))
    resumed=trainer.train(config,data,3,2,71,torch.device('cpu'),tmp_path/'resume','id',max_epochs=10,patience=80,
        on_initialization=lambda:calls.append(1))
    assert baseline['trajectory']==resumed['trajectory'] and calls==[1]
    states=[torch.load(tmp_path/n/'best_model.pt',weights_only=False)['model'] for n in ['baseline','resume']]
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    before={p.name:sha(p) for p in (tmp_path/'resume').iterdir() if p.is_file()}
    trainer.train(config,data,3,2,71,torch.device('cpu'),tmp_path/'resume','id',max_epochs=10,patience=80,
        on_initialization=lambda:calls.append(1))
    assert calls==[1] and before=={p.name:sha(p) for p in (tmp_path/'resume').iterdir() if p.is_file()}

def test_stopped_grant_recovers_only_complete_identity_state(tmp_path,monkeypatch):
    import torch
    from analyse.final_configuration_confirmation_v1 import controller
    q=Queue(tmp_path/'q.sqlite')
    task={'task_id':'synthetic','seed':11};q.add(task)
    gpu={'uuid':'synthetic-gpu','index':0};a=q.claim(1,gpu)
    write(tmp_path/'GPUS.json',{'workers':{'1':gpu}})
    write(tmp_path/'worker_events'/(a['token']+'.json'),{'stopped':True,'task_id':'synthetic','token':a['token'],'worker':1})
    folder=tmp_path/'training'/'synthetic';folder.mkdir(parents=True)
    saved={'identity':a['execution_identity'],**{k:{} for k in ['model','optimizer','scheduler','rng','best_state']},
        'epoch':5,'best_val':.5,'no_improve':0}
    torch.save(saved,folder/'latest.pt')
    controller.recover(tmp_path,q)
    assert q.rows()[0]['state']=='READY' and q.rows()[0]['reserved']==1
    resumed=q.claim(1,gpu)
    assert resumed['execution_identity']==a['execution_identity'] and q.rows()[0]['reserved']==1
    (folder/'latest.pt').unlink()
    with pytest.raises(RuntimeError,match='lacks complete latest state'):
        controller.recover(tmp_path,q)

def test_synthetic_worker_completion_has_no_second_grant(tmp_path):
    q=Queue(tmp_path/'q.sqlite')
    for i in range(6):q.add({'task_id':str(i),'seed':i})
    received=[]
    while q.rows('READY'):
        assignments=[q.claim(w,{'uuid':str(w),'index':w-1}) for w in (1,2)]
        for a in assignments:
            if a:
                received.append(a['task']['task_id'])
                q.finish({'task_id':a['task']['task_id'],'token':a['token'],'worker':a['worker'],
                    'completed':True,'exit_code':0})
    assert len(received)==len(set(received))==6 and len(q.rows('COMPLETE'))==6
    assert all(q.claim(w,{'uuid':str(w),'index':w-1}) is None for w in (1,2))

def test_synthetic_full_development_acceptance_and_report(tmp_path,monkeypatch):
    """All 20 groups and 60 budgets, fake scores/files only, zero GNN training."""
    from analyse.final_configuration_confirmation_v1 import controller,analysis,common
    import time
    root=tmp_path;configs={f'{i:04d}':{'configuration_hash':str(i),'normalized_config_id':f'{i:04d}',
        'parameter_count':i+1,'effective_configuration':{'synthetic':i}} for i in range(80)}
    write(root/'CONFIGURATIONS.json',{'by_id':configs})
    write(root/'SPLITS.json',{'splits':{k:{'identity':k} for k in ['original','split1','split2']},'formal_M_allowed':True})
    write(root/'FREEZE.json',{'identity':'synthetic','inputs':{'development_data.pt':'x','TRAINING_RECIPE.json':'y','CODE_MANIFEST.json':'z'}})
    write(root/'CODE_MANIFEST.json',{'files':{}});write(root/'PROTOCOL.json',{'synthetic':True})
    groups={str(g):{'rerank':list(range(16*g,16*g+6)),'M_each_split':list(range(16*g,16*g+3)),
        'evaluation':list(range(16*g+6,16*g+16))} for g in range(20)}
    write(root/'SEEDS.json',{'groups':groups})
    write(root/'RUN_STARTED.json',{'unix':time.time()});write(root/'SEARCH_COMPLETE.json',{'synthetic':True})
    q=Queue(root/'q.sqlite')
    observations=[{'candidate_id':f'{i:04d}','query_correct_sum':900+i,'query_denominator':1500} for i in range(71)]
    for g in range(20):
        folder=root/'trajectories'/f'g{g:02d}'
        write(folder/'U_observations.json',observations);write(folder/'SM_observations.json',observations[:61])
        top=[f'{i:04d}' for i in range(60,55,-1)]
        write(root/'search_ready'/f'g{g:02d}_SM.json',{'observations':observations[:61],'top5':top})
        write(root/'search_ready'/f'g{g:02d}_U.json',observations[-1])
        for name,kind in [('SM_terminal_n60','terminal_mean'),('U_terminal_n70','terminal_mean'),('U_q11','qlogei')]:
            write(folder/'decisions'/(name+'.json'),{'kind':kind})
        for n in range(50,71):write(folder/'fits'/f'n{n:02d}'/'fit.json',{'starts':[{}, {}, {}],'fit_seconds':0})
    controller.discover(root,q)
    assert len(q.rows())==1200 and len(list((root/'returns').glob('*.json')))==20
    controller.progress(root,q,'SYNTHETIC_RERANK')
    # No evaluation is released before every return in that group is sealed.
    assert all(json.loads(r['payload'])['role']!='final_validation' for r in q.rows())
    for row in q.rows():
        folder=root/'training'/row['id'];write(folder/'BINDING.json',{'correct':400,'denominator':500})
        write(folder/'INITIALIZATION.json',{'time':'2999'})
    q.db.execute("UPDATE tasks SET state='COMPLETE'");q.db.commit()
    controller.discover(root,q)
    assert len(list((root/'returns').glob('*.json')))==60 and len(q.rows())==1600
    def fake_binding(root,row):
        folder=root/'training'/row['id'];task=json.loads(row['payload'])
        correct=401 if task['role']=='final_validation' and task['candidate_id']=='0056' else 400
        result={'worker_seconds':0,'gpu_event_seconds':0,'stopped_epoch':1,'peak_gpu_bytes':0}
        binding={'correct':correct,'denominator':500,'return_seals_before_evaluation':{
            r:sha(root/'returns'/f'g{task["group"]:02d}_{r}.json') for r in controller.associated_rules(root,task)}}
        for name in ['best_model.pt','selection.npz']:(folder/name).parent.mkdir(parents=True,exist_ok=True);(folder/name).write_bytes(b'synthetic')
        write(folder/'result.json',result);write(folder/'BINDING.json',binding)
        write(folder/'INITIALIZATION.json',{'time':'2999'})
        (folder/'attempts.jsonl').write_text('{}\n')
        return result,binding
    monkeypatch.setattr(controller,'complete_binding',fake_binding)
    monkeypatch.setattr(analysis,'verify_run',lambda *args:'synthetic')
    for row in q.rows():fake_binding(root,row)
    q.db.execute("UPDATE tasks SET state='COMPLETE',worker=1,gpu='synthetic',event=?",(json.dumps({'strict_resumes':0,'wall_seconds':0}),));q.db.commit()
    controller.refresh_manifest(root,q);controller.progress(root,q,'SYNTHETIC_FINAL')
    for name in ['TIMING_PILOT.csv','ETA.json','CACHE_ACCEPTANCE.json','H_QUALIFICATION.json','PILOT_MANIFEST.json']:
        write(root/name,{'synthetic':True})
    (root/'ACCEPTANCE_CHECKS.jsonl').write_text('{}\n');(root/'GRANTS.jsonl').write_text('{}\n');(root/'frozen_code').mkdir()
    (root/'worker_events').mkdir()
    write(root/'worker_events'/'synthetic.json',{'completed':True})
    analysis.finalize(root,tmp_path,q)
    assert read(root/'STATUS.json')['state']=='COMPLETE'
    assert read(root/'ACCEPTANCE.json')['logical_training_cost']==13380
    assert read(root/'SELECTED_RETURN_RULE.json')['selected_rule']=='S'
    assert read(root/'STATISTICS.json')['comparisons']['M-S']['selection_CI']==[0,0]
    assert (root/'EVIDENCE.tar.gz').exists() and (root/'DEVELOPMENT_RESULTS.md').exists()
    assert analysis.verify_complete(root,tmp_path)['new_training_or_scoring_on_resume']==0
