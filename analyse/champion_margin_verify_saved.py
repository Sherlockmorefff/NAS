"""Forward-only verification of both completed replay stages and preserved baselines."""
import argparse,collections,hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import torch
from analyse.relative_metric_pilot import read,write,load,tensor_hash,file_hash,digest
from surrogate.gp_guidance_replay_engine import scaler_from_initial,input_tensors,reload_model,score_candidates,journal
from surrogate.champion_margin import list_indices,objective,select_batch
from analyse.champion_margin_run import verify_protocol,budget


def main(o):
    def forbidden(*a,**k):raise RuntimeError('verification cannot train')
    torch.optim.Adam.step=forbidden;torch.Tensor.backward=forbidden;torch.autograd.grad=forbidden
    p=verify_protocol(o);cfg=p['config'];tasks=read(o/'tasks.json');features=read(o/'features.json');decision=read(o/'stage1_decision.json')
    counts=collections.Counter();details=[];checkpoints=[]
    for task in tasks:
        d=o/'trajectories'/task['name'];done=read(d/'completed.json');assert done['state_sha']==file_hash(d/'state.pt')
        st=load(d/'state.pt');assert st['protocol']==p['identity'];initial=read(o/task['label_file']);known=dict(initial);chosen=[];sources=[]
        loss='gap' if task['stage']==1 else decision['selected_loss'];assert st['loss']==loss
        scaler=scaler_from_initial(features,task['initial_ids'],initial);assert st['scaler']==scaler
        before_weights=load(o/task['weight_file'])['model'];assert st['initial_weight_hash']==tensor_hash(before_weights)
        models={};previous=before_weights
        for q in range(0,cfg['query_budget'],cfg['update_interval']):
            ev='initial' if q==0 else f'new{q}';folder=d/'events'/ev;zero=load(folder/'post_step000.pt');end=load(folder/'latest.pt')
            steps=cfg['initial_steps'] if q==0 else cfg['update_steps'];expected_ids=sorted(set(task['initial_ids'])|set(st['query_order'][:q]))
            assert zero['step']==0 and not zero['optimizer']['state'] and tensor_hash(zero['model'])==tensor_hash(previous)
            assert end['known_ids']==expected_ids and end['scaler']==scaler and end['step']==steps and end['config']==cfg
            assert end['mode']==cfg['margin_modes'][loss] and end['weight']==(cfg['ranking_weight'] if loss=='gap' else 0.)
            rho=(cfg['query_budget']-q)/(179-cfg['initial_count']-q);assert end['rho']==rho
            assert all(float(v['step'])==steps for v in end['optimizer']['state'].values())
            labels={c:st['known'][c] for c in expected_ids};data=input_tensors(features,scaler,labels)
            logs=sorted((folder/'step_logs').glob('*.json'));assert len(logs)==steps
            for fp in logs:
                log=read(fp);lists=list_indices(expected_ids,'replay',task['task_id'],ev,log['completed_steps'],cfg)
                assert log['list_hash']==digest(lists) and log['known_ids_hash']==digest(expected_ids) and log['scaler_hash']==digest(scaler)
                assert log['k']==[math.ceil(16*rho)]*16 and all(math.isfinite(log[k]) for k in ['regression','size','ranking','total'])
                assert log['margin_min']>=0 and log['margin_max']<=cfg['margin_cap'];counts['step_logs']+=1
            for prepath in folder.glob('pre_step*.pt'):
                pre=load(prepath);assert pre['timing']=='pre_update' and pre['model_hash']==tensor_hash(pre['model'])
                model=reload_model(pre['model']);model.train()
                with torch.no_grad():
                    e,s=model(data['z'],data['hp']);assert torch.equal(e,pre['embedding']) and torch.equal(s,pre['scores'])
                    losses,diag=objective(e,s,data['target32'],data['y'],expected_ids,pre['lists'],rho,scaler['y_scale'],cfg,end['mode'],end['weight'])
                log=pre['diagnostics'];assert log['completed_steps']==pre['step'] and log['model_hash']==pre['model_hash']
                assert all(float(v)==log[k] for k,v in losses.items());assert diag['margins']==log['margins']
                post=load(folder/f"post_step{pre['step']+1:03}.pt");assert post['step']==pre['step']+1 and post['timing']=='post_update'
                assert post['model_hash']==tensor_hash(post['model'])==post['last_log']['post_model_hash']
                counts['pre_post_snapshot_pairs']+=1
            models[q//cfg['update_interval']]=end['model'];previous=end['model'];counts['training_events']+=1
        prior=None
        for b in range(10):
            folder=d/'batches'/f'b{b:02}';sel=read(folder/'selection.json');receipt=read(folder/'reveal.json')
            assert sel['known_ids']==sorted(known) and sel['known_labels_hash']==digest(known)
            assert receipt['selection_sha']==file_hash(folder/'selection.json') and sel['selected_at']<receipt['revealed_at']
            model=reload_model(models[b//2]);data=input_tensors(features,scaler,known);rows,_,_=score_candidates(model,data)
            assert rows==sel['rows'] and sel['model_hash']==tensor_hash(models[b//2]) and sel['version']==b//2
            pick=select_batch(rows,known,task['search_seed'],b,cfg,task['strategy'])
            assert all(pick[k]==sel[k] for k in pick)
            assert set(receipt['labels'])==set(sel['selected_ids']) and not set(sel['selected_ids'])&set(known)
            assert read(folder/'aggregated.json')['query_count']==(b+1)*5
            if task['strategy']=='greedy' and b%2:
                assert prior['selected_ids']+sel['selected_ids']==[r['candidate_id'] for r in prior['rows'][:10]];counts['two5_equal_top10']+=1
            prior=sel;chosen.extend(sel['selected_ids']);sources.extend(sel['sources']);known.update(receipt['labels']);counts['selection_batches']+=1
        assert chosen==st['query_order'] and known==st['known'] and len(set(chosen))==50
        assert sources.count('greedy')==(50 if task['stage']==1 else 40) and sources.count('exploration')==(0 if task['stage']==1 else 10)
        details.append(dict(task=task['name'],queries=50,known=70,greedy=sources.count('greedy'),exploration=sources.count('exploration'),initial_weights_equal=True,scaler_fixed=True))
        for path in d.rglob('*.pt'):checkpoints.append(dict(path=str(path),sha256=file_hash(path),bytes=path.stat().st_size))
    resumes=[]
    for stage in [1,2]:
        folder=o/'trajectories'/f'stage{stage}-start{cfg["search_seeds"][0]}-s0/batches/b00'
        pause=read(folder/'pause-selected.json');sel=file_hash(folder/'selection.json')
        assert pause['selection_sha']==sel and read(folder/'reveal.json')['selection_sha']==sel
        resumes.append(dict(stage=stage,boundary='selected_not_revealed',selection_sha=sel,preserved=True))
        if stage==2:
            pause=read(folder/'pause-revealed.json');assert pause['selection_sha']==sel
            resumes.append(dict(stage=stage,boundary='revealed_not_aggregated',selection_sha=sel,logical_queries_counted_once=True))
    cost=budget(o);assert cost['formal_logical_updates']==cost['formal_physical_confirmed_steps']==11000 and cost['new_history_queries']==500
    assert cost['repeated_physical_steps']==cost['unconfirmed_physical_intents']==0
    assert cost['synthetic_steps']==6
    protected=read(o/'protected_before.json');changed=[n for n,h in protected.items() if file_hash(ROOT/n)!=h];assert not changed
    before=read(o/'workspace_before_code.json');assert hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest()==before['diff_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==before['HEAD']
    assert file_hash(Path(p['VAE']['checkpoint']))==p['VAE']['checkpoint_sha256']
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True);assert diff.returncode==0
    checks=read(o/'checks.json');checks.update(formal_verification_passed=True,formal_verification_counts=dict(counts),formal_trajectories=details,formal_boundary_resume=resumes,
        protected_files_rechecked=len(protected),protected_changed=changed,tracked_user_diff_unchanged=True,HEAD_unchanged=True,VAE_checkpoint_unchanged=True,git_diff_check_final=True,
        verification_optimizer_steps=0,missing_evidence=['A-only conditional development results; no independent B/test confirmation.','Per-component gradient contribution throughout training was not measured.'])
    write(o/'checks.json',checks,immutable=False);write(o/'checkpoint_manifest.json',checkpoints)
    print(json.dumps(dict(counts=dict(counts),budget=cost,protected_files_unchanged=len(protected)),ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);main(a.output.resolve())
