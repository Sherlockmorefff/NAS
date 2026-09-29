"""Unique serial B runner; no imports of scored A/B observations or metrics."""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from stage01_core import cpu_setup, tensor_hash
from residual_core import Budget, BudgetExceeded, ResidualBranch, RawBranch, BlockSession, snapshot, restore, check_forks
from residual_diagnostics import diagnose_isolated, stability_at
from prepare_residual import read, write, sha, digest, now

METHODS=['R_FROZEN','R_REG','R_DKL_C','R_DKL_U']
def verify_freeze(root):
    for filename in ['frozen_manifest.json','code_manifest.json']:
        for relative,value in read(root/filename)['files'].items():
            if sha(root/relative)!=value:raise ValueError('Frozen identity mismatch: '+relative)
    for relative,value in read(root/'source_manifest.json')['files'].items():
        if sha(root/relative)!=value:raise ValueError('Source changed: '+relative)

def save_torch(path,state):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)

class Runner:
    def __init__(self,root,resume=False):
        self.root=Path(root);self.resume=resume;verify_freeze(self.root)
        # Only the gate decision is admitted here. The truth-bearing A tables are never imported.
        gate=read(self.root/'admission.json')
        if gate['route']!='RUN_RESIDUAL_ABLATION':raise RuntimeError('B not admitted by the frozen gate')
        if gate['diagnostic_gate_sha256']!=sha(self.root/'diagnostic_gate.json') or not gate['diagnostic_complete']:
            raise ValueError('Admission identity differs from sealed diagnostic gate')
        if gate['protocol_sha256']!=sha(self.root/'protocol.json') or gate['code_manifest_sha256']!=sha(self.root/'code_manifest.json'):
            raise ValueError('Protocol/code differs from A admission')
        if (self.root/'predictions_manifest.json').exists():raise RuntimeError('B already sealed; no further training')
        self.budget=Budget(self.root/'resource_ledger.json',limit=1200)
        self.contexts=read(self.root/'worklist.json')['contexts'];self.active=None;self.active_key=None
        self.started=time.perf_counter();self.forecasts={};self.completed_contexts=0
        for folder in ['checkpoints','states','predictions','diagnostics','block_summaries','recovery']:(self.root/folder).mkdir(exist_ok=True)

    def compute(self,fn,kind='prediction'):
        self.budget.check();start=time.perf_counter()
        try: result=fn()
        finally:self.budget.charge(time.perf_counter()-start,kind=kind)
        self.budget.check();return result

    def persist_checkpoint(self,session,path,check=True):
        start=time.perf_counter()
        try: state=session.snapshot()
        finally:self.budget.charge(time.perf_counter()-start,kind='prediction')
        state['runner_budget_at_save']=self.budget.as_dict()
        state['runner_block_model_seconds']=self.previous_model_seconds+self.budget.seconds-self.block_start_seconds
        save_torch(path,state);self.budget.persist()
        if check:self.budget.check()
        return state

    def diagnostic(self,model,query,training,step,mode,key,last=None,lrs=None):
        q=torch.tensor(query['x_raw'],dtype=torch.double)
        scaler=training['scaler'];q=(q-torch.tensor(scaler['x_mean']))/torch.tensor(scaler['x_scale'])
        record=self.compute(lambda:diagnose_isolated(model,q,query['ids'],scaler['y_mean'],scaler['y_scale'],
            step,mode=mode,with_geometry=step in (0,1000) or '_COMMON_' in key,training_ids=training['ids']))
        record.update(block_key=key,block_step=step,last_update=last,lrs=lrs,cumulative_resources=self.budget.as_dict())
        matrices={k:np.asarray(record.pop(k),dtype=np.float64) for k in
            ['training_kernel','combined_kernel','training_correlation','combined_correlation'] if k in record}
        if matrices:
            matrix_path=self.root/'diagnostics'/key/f'kernels_j{step}.npz';matrix_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(matrix_path,**matrices)
            record['kernel_arrays_path']=str(matrix_path.relative_to(self.root));record['kernel_arrays_sha256']=sha(matrix_path)
        write(self.root/'diagnostics'/key/f'j{step}.json',record)
        return record

    def diagnostic_record(self,key,step):
        record=read(self.root/'diagnostics'/key/f'j{step}.json')
        if 'kernel_arrays_path' in record:
            path=self.root/record['kernel_arrays_path']
            if sha(path)!=record['kernel_arrays_sha256']:raise ValueError('Diagnostic arrays changed')
            with np.load(path,allow_pickle=False) as arrays:
                record.update({k:arrays[k] for k in arrays.files})
        return record

    def block(self,model,mode,steps,key,query,training,final_block=False):
        checkpoint=self.root/'checkpoints'/f'{key}.pt';summarypath=self.root/'block_summaries'/f'{key}.json'
        if summarypath.exists():
            if not self.resume:raise FileExistsError(summarypath)
            summary=read(summarypath)
            if summary['completed']!=steps:raise ValueError('Completed summary wrong steps')
            return self.compute(lambda:restore(torch.load(checkpoint,map_location='cpu',weights_only=False))),summary
        recovery=self.root/'recovery'/f'{key}.pt'
        block_start_seconds=self.budget.seconds
        if recovery.exists():
            if not self.resume:raise FileExistsError(recovery)
            state=torch.load(recovery,map_location='cpu',weights_only=False)
            if state['runner_budget_at_save']['updates']!=self.budget.updates:
                raise RuntimeError('Recovery checkpoint and budget update ledger differ; refuse replay')
            session=self.compute(lambda:BlockSession.from_snapshot(state))
        else:session=self.compute(lambda:BlockSession(model,mode,steps,key))
        previous_model_seconds=float(state.get('runner_block_model_seconds',0)) if recovery.exists() else 0.0
        self.block_start_seconds=block_start_seconds;self.previous_model_seconds=previous_model_seconds
        self.active=session;self.active_key=key
        if session.completed==0:self.diagnostic(session.model,query,training,0,mode,key,lrs=session.lrs)
        tracepath=self.root/'diagnostics'/key/'updates.jsonl';tracepath.parent.mkdir(parents=True,exist_ok=True)
        with tracepath.open('a') as stream:
            while session.completed<steps:
                row=session.step(self.budget)
                if session.completed%100==0 or session.completed==steps:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    self.diagnostic(session.model,query,training,session.completed,mode,key,last=session.last_update,lrs=session.lrs)
                    stream.flush();self.persist_checkpoint(session,recovery)
        state=self.persist_checkpoint(session,checkpoint)
        result={'block_key':key,'mode':mode,'planned_steps':steps,'completed':session.completed,
            'training_seconds':session.training_seconds,'model_seconds':previous_model_seconds+self.budget.seconds-block_start_seconds,'math_state_hash':session.model.fingerprint(),
            'checkpoint':str(checkpoint.relative_to(self.root)),'checkpoint_sha256':sha(checkpoint)}
        if final_block:
            records=[self.diagnostic_record(key,i) for i in range(0,1001,100)]
            result['stability']=stability_at(records,1000)
        write(summarypath,result);self.active=None;self.active_key=None
        return session.model,result

    def endpoint(self,model,ctx,method,training,query,key,common_seconds,branch_seconds,common_model_seconds=None,branch_model_seconds=None):
        uid=ctx['subset_id']+'_RAW16' if method=='RAW16' else ctx['context_id']+'_'+method
        endpoint_path=self.root/'predictions'/f'{uid}.json'
        if endpoint_path.exists():
            if not self.resume:raise FileExistsError(endpoint_path)
            return
        record=read(self.root/'diagnostics'/key/'j1000.json')
        checkpoint=self.root/'checkpoints'/f'{key}.pt';scaler=training['scaler']
        mean=record['query_mean_raw'];variance=record['query_latent_variance_standardized']
        extra=record['gp_parameters']['extra_noise_variance_standardized']
        geometry=record['geometry_row']
        payload={**{k:ctx[k] for k in ['context_id','subset_id','fold','sample_seed','n','model_seed']},
            'unit_id':uid,'method':method,'model_seed':None if method=='RAW16' else ctx['model_seed'],
            'candidate_ids':query['ids'],'architecture_fingerprints':query['architectures'],
            'mean_raw':mean,'mean_standardized':[(m-scaler['y_mean'])/scaler['y_scale'] for m in mean],
            'latent_variance_standardized':variance,'latent_variance_raw':[v*scaler['y_scale']**2 for v in variance],
            'y_center':scaler['y_mean'],'y_scale':scaler['y_scale'],'extra_noise_variance_standardized':extra,
            'endpoint_state_hash':model.fingerprint(),'state_path':str(checkpoint.relative_to(self.root)),
            'state_file_sha256':sha(checkpoint),'training_ids_sha256':digest(training['ids']),
            'training_tensor_sha256':tensor_hash({'x':model.x,'y':model.y,'known':model.known}),
            'scaler_sha256':digest(scaler),'completed_steps':1300 if method=='RAW16' else (3300 if method=='R_REG' else 2300),
            'input_dimension':16,'query_labels_or_SE_present':False,
            'negative_roundoff_variance_count':sum(v<0 for v in variance),
            'geometry_row':geometry,'raw_reference_source':'new protocol-matched fit' if method=='RAW16' else ctx['subset_id']+'_RAW16',
            'cost':{'common_training_seconds':common_seconds,'branch_training_seconds':branch_seconds,
               'prediction_seconds':0.0,'prediction_accounting':'all endpoint predictions reused from charged final diagnostic',
               'common_model_seconds':common_model_seconds,'branch_model_seconds':branch_model_seconds,
               'standalone_fit_predict_seconds':common_model_seconds+branch_model_seconds},
            'units':{'mean_raw':'accuracy fraction','latent_variance_standardized':'unclamped standardized variance',
               'extra_noise_variance_standardized':'variance, not std'},'finished_at_utc':now()}
        if min(variance)<-1e-8:raise FloatingPointError('Latent variance below -1e-8')
        if method=='RAW16':
            old=read(self.root/'source/stage2/predictions'/f'{uid}.json')
            differences={k:max(abs(a-b) for a,b in zip(payload[k],old[k])) for k in ['mean_standardized','latent_variance_standardized']}
            same=all(np.allclose(payload[k],old[k],rtol=1e-6,atol=1e-8) for k in differences)
            write(self.root/'diagnostics'/f'{uid}_historical_consistency.json',{'pass':same,'maximum_absolute_difference':differences,
               'source_sha256':sha(self.root/'source/stage2/predictions'/f'{uid}.json'),'new_fit_not_substitution':True})
            if not same:raise ValueError('New RAW16 differs from identical historical protocol')
        write(endpoint_path,payload)

    def context(self,ctx):
        self.active=None;self.active_key=None
        cid,sid=ctx['context_id'],ctx['subset_id'];training=read(self.root/f'training/{sid}.json');query=read(self.root/f'queries/fold{ctx["fold"]}.json')
        if set(training['ids'])&set(query['ids']):raise ValueError('Training/query overlap')
        tensors=tuple(torch.tensor(training[k],dtype=torch.double) for k in ['x_standardized','y_standardized','known_variance_standardized'])
        rawcost=0;before=self.budget.seconds
        if ctx['run_raw']:
            cpu_setup(0);raw=self.compute(lambda:RawBranch(*tensors))
            raw_initial_seconds=self.budget.seconds-before
            raw,a=self.block(raw,'gp',300,cid+'_RAW16_initial',query,training)
            raw,b=self.block(raw,'gp',1000,cid+'_RAW16_final',query,training,True)
            self.endpoint(raw,ctx,'RAW16',training,query,cid+'_RAW16_final',0,a['training_seconds']+b['training_seconds'],0,raw_initial_seconds+a['model_seconds']+b['model_seconds'])
            rawcost=self.budget.seconds-before
        mappedbefore=self.budget.seconds
        cpu_setup(ctx['model_seed']);common=self.compute(lambda:ResidualBranch(*tensors,ctx['model_seed']))
        initial=self.compute(lambda:snapshot(common));save_torch(self.root/'states'/f'{cid}_INITIAL.pt',initial)
        initial_model_seconds=self.budget.seconds-mappedbefore
        common,a=self.block(common,'regression',1000,cid+'_COMMON_regression',query,training)
        common,b=self.block(common,'gp',300,cid+'_COMMON_gp',query,training)
        common_seconds=a['training_seconds']+b['training_seconds'];common_model_seconds=initial_model_seconds+a['model_seconds']+b['model_seconds']
        forks=self.compute(lambda:check_forks(common),kind='verification')
        if not forks['checks']['pass']:raise AssertionError('Fork identity/isolation failed')
        write(self.root/'diagnostics'/f'{cid}_fork.json',forks['checks'])
        for method in METHODS:
            branch_start=self.budget.seconds
            model=forks['branches'][method];head_before=tensor_hash(model.mapper.head.state_dict())
            mapped_before=tensor_hash(model.mapper.state_dict())
            branch_seconds=0;branch_model_seconds=0
            if method=='R_REG':
                model,r=self.block(model,'regression',1000,cid+'_'+method+'_regression',query,training)
                branch_seconds+=r['training_seconds'];branch_model_seconds+=r['model_seconds'];mapped_before=tensor_hash(model.mapper.state_dict())
            mode='joint' if method.startswith('R_DKL') else 'gp';key=cid+'_'+method+'_'+mode
            model,r=self.block(model,mode,1000,key,query,training,True);branch_seconds+=r['training_seconds'];branch_model_seconds+=r['model_seconds']
            if mode=='gp' and tensor_hash(model.mapper.state_dict())!=mapped_before:raise AssertionError('Frozen mapper changed')
            if mode=='joint' and tensor_hash(model.mapper.head.state_dict())!=head_before:raise AssertionError('DKL head changed')
            self.endpoint(model,ctx,method,training,query,key,common_seconds,branch_seconds,common_model_seconds,branch_model_seconds)
        self.forecasts.setdefault(ctx['n'],{'mapped_seconds':self.budget.seconds-mappedbefore,'raw_seconds':rawcost})

    def progress(self,index):
        forecast=None
        if set(self.forecasts)=={20,50}:
            expected=sum(self.forecasts[c['n']]['mapped_seconds']+(self.forecasts[c['n']]['raw_seconds'] if c['run_raw'] else 0) for c in self.contexts[index+1:])
            forecast={'best':.8*expected,'expected':expected,'worst':1.25*expected,
                'assumptions':'first retained n20/n50 runtime; .8/1/1.25 multipliers, same CPU workload; not a guarantee'}
        value={'completed_contexts':index+1,'completed_endpoints':len(list((self.root/'predictions').glob('*.json'))),
            'updates':self.budget.updates,'budget_used_seconds':self.budget.seconds,'limit':1200,'forecast_remaining_seconds':forecast,
            'formal_wall_seconds':time.perf_counter()-self.started,'at_utc':now()}
        write(self.root/'progress.json',value);print(json.dumps(value),flush=True)

    def run(self):
        failure=None;status='COMPLETE_VALID';ctx=None
        try:
            for index,ctx in enumerate(self.contexts):
                self.budget.check();self.context(ctx);self.completed_contexts=index+1;self.progress(index)
            self.budget.check()
            if self.budget.updates!=404000:raise AssertionError('Unexpected formal update count')
        except Exception as exc:
            status='PARTIAL_BUDGET' if isinstance(exc,BudgetExceeded) else 'INVALID'
            failure={'type':type(exc).__name__,'message':str(exc),'context':ctx,'traceback':traceback.format_exc(),'at_utc':now()}
            write(self.root/'failure.json',failure)
            if self.active is not None:self.persist_checkpoint(self.active,self.root/'recovery'/f'{self.active_key}_STOPPED.pt',False)
        self.budget.persist()
        files=[]
        for path in sorted((self.root/'predictions').glob('*.json')):
            p=read(path);files.append({**{k:p[k] for k in ['unit_id','context_id','subset_id','method','fold','sample_seed','model_seed','n']},
                'path':str(path.relative_to(self.root)),'sha256':sha(path),'state_path':p['state_path'],
                'state_sha256':p['state_file_sha256'],'endpoint_state_hash':p['endpoint_state_hash']})
        expected=[e['unit_id'] for e in read(self.root/'worklist.json')['endpoints']]
        missing=sorted(set(expected)-{p['unit_id'] for p in files})
        if status=='COMPLETE_VALID' and missing:status='INVALID'
        seal={'sealed':True,'sealed_at_utc':now(),'execution_status':status,'expected_endpoint_count':260,
            'expected_unit_ids':expected,'files':files,'missing_unit_ids':missing,
            'protocol_sha256':sha(self.root/'protocol.json'),'source_manifest_sha256':sha(self.root/'source_manifest.json'),
            'code_manifest_sha256':sha(self.root/'code_manifest.json'),'worklist_sha256':sha(self.root/'worklist.json'),
            'failure':failure,'resources':self.budget.as_dict(),'next_stage_authorized':False}
        write(self.root/'predictions_manifest.json',seal)
        result={'execution_status':status,'diagnostic_route':'RUN_RESIDUAL_ABLATION','diagnostic_complete':True,
            'formal_contexts_planned_if_admitted':60,'formal_endpoints_planned_if_admitted':260,'formal_updates_planned_if_admitted':404000,
            'formal_contexts_actual':self.completed_contexts,'formal_endpoints_actual':len(files),'formal_updates_actual':self.budget.updates,
            'fit_predict_verify_budget_seconds':1200,'resource_seconds':self.budget.seconds,
            'formal_wall_seconds':time.perf_counter()-self.started,'scientific_signal_n20':'not_scored','scientific_signal_n50':'not_scored',
            'constraint_signal_n20':'not_scored','constraint_signal_n50':'not_scored','next_stage_authorized':False}
        write(self.root/'status.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    Runner(a.root,a.resume).run()
