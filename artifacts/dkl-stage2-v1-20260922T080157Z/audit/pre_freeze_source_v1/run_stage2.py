"""The single, serial preregistered stage-2 runner. This module never scores."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import gpytorch
import numpy as np
import torch

from stage01_core import tensor_hash
from stage2_core import (Branch, RawBranch, Budget, BudgetExceeded, cpu_setup,
                         train_block, predict_timed, snapshot, restore, check_forks)
from stage2_data import Stage2Adapter


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists():raise FileExistsError(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def load_state(path):return torch.load(path,map_location='cpu',weights_only=False)


def verify_freeze(root):
    frozen=read(root/'frozen_manifest.json')
    for name,value in frozen['files'].items():
        if sha(root/name)!=value:raise ValueError('Frozen file changed: '+name)
    codes=read(root/'code_manifest.json')
    for name,value in codes['files'].items():
        if sha(root/name)!=value:raise ValueError('Frozen code changed: '+name)
    return frozen


class Runner:
    def __init__(self,root,resume=False):
        self.root=Path(root);self.resume=resume
        self.frozen=verify_freeze(self.root)
        if (self.root/'predictions_manifest.json').exists():
            raise RuntimeError('Predictions are already sealed; training cannot resume after scoring exposure')
        self.adapter=Stage2Adapter(self.root,self.root/'frozen_manifest.json')
        self.contexts=read(self.root/'worklist.json')['contexts']
        self.subsets=read(self.root/'subsets.json')['subsets']
        self.budget=Budget(self.root/'resource_ledger.json',limit=7200)
        if not resume:
            if self.budget.updates or self.budget.verification_updates:
                raise RuntimeError('Existing resource ledger requires explicit resume')
            verification=read(self.root/'verification/preflight_summary.json')
            if not verification['pass']:raise RuntimeError('Targeted preflight failed')
            self.budget.charge(verification['verification_seconds'],kind='verification',
                               updates=verification['synthetic_optimizer_updates'])
            self.budget.persist()
        self.traces=[]
        self.summaries=[]
        self.started=time.perf_counter()
        self.forecast_samples={}
        for folder in ['checkpoints','states','predictions','training','diagnostics','audit']:
            (self.root/folder).mkdir(exist_ok=True)
        if resume:
            for path in sorted((self.root/'diagnostics').glob('*_block.json')):
                self.summaries.append(read(path))
            p=self.root/'training_traces.json'
            if p.exists():self.traces=read(p)

    def block(self,model,mode,steps,context_id,method,name):
        key=f'{context_id}_{method}_{name}'
        checkpoint=self.root/'checkpoints'/f'{key}.pt'
        record=self.root/'diagnostics'/f'{key}_block.json'
        if record.exists():
            if not self.resume:raise FileExistsError(record)
            saved=read(record)
            if saved['completed']!=steps:raise ValueError('Completed block record has wrong step count')
            return restore(load_state(checkpoint)),saved
        previous=None
        if checkpoint.exists():
            if not self.resume:raise FileExistsError(checkpoint)
            previous=load_state(checkpoint)
            model=restore(previous)
        result=train_block(model,mode,steps,self.traces,context_id,method,name,self.budget,
                           checkpoint_path=checkpoint,resume=previous,
                           diagnostic_every=100,checkpoint_every=100)
        metadata={k:v for k,v in result.items() if k!='state'}
        metadata.update(context_id=context_id,method=method,block=name,mode=mode,
                        planned_steps=steps,state_hash=model.fingerprint(),checkpoint=str(checkpoint.relative_to(self.root)))
        write(record,metadata,exclusive=True)
        self.summaries.append(metadata)
        write(self.root/'training_traces.json',self.traces)
        csv_write(self.root/'training_traces.csv',self.traces)
        csv_write(self.root/'block_summaries.csv',self.summaries)
        self.budget.persist()
        return model,metadata

    def prediction(self,model,query,training,ctx,method,common_seconds,branch_seconds):
        unit_id=f'{ctx["subset_id"]}_RAW16' if method=='RAW16' else f'{ctx["context_id"]}_{method}'
        path=self.root/'predictions'/f'{unit_id}.json'
        if path.exists():
            if not self.resume:raise FileExistsError(path)
            return read(path)
        # Query metadata has no observation, uncertainty or ranking fields.
        query_tensor=torch.tensor(query['x_raw'],dtype=torch.double)
        scaler=training['scaler']
        query_tensor=(query_tensor-torch.tensor(scaler['x_mean']))/torch.tensor(scaler['x_scale'])
        prior_prediction_seconds=self.budget.prediction_seconds
        mean,variance=predict_timed(model,query_tensor,self.budget)
        prediction_seconds=self.budget.prediction_seconds-prior_prediction_seconds
        center,scale=float(scaler['y_mean']),float(scaler['y_scale'])
        if float(variance.min()) < -1e-8:raise FloatingPointError('Standardized latent variance below -1e-8')
        state=snapshot(model)
        state_path=self.root/'states'/f'{unit_id}.pt'
        torch.save(state,state_path)
        extra=float(model.gp.likelihood.second_noise.detach().item())
        payload={'unit_id':unit_id,'context_id':ctx['context_id'],'subset_id':ctx['subset_id'],
            'fold':ctx['fold'],'sample_seed':ctx['sample_seed'],'n':ctx['n'],
            'model_seed':None if method=='RAW16' else ctx['model_seed'],'method':method,
            'candidate_ids':list(query['ids']),'architecture_fingerprints':list(query['architectures']),
            'mean_standardized':mean.tolist(),'mean_raw':(mean*scale+center).tolist(),
            'latent_variance_standardized':variance.tolist(),'latent_variance_raw':(variance*scale**2).tolist(),
            'y_center':center,'y_scale':scale,'extra_noise_variance_standardized':extra,
            'endpoint_state_hash':model.fingerprint(),'state_path':str(state_path.relative_to(self.root)),
            'state_file_sha256':sha(state_path),'training_ids_sha256':digest(training['ids']),
            'training_tensor_sha256':tensor_hash({'x':model.x,'y':model.y,'known':model.known}),
            'query_labels_or_SE_present':False,'negative_roundoff_variance_count':int((variance<0).sum()),
            'input_dimension':int(model.gp.train_inputs[0].shape[-1]),
            'units':{'mean_raw':'accuracy fraction','latent_variance_raw':'accuracy squared',
                     'latent_variance_standardized':'standardized y variance; raw unclamped posterior diagonal',
                     'extra_noise_variance_standardized':'variance, not std'},
            'cost':{'common_training_seconds':common_seconds,'branch_training_seconds':branch_seconds,
                    'prediction_seconds':prediction_seconds,
                    'standalone_fit_predict_seconds':common_seconds+branch_seconds+prediction_seconds},
            'finished_at_utc':now()}
        write(path,payload,exclusive=True)
        self.budget.persist()
        return payload

    def tensors(self,training):
        return tuple(torch.tensor(training[k],dtype=torch.double) for k in
                     ['x_standardized','y_standardized','known_variance_standardized'])

    def context(self,ctx):
        context_id=ctx['context_id'];subset_id=ctx['subset_id']
        subset=self.subsets[subset_id]
        training=self.adapter.release_training(subset['ids'])
        if training['ids']!=subset['ids'] or len(training['ids'])!=ctx['n']:
            raise ValueError('Training release/order mismatch')
        # The adapter only passes these selected observations to this fit.
        train_path=self.root/'training'/f'{subset_id}.json'
        if train_path.exists():
            prior=read(train_path)
            for key in ['ids','x_standardized','y_standardized','known_variance_standardized','scaler']:
                if prior[key]!=training[key]:raise ValueError('Subset release changed across model seeds')
        else:write(train_path,training,exclusive=True)
        query=self.adapter.query_inputs(ctx['fold'])
        if set(training['ids'])&set(query['ids']):raise ValueError('Training/query ID leakage')
        x,y,known=self.tensors(training)
        raw_seconds=0.0
        if ctx['run_raw']:
            cpu_setup(0)
            raw=RawBranch(x,y,known)
            raw,a=self.block(raw,'gp',300,context_id,'RAW16','raw_initial')
            raw,b=self.block(raw,'gp',1000,context_id,'RAW16','raw_continue')
            raw_seconds=a['seconds']+b['seconds']
            self.prediction(raw,query,training,ctx,'RAW16',0.0,raw_seconds)
            del raw
        cpu_setup(ctx['model_seed'])
        common=Branch(x,y,known,ctx['model_seed'])
        random_state=common.fingerprint()
        common,a=self.block(common,'regression',1000,context_id,'COMMON','regression')
        common,b=self.block(common,'gp',300,context_id,'COMMON','gp')
        common_seconds=a['seconds']+b['seconds']
        common_path=self.root/'states'/f'{context_id}_COMMON.pt'
        torch.save(snapshot(common),common_path)
        check_start=time.perf_counter()
        forks=check_forks(common)
        self.budget.charge(time.perf_counter()-check_start,kind='prediction')
        if not forks['checks']['pass']:raise AssertionError('Common fork equality/storage check failed')
        fork_checks={'context_id':context_id,'random_state_hash':random_state,'common_state_hash':common.fingerprint(),
                     'common_state_file':str(common_path.relative_to(self.root)),**forks['checks']}
        write(self.root/'diagnostics'/f'{context_id}_fork.json',fork_checks)
        total_branch_seconds=0.0
        for method in ['FROZEN','REG','DKL']:
            model=forks['branches'][method]
            map_before=tensor_hash(model.mapper.state_dict())
            head_before=tensor_hash(model.mapper.regressor.state_dict())
            if method=='FROZEN':
                model,a=self.block(model,'gp',1000,context_id,method,'gp')
                branch_seconds=a['seconds']
                if tensor_hash(model.mapper.state_dict())!=map_before:raise AssertionError('Frozen mapper changed')
            elif method=='REG':
                model,a=self.block(model,'regression',1000,context_id,method,'regression')
                map_fixed=tensor_hash(model.mapper.state_dict())
                model,b=self.block(model,'gp',1000,context_id,method,'gp')
                branch_seconds=a['seconds']+b['seconds']
                if tensor_hash(model.mapper.state_dict())!=map_fixed:raise AssertionError('REG GP stage changed mapper')
            else:
                model,a=self.block(model,'joint',1000,context_id,method,'joint')
                branch_seconds=a['seconds']
                if tensor_hash(model.mapper.regressor.state_dict())!=head_before:raise AssertionError('DKL head changed')
            self.prediction(model,query,training,ctx,method,common_seconds,branch_seconds)
            total_branch_seconds+=branch_seconds
        self.forecast_samples.setdefault(ctx['n'],{'mapped_seconds':common_seconds+total_branch_seconds,
                                                   'raw_seconds':raw_seconds})

    def progress(self,index):
        remaining=None
        if set(self.forecast_samples)=={20,50}:
            remaining=sum(self.forecast_samples[c['n']]['mapped_seconds']+
                          (self.forecast_samples[c['n']]['raw_seconds'] if c['run_raw'] else 0)
                          for c in self.contexts[index+1:])
        value={'completed_contexts':index+1,'planned_contexts':len(self.contexts),
               'completed_endpoints':len(list((self.root/'predictions').glob('*.json'))),
               'formal_updates_so_far':self.budget.updates,'fit_predict_seconds':self.budget.seconds,
               'training_seconds':self.budget.training_seconds,'prediction_seconds':self.budget.prediction_seconds,
               'verification_seconds':self.budget.verification_seconds,'verification_updates':self.budget.verification_updates,
               'estimated_remaining_fit_seconds_from_first_n20_n50':remaining,'budget_seconds':7200,
               'experiment_wall_seconds':time.perf_counter()-self.started,'at_utc':now()}
        write(self.root/'progress.json',value)
        print(json.dumps(value),flush=True)

    def run(self):
        failure=None;fit_status='COMPLETE';phase2_status='COMPLETE_VALID'
        try:
            for index,ctx in enumerate(self.contexts):
                self.budget.check()
                self.context(ctx)
                self.progress(index)
        except Exception as exc:
            failure={'type':type(exc).__name__,'message':str(exc),'context':ctx,
                     'traceback':traceback.format_exc(),'at_utc':now()}
            fit_status='PARTIAL'
            phase2_status='PARTIAL' if isinstance(exc,BudgetExceeded) else 'INVALID'
            write(self.root/'failure.json',failure,exclusive=True)
        finally:
            self.budget.persist()
            write(self.root/'training_traces.json',self.traces)
            csv_write(self.root/'training_traces.csv',self.traces)
            csv_write(self.root/'block_summaries.csv',self.summaries)
        expected=[]
        for ctx in self.contexts:
            if ctx['run_raw']:expected.append(f'{ctx["subset_id"]}_RAW16')
            expected.extend(f'{ctx["context_id"]}_{m}' for m in ['FROZEN','REG','DKL'])
        files=[]
        for path in sorted((self.root/'predictions').glob('*.json')):
            p=read(path)
            files.append({k:p[k] for k in ['unit_id','context_id','subset_id','fold','sample_seed','n','model_seed','method']})
            files[-1].update(path=str(path.relative_to(self.root)),sha256=sha(path),
                             state_path=p['state_path'],state_sha256=p['state_file_sha256'],endpoint_state_hash=p['endpoint_state_hash'])
        existing={f['unit_id'] for f in files}
        if fit_status=='COMPLETE' and (len(files)!=200 or existing!=set(expected) or self.budget.updates!=344000):
            phase2_status='INVALID';fit_status='PARTIAL'
            failure={'type':'MatrixMismatch','expected_endpoints':200,'actual_endpoints':len(files),
                     'expected_updates':344000,'actual_updates':self.budget.updates}
            write(self.root/'matrix_failure.json',failure,exclusive=True)
        manifest={'status':'SEALED','sealed':True,'sealed_at_utc':now(),'phase2_fit_status':fit_status,
                  'phase2_status':phase2_status,'expected_endpoint_count':200,'expected_unit_ids':expected,
                  'files':files,'missing_unit_ids':sorted(set(expected)-existing),
                  'protocol_path':'protocol.json','protocol_sha256':sha(self.root/'protocol.json'),
                  'code_manifest_sha256':sha(self.root/'code_manifest.json'),
                  'worklist_path':'worklist.json','worklist_sha256':sha(self.root/'worklist.json'),
                  'fitting_and_prediction_complete_before_any_query_scoring':True,'failure':failure,
                  'resources':{'training_seconds':self.budget.training_seconds,'prediction_seconds':self.budget.prediction_seconds,
                               'fit_predict_seconds':self.budget.seconds,'formal_optimizer_updates':self.budget.updates,
                               'verification_seconds':self.budget.verification_seconds,'verification_updates':self.budget.verification_updates,
                               'experiment_wall_seconds':time.perf_counter()-self.started,'budget_seconds':7200},
                  'next_stage_authorized':False}
        write(self.root/'predictions_manifest.json',manifest,exclusive=True)
        print(json.dumps({'SEALED':True,'phase2_status':phase2_status,'endpoints':len(files),
                          'resources':manifest['resources'],'failure':failure}),flush=True)
        if phase2_status=='INVALID':raise RuntimeError('Invalid experiment; inspect preserved failure before scoring')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_interop_threads(1);cpu_setup(0)
    env={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'gpytorch':gpytorch.__version__,
         'numpy':np.__version__,'platform':platform.platform(),'device':'cpu','threads':1,
         'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS'),
         'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),'started_at_utc':now()}
    write(args.root/'environment.json',env,exclusive=not args.resume)
    Runner(args.root,args.resume).run()


if __name__=='__main__':main()
