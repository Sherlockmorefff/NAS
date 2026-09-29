"""A-only serial read-only geometry runner. No optimizer or scoring imports.

Reads only frozen public inputs, selected-training snapshots and named source
states/predictions. All new model restore/forward/predict work is charged to the
single 1200-second ledger. Old sealed predictions are reused without predicting.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from stage01_core import cpu_setup, tensor_hash
from stage2_core import restore as restore_historical, numerical_settings
from residual_core import Budget, BudgetExceeded
from residual_diagnostics import geometry_from_embeddings, geometry_comparison, gp_parameters, _detach_shared_modules

REPRESENTATIONS=('RAW_X','OLD_RAW16','COMMON_REG','COMMON_GP','OLD_DKL_1000','OLD_DKL_4000')
FORBIDDEN={'labels','query_labels','query_se','query_SE','validation_mean','validation_se',
           'validation_seeds','true_y','y_true','targets','ground_truth','query_y','observed_y',
           'champion_id','test_accuracy','test_acc'}


def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');tmp.replace(p)
def no_truth(v):
    if isinstance(v,dict):
        if FORBIDDEN & set(v):raise ValueError('Truth field in unlabelled historical output')
        for x in v.values():no_truth(x)
    elif isinstance(v,list):
        for x in v:no_truth(x)


class SourceBlocked(RuntimeError):pass


def verify_sources(root):
    """Validate all declared frozen inputs/code/source bytes before model work."""
    for manifest in ('frozen_manifest.json','code_manifest.json','source_manifest.json'):
        try:values=read(root/manifest)['files']
        except (OSError,KeyError,ValueError) as exc:raise SourceBlocked('Missing/invalid '+manifest) from exc
        for relative,expected in values.items():
            p=(root/relative).resolve()
            if root.resolve() not in p.parents:raise SourceBlocked('Escaping source path')
            if not p.is_file() or sha(p)!=expected:raise SourceBlocked('Missing/mismatched source: '+relative)
    work=read(root/'worklist.json')['contexts']
    source=read(root/'source/stage2/worklist.json')['contexts']
    keys=('context_id','subset_id','fold','sample_seed','n','model_seed','run_raw')
    if len(work)!=60 or [{k:c[k] for k in keys} for c in work]!=[{k:c[k] for k in keys} for c in source]:
        raise SourceBlocked('Historical context identity/order mismatch')
    return work


def expected_units(contexts):
    units=[];seen=set()
    for c in contexts:
        sid,cid=c['subset_id'],c['context_id']
        if sid not in seen:
            seen.add(sid);units.extend([sid+'_RAW_X',sid+'_OLD_RAW16'])
        units.extend(cid+'_'+r for r in REPRESENTATIONS[2:])
    if len(units)!=280 or len(set(units))!=280:raise SourceBlocked('Historical expected matrix size')
    return units


def _field(d,path):
    for key in path.split('.'):
        if d is None or not isinstance(d,dict):return None
        d=d.get(key)
    return d


def summarize_geometry(rows):
    """Same nested equal weighting as scientific summaries, no truth values."""
    metrics={
        's_h':'geometry.s_h','ell_over_s_h':'geometry.ell_over_s_h',
        'TT_median_distance':'geometry.distance_stats.TT.median','QT_median_distance':'geometry.distance_stats.QT.median',
        'TT_p10_distance':'geometry.distance_stats.TT.p10','TT_p90_distance':'geometry.distance_stats.TT.p90',
        'QT_p10_distance':'geometry.distance_stats.QT.p10','QT_p90_distance':'geometry.distance_stats.QT.p90',
        'TT_zero_fraction':'geometry.distance_stats.TT.zero_fraction','QT_zero_fraction':'geometry.distance_stats.QT.zero_fraction',
        'TT_effective_distance_median':'geometry.effective_distance_stats.TT.median',
        'QT_effective_distance_median':'geometry.effective_distance_stats.QT.median',
        'outputscale':'geometry.gp_parameters.outputscale','lengthscale':'geometry.gp_parameters.lengthscale',
        'extra_noise_variance':'geometry.gp_parameters.extra_noise_variance_standardized',
        'TT_shape_change_vs_raw':'comparisons_to_raw.shape_change.TT','QT_shape_change_vs_raw':'comparisons_to_raw.shape_change.QT',
        'nearest_changed_vs_raw':'comparisons_to_raw.nearest_changed_fraction',
        'five_neighbor_set_changed_vs_raw':'comparisons_to_raw.five_neighbor_set_changed_fraction',
        'TT_shape_change_vs_previous':'comparisons_to_previous.shape_change.TT','QT_shape_change_vs_previous':'comparisons_to_previous.shape_change.QT',
        'TT_K_change_vs_previous':'comparisons_to_previous.K_relative_change.TT',
        'QT_K_change_vs_previous':'comparisons_to_previous.K_relative_change.QT',
        'TT_R_change_vs_previous':'comparisons_to_previous.R_relative_change.TT',
        'QT_R_change_vs_previous':'comparisons_to_previous.R_relative_change.QT',
        'outputscale_relative_change_vs_previous':'comparisons_to_previous.outputscale_relative_change'}
    result=[]
    for n in (20,50):
        for rep in REPRESENTATIONS:
            group=[r for r in rows if r['n']==n and r['representation']==rep]
            item={'n':n,'representation':rep,'actual_contexts':len(group),'expected_contexts':10 if rep in REPRESENTATIONS[:2] else 30,
                  'constant_representation_count':sum(r['geometry']['constant_representation'] for r in group),'metrics':{}}
            for name,path in metrics.items():
                sampled=defaultdict(list)
                for r in group:
                    v=_field(r,path)
                    if v is not None:sampled[(r['fold'],r['sample_seed'])].append(float(v))
                folds=defaultdict(list)
                for (fold,_),values in sampled.items():folds[fold].append(statistics.fmean(values))
                folded={str(f):statistics.fmean(v) for f,v in sorted(folds.items())}
                item['metrics'][name]={'fivefold_equal_mean':statistics.fmean(folded.values()) if folded else None,
                    'fold_values':folded,'defined_contexts':sum(len(v) for v in sampled.values()),
                    'defined_folds':len(folded)}
            result.append(item)
    return {'summary':result,'aggregation':'model seeds within fold/sample, samples within fold, five equal folds; raw once per subset',
            'no_new_gate_or_threshold':True,'query_labels_or_SE_present':False}


class HistoricalRunner:
    def __init__(self,root):
        self.root=Path(root);self.budget=Budget(self.root/'resource_ledger.json',limit=1200)
        if (self.root/'historical_manifest.json').exists():raise RuntimeError('A already sealed; refuse recompute/overwrite')
        self.started=time.perf_counter();self.before_seconds=self.budget.seconds;self.rows=[];self.files=[]
        self.contexts=[];self.expected=[];self.active=None
        self.sources=read(self.root/'source_manifest.json')

    def compute(self,fn):
        self.budget.check();start=time.perf_counter()
        try:result=fn()
        finally:
            self.budget.charge(time.perf_counter()-start,kind='prediction');self.budget.persist()
        self.budget.check()
        return result

    def source_path(self,relative):
        if relative not in self.sources['files']:raise SourceBlocked('Undeclared historical source: '+relative)
        p=self.root/relative
        if not p.is_file() or sha(p)!=self.sources['files'][relative]:raise SourceBlocked('Historical source changed: '+relative)
        return p

    def load_state(self,relative):
        # Deserialize disk bytes before timing model reconstruction/forward.
        state=torch.load(self.source_path(relative),map_location='cpu',weights_only=False)
        return state

    def _restore_geometry(self,state,training,query_x,query, fitted_gp,predict):
        rng=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
        try:
            model=restore_historical(state)
            for key,attr in [('x_standardized','x'),('y_standardized','y'),('known_variance_standardized','known')]:
                if not torch.equal(getattr(model,attr),torch.tensor(training[key],dtype=torch.float64)):
                    raise RuntimeError('Historical selected tensors differ: '+key)
            with torch.no_grad(),numerical_settings():
                th,_=model.encode(model.x);qh,_=model.encode(query_x)
                geom=geometry_from_embeddings(th.numpy(),qh.numpy(),training['ids'],query['ids'],gp_parameters(model) if fitted_gp else None)
                mapper_hash=tensor_hash(model.mapper.state_dict()) if hasattr(model,'mapper') else None
                values=None
                if predict:
                    isolated=model.clone();_detach_shared_modules(model,isolated)
                    if isolated.fingerprint()!=model.fingerprint():raise RuntimeError('Historical diagnostic clone identity')
                    mean,var=isolated.predict(query_x)
                    if not bool(torch.isfinite(mean).all() and torch.isfinite(var).all()) or bool((var < -1e-8).any()):
                        raise FloatingPointError('COMMON_GP invalid raw posterior')
                    values={'mean':mean.tolist(),'variance':var.tolist(),'extra':gp_parameters(model)['extra_noise_variance_standardized']}
                return geom,mapper_hash,th.numpy().copy(),qh.numpy().copy(),values,model.fingerprint(),tensor_hash({'x':model.x,'y':model.y,'known':model.known})
        finally:
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])

    def source_prediction(self,relative,training,query,expected_fingerprint):
        p=read(self.source_path(relative));no_truth(p)
        if p['candidate_ids']!=query['ids'] or p['architecture_fingerprints']!=query['architectures']:
            raise SourceBlocked('Historical prediction query order/identity')
        if p['training_ids_sha256']!=digest(training['ids']) or p['endpoint_state_hash']!=expected_fingerprint:
            raise SourceBlocked('Historical prediction state/training identity')
        if p['y_center']!=training['scaler']['y_mean'] or p['y_scale']!=training['scaler']['y_scale']:
            raise SourceBlocked('Historical prediction scaler identity')
        p=copy.deepcopy(p)
        p['historical_reused_prediction']={'path':relative,'sha256':self.sources['files'][relative]}
        return p

    def common_prediction(self,c,training,query,values,fingerprint,tensor_digest,state_relative,model_seconds):
        sc=training['scaler'];mean=np.asarray(values['mean']);var=np.asarray(values['variance'])
        return {**{k:c[k] for k in ('context_id','subset_id','fold','sample_seed','n','model_seed')},
            'unit_id':c['context_id']+'_COMMON_GP','method':'COMMON_GP',
            'candidate_ids':query['ids'],'architecture_fingerprints':query['architectures'],
            'mean_standardized':mean.tolist(),'mean_raw':(mean*sc['y_scale']+sc['y_mean']).tolist(),
            'latent_variance_standardized':var.tolist(),'latent_variance_raw':(var*sc['y_scale']**2).tolist(),
            'y_center':sc['y_mean'],'y_scale':sc['y_scale'],'extra_noise_variance_standardized':values['extra'],
            'endpoint_state_hash':fingerprint,'state_path':state_relative,'state_file_sha256':self.sources['files'][state_relative],
            'training_ids_sha256':digest(training['ids']),'training_tensor_sha256':tensor_digest,'scaler_sha256':digest(sc),
            'input_dimension':8,'completed_steps':1300,'query_labels_or_SE_present':False,
            'negative_roundoff_variance_count':int((var<0).sum()),
            'units':{'mean_raw':'accuracy fraction','latent_variance_standardized':'raw unclamped standardized variance',
                     'extra_noise_variance_standardized':'variance, not std'},
            'cost':{'common_training_seconds':0.,'branch_training_seconds':0.,'prediction_seconds':model_seconds,
                    'standalone_fit_predict_seconds':model_seconds,
                    'accounting_scope':'A-only charged restore/geometry/prediction; historical training reused without new updates'},
            'historical_readonly_prediction':True,'new_optimizer_updates':0}

    def save_row(self,c,rep,geom,prediction,source,raw,previous):
        raw_rep=rep in REPRESENTATIONS[:2]
        uid=(c['subset_id'] if raw_rep else c['context_id'])+'_'+rep
        raw_comparison,previous_comparison=self.compute(lambda:(geometry_comparison(raw,geom),
            None if previous is None else geometry_comparison(previous,geom)))
        row={**{k:c[k] for k in ('context_id','subset_id','fold','sample_seed','n','model_seed')},
             'unit_id':uid,'model_seed':None if raw_rep else c['model_seed'],'representation':rep,**geom,
             'prediction':prediction,'source':source,
             'comparisons_to_raw':raw_comparison,
             'comparisons_to_previous':previous_comparison}
        no_truth(row)
        relative='historical/'+uid+'.json';write(self.root/relative,row)
        self.rows.append(row);self.files.append({'path':relative,'sha256':sha(self.root/relative),
            **{k:row[k] for k in ('unit_id','context_id','subset_id','fold','sample_seed','n','model_seed','representation')}})
        return row

    def context(self,c,seen):
        self.active=c;sid,cid=c['subset_id'],c['context_id']
        training=read(self.root/f'training/{sid}.json');query=read(self.root/f'queries/fold{c["fold"]}.json')
        sc=training['scaler']
        q=(np.asarray(query['x_raw'],dtype=np.float64)-np.asarray(sc['x_mean']))/np.asarray(sc['x_scale'])
        qt=torch.tensor(q,dtype=torch.float64)
        raw=self.compute(lambda:geometry_from_embeddings(training['x_standardized'],q,training['ids'],query['ids']))
        if sid not in seen:
            self.save_row(c,'RAW_X',raw,None,{'training_snapshot':f'training/{sid}.json','sha256':sha(self.root/f'training/{sid}.json')},raw,None)
            relative=f'source/stage2/states/{sid}_RAW16.pt';state=self.load_state(relative)
            geom,_,_,_,_,fingerprint,_=self.compute(lambda:self._restore_geometry(state,training,qt,query,True,False))
            pred=self.source_prediction(f'source/stage2/predictions/{sid}_RAW16.json',training,query,fingerprint)
            self.save_row(c,'OLD_RAW16',geom,pred,{'path':relative,'sha256':self.sources['files'][relative]},raw,raw)
            self.budget.check();seen.add(sid)
        previous=raw;common=None
        for rep,suffix,folder in [('COMMON_REG','COMMON_regression','stage2'),('COMMON_GP','COMMON_gp','stage2'),
                                  ('OLD_DKL_1000','DKL_joint','stage2'),('OLD_DKL_4000','j4000','length')]:
            relative=f'source/{folder}/checkpoints/{cid}_{suffix}.pt';state=self.load_state(relative)
            before=self.budget.seconds
            geom,mapping_hash,th,qh,values,fingerprint,tensor_digest=self.compute(
                lambda:self._restore_geometry(state,training,qt,query,rep!='COMMON_REG',rep=='COMMON_GP'))
            model_seconds=self.budget.seconds-before
            source={'path':relative,'sha256':self.sources['files'][relative],'state_hash':fingerprint,'mapping_state_hash':mapping_hash}
            if rep=='COMMON_REG':common=(mapping_hash,th,qh);prediction=None
            elif rep=='COMMON_GP':
                if mapping_hash!=common[0] or not np.array_equal(th,common[1]) or not np.array_equal(qh,common[2]):
                    raise RuntimeError('COMMON_REG/COMMON_GP mapping parameters or representations differ')
                source['common_reg_mapping_and_representation_exact']=True
                prediction=self.common_prediction(c,training,query,values,fingerprint,tensor_digest,relative,model_seconds)
            else:
                predname=f'{cid}_DKL.json' if rep=='OLD_DKL_1000' else f'{cid}_j4000.json'
                prediction=self.source_prediction(f'source/{folder}/predictions/{predname}',training,query,fingerprint)
            self.save_row(c,rep,geom,prediction,source,raw,previous);previous=geom
            self.budget.check()

    def run(self):
        status='COMPLETE_VALID';failure=None
        try:
            self.contexts=verify_sources(self.root);self.expected=expected_units(self.contexts)
            cpu_setup(0);seen=set()
            for i,c in enumerate(self.contexts):
                self.budget.check();self.context(c,seen)
                print(json.dumps({'historical_contexts_completed':i+1,'historical_rows':len(self.rows),
                    'budget_seconds':self.budget.seconds,'limit':1200}),flush=True)
            if len(self.rows)!=280 or self.budget.updates!=0:raise RuntimeError('Historical matrix/update contract')
        except Exception as exc:
            status='PARTIAL_BUDGET' if isinstance(exc,BudgetExceeded) else ('BLOCKED' if isinstance(exc,(SourceBlocked,FileNotFoundError)) else 'INVALID')
            failure={'type':type(exc).__name__,'message':str(exc),'context':self.active,'traceback':traceback.format_exc()}
        self.budget.persist()
        summary=summarize_geometry(self.rows);write(self.root/'historical_geometry_summary.json',summary)
        missing=sorted(set(self.expected)-{r['unit_id'] for r in self.rows})
        seal={'sealed':True,'execution_status':status,'expected_endpoint_count':280,'actual_endpoint_count':len(self.rows),
            'expected_unit_ids':self.expected,'missing_unit_ids':missing,'files':self.files,
            'protocol_sha256':sha(self.root/'protocol.json'),'code_manifest_sha256':sha(self.root/'code_manifest.json'),
            'source_manifest_sha256':sha(self.root/'source_manifest.json'),
            'geometry_summary_path':'historical_geometry_summary.json','geometry_summary_sha256':sha(self.root/'historical_geometry_summary.json'),
            'failure':failure,'resources':self.budget.as_dict(),'A_new_model_seconds':self.budget.seconds-self.before_seconds,
            'A_wall_seconds':time.perf_counter()-self.started,'formal_updates_actual':0,'query_labels_or_SE_present':False,
            'historical_random_initialization':'not present in named source states; not reconstructed as historical evidence'}
        write(self.root/'historical_manifest.json',seal)
        print(json.dumps({k:seal[k] for k in ('execution_status','actual_endpoint_count','A_new_model_seconds','A_wall_seconds')}),flush=True)
        return 0 if status=='COMPLETE_VALID' else 2


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    raise SystemExit(HistoricalRunner(args.root).run())
