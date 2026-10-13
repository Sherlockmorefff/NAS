"""Explicit budgets, complete visit archives and per-run isolated caches."""
import math
import os
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from .common import write,append,now,digest,ZERO
from .geometry import point_key,norm_u

class BudgetStop(Exception):
    """The fixed vector request cap, not numerical failure."""

def de_trials(pop,rng):
    trials=np.empty_like(pop)
    for i in range(32):
        r1,r2,r3=rng.choice(np.delete(np.arange(32),i),3,replace=False)
        mutant=np.clip(pop[r1]+.5*(pop[r2]-pop[r3]),0.,1.)
        mask=rng.random(16)<.7;mask[rng.integers(16)]=True
        trials[i]=np.where(mask,mutant,pop[i])
    return trials
def better(a,b):return (a['valid'] and not b['valid']) or (a['valid'] and b['valid'] and a['score']>b['score'])

class Run:
    def __init__(self,root,spec,problem,decoder):
        self.root,self.spec,self.problem,self.decoder=root,spec,problem,decoder
        self.folder=root/'runs'/spec['id'];self.folder.mkdir(parents=True,exist_ok=True)
        if (self.folder/'trace.jsonl').exists():raise RuntimeError('Interrupted run cannot restart with a fresh budget')
        self.decoder.reset();self.archive=[];self.actual_cache={};self.inner_cache={}
        self.requests=0;self.post_requests=0;self.gradient_calls=0;self.started=None;self.start_wall=None
        self.times=dict(proposal=0.,decoder=0.,WL=0.,input_transform=0.,posterior_qLogEI=0.,backward=0.,postprocessing=0.,file_IO=0.)
        self.milestones=[];self.time_nodes=[30,60,120,180,300];self.start_results=[]
        self.posterior_baseline=problem.posterior_calls
    def progress(self,phase):
        value={'id':self.spec['id'],'pid':os.getpid(),'state':'running','phase':phase,'requests':self.requests,
          'post_requests':self.post_requests,'start_unix':self.start_wall,'elapsed':0. if self.started is None else time.perf_counter()-self.started,'at':now()}
        started=time.perf_counter();write(self.folder/'progress.json',value);self.times['file_IO']+=time.perf_counter()-started
    def log(self,x):
        started=time.perf_counter();append(self.folder/'trace.jsonl',x);self.times['file_IO']+=time.perf_counter()-started
    def tick(self):
        elapsed=time.perf_counter()-self.started
        best=max((a['score'] for a in self.archive if a['valid']),default=None)
        nodes=[]
        if self.requests in [64,128,256,512]:nodes.append({'kind':'request','node':self.requests})
        while self.time_nodes and elapsed>=self.time_nodes[0]:nodes.append({'kind':'time','node':self.time_nodes.pop(0)})
        for node in nodes:
            m={**node,'requests':self.requests,'elapsed':elapsed,'best_so_far':best,'objective':'relaxed' if self.spec['family']=='A' else 'realized'}
            self.milestones.append(m);self.log({'event':'milestone',**m})
    def actual(self,u):
        started=time.perf_counter();cfg,meta=self.decoder.decode(u);self.times['decoder']+=time.perf_counter()-started
        if cfg is None:return {'valid':False,'score':None,**meta}
        cid=digest(cfg)
        if cid in self.problem.seen:return {'valid':False,'score':None,'reason':'observed_configuration','configuration':cfg,**meta}
        # RAW16 retains the decoded float32 latent, even if two points yield one graph/HP.
        cache_key=cid if self.spec['family']=='WI' else cid+np.asarray(meta['z_float32'][:12],dtype='<f4').tobytes().hex()
        cached=cache_key in self.actual_cache
        if cached:score,unknown=self.actual_cache[cache_key]
        else:
            started=time.perf_counter();score,posterior_time,unknown=self.problem.realized(cfg,meta['z_float32'])
            self.times['posterior_qLogEI']+=posterior_time
            self.times['WL' if self.spec['family']=='WI' else 'input_transform']+=max(0.,time.perf_counter()-started-posterior_time)
            self.actual_cache[cache_key]=(score,unknown)
        return {'valid':True,'score':score,'configuration':cfg,'candidate_id':cid,'outside_original_pool':cid not in self.decoder.pool_ids,
                'unknown_WL_symbols':unknown,'actual_score_cache_hit':cached,**meta}
    def request(self,u,gradient=False):
        if self.requests>=512:raise BudgetStop('global 512 request budget')
        if self.started is None:self.started=time.perf_counter();self.start_wall=time.time()
        self.requests+=1;u=norm_u(u);key=point_key(u)
        self.progress('inner');self.log({'event':'request','number':self.requests,'u_float64':u.tolist(),'vector_key':key,'gradient':gradient})
        if self.spec['family']=='A':
            if key in self.inner_cache and not gradient:value=self.inner_cache[key];grad=None;cached=True
            else:
                value,grad,forward,backward=self.problem.relaxed(u,gradient)
                self.times['posterior_qLogEI']+=forward;self.times['backward']+=backward
                if gradient:self.gradient_calls+=1
                self.inner_cache[key]=value;cached=False
            result={'valid':True,'score':value,'relaxed':True,'inner_score_cache_hit':cached}
        else:result=self.actual(u);grad=None
        entry={'number':self.requests,'u_float64':u.tolist(),'vector_key':key,'elapsed':time.perf_counter()-self.started,**result}
        self.archive.append(entry);self.log({'event':'result',**entry});self.tick()
        return entry,grad
    def optimize(self):
        rng=np.random.default_rng(self.spec['optimizer_seed']);started=time.perf_counter()
        pop=qmc.LatinHypercube(d=16,seed=self.spec['initial_seed']).random(32)
        self.times['proposal']+=time.perf_counter()-started
        initial=[self.request(u)[0] for u in pop]
        if self.spec['method']=='DE':
            scores=initial
            for generation in range(15):
                started=time.perf_counter();trials=de_trials(pop,rng);self.times['proposal']+=time.perf_counter()-started
                values=[self.request(u)[0] for u in trials]
                for i in range(32):
                    if better(values[i],scores[i]):pop[i]=trials[i];scores[i]=values[i]
                self.log({'event':'DE_generation','generation':generation+1,'requests':self.requests,'population':pop.tolist(),
                     'scores':[v['score'] for v in scores],'feasible':[v['valid'] for v in scores],'rng':rng.bit_generator.state})
            if self.requests!=512:raise AssertionError('DE must use exactly 32+15*32 requests')
        elif self.spec['method']=='RANDOM':
            for i in range(480):
                started=time.perf_counter();u=rng.random(16);self.times['proposal']+=time.perf_counter()-started
                self.request(u)
        else:
            starts=sorted(initial,key=lambda v:(-v['score'],v['vector_key']))[:8]
            for index,start in enumerate(starts):
                calls=0
                def fun(u):
                    nonlocal calls
                    if calls>=60:raise BudgetStop('60 fun+jac requests for this start')
                    calls+=1;v,grad=self.request(u,gradient=True)
                    return -v['score'],-grad
                try:
                    result=minimize(fun,np.asarray(start['u_float64']),jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*16,
                         options={'maxiter':60,'maxls':20,'ftol':1e-9,'gtol':1e-6})
                    reason='converged' if result.success else 'max_iterations' if result.status==1 else 'optimizer_failure'
                    record={'start':index,'requests':calls,'reason':reason,'status':int(result.status),'message':str(result.message),
                            'iterations':int(result.nit),'function_evaluations':int(result.nfev)}
                    self.start_results.append(record);self.log({'event':'gradient_start',**record})
                    if reason=='optimizer_failure':raise FloatingPointError('L-BFGS-B abnormal termination: '+str(result.message))
                except BudgetStop as exc:
                    record={'start':index,'requests':calls,'reason':'budget_stop','message':str(exc)}
                    self.start_results.append(record);self.log({'event':'gradient_start',**record})
    def finish(self):
        phase_started=time.perf_counter();before_detail=sum(self.times.values())
        self.progress('postprocessing');unique={}
        for row in sorted((a for a in self.archive if a['valid']),key=lambda a:(-a['score'],a['vector_key'])):
            unique.setdefault(row['vector_key'],row)
        chosen=list(unique.values())[:32];post=[];seen_cfg=set()
        for row in chosen:
            self.post_requests+=1;self.progress('postprocessing')
            self.log({'event':'post_request','number':self.post_requests,'vector_key':row['vector_key'],'u_float64':row['u_float64']})
            started=time.perf_counter();actual=self.actual(row['u_float64'])
            # Detailed decoder/WL/posterior time is counted in its own nonoverlapping category.
            item={'u_float64':row['u_float64'],'vector_key':row['vector_key'],'inner_score':row['score'],**actual}
            if actual['valid']:
                item['duplicate_configuration']=actual['candidate_id'] in seen_cfg;seen_cfg.add(actual['candidate_id'])
            post.append(item);self.log({'event':'post_result',**item})
        valid=[x for x in post if x['valid']]
        returned=min(valid,key=lambda x:(-x['score'],x['candidate_id'],x['vector_key'])) if valid else None
        state=('complete_early' if self.spec['method']=='GRADIENT' and self.requests<512 else 'complete') if returned else 'no_valid_return'
        relaxed=max((x['score'] for x in self.archive if x['valid']),default=None) if self.spec['family']=='A' else None
        self.times['postprocessing']+=max(0.,time.perf_counter()-phase_started-(sum(self.times.values())-before_detail))
        result={**self.spec,**ZERO,'state':state,'started_at':self.start_wall,'ended_at':now(),'elapsed_seconds':time.perf_counter()-self.started,
          'requests':self.requests,'post_requests':self.post_requests,'post_valid':len(valid),'post_unique_configurations':len(seen_cfg),
          'inner_valid':sum(x['valid'] for x in self.archive),'inner_unique_vectors':len({x['vector_key'] for x in self.archive}),
          'inner_unique_valid_configurations':len({x.get('candidate_id') for x in self.archive if x['valid'] and 'candidate_id' in x}),
          'inner_validity_denominator':'only postprocessing for A; all inner vectors for WI',
          'gradient_calls':self.gradient_calls,'posterior_calls':self.problem.posterior_calls-self.posterior_baseline,
          'decoder_calls':self.decoder.decode_calls,'frozen_architecture_map_hits':self.decoder.map_hits,'decoder_cache_hits':self.decoder.cache_hits,
          'times':self.times,'relaxed_best':relaxed,'realized_best':None if returned is None else returned['score'],
          'relaxed_minus_realized':None if relaxed is None or returned is None else relaxed-returned['score'],
          'returned':returned,'post_candidates':post,'milestones':self.milestones,'gradient_starts':self.start_results,
          'GP_parameter_hash_before':self.problem.learned_before,'GP_parameter_hash_after':self.problem.learned_before,
          'frozen_y_scale':self.problem.scaler['y_scale'],'MC_seed':self.problem.mc_seed,'MC_base_tensor_hash':self.problem.manifest()['MC_base_tensor_hash'],
          'stop_reason':'normal_early_convergence' if state=='complete_early' else 'fixed_budget_completed' if state=='complete' else 'no_valid_unseen_post_candidate'}
        if returned:
            result['improvement_approx_validation']=self.problem.scaler['y_scale']*math.exp(returned['score'])
        else:result['improvement_approx_validation']=None
        self.problem.assert_unchanged();write(self.folder/'result.json',result)
        return result
    def execute(self):
        self.optimize();return self.finish()
