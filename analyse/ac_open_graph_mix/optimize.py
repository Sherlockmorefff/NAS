"""Fixed A-gradient/C graph-HP rules, durable requests and bounded recovery."""
import hashlib
import itertools
import time
import traceback
import numpy as np
import torch
from scipy.optimize import minimize
from .common import *
from . import graphs

class BudgetStop(Exception):
    """A consumed fixed request allowance."""

class OpenDecoder:
    def __init__(self,root,g):
        from architecture_diagnostics import load_joint_vae
        self.old={};self.cache={};self.calls=0;self.hits=0
        for row in read(inputs(root,g)/'candidates.json'):
            key=np.asarray(row['x_raw'][:12],dtype='<f4').tobytes();cfg=row['configuration'];arch={k:cfg[k] for k in ['operations','edges','effective_layers']}
            if key in self.old and digest(arch)!=digest(self.old[key]):raise ValueError('Conflicting original latent mapping')
            self.old[key]=arch
        old=torch.get_default_dtype();torch.set_default_dtype(torch.float32)
        try:self.vae=load_joint_vae(read(root/'PROTOCOL.json')['vae']['path'],'global4',device='cpu').float().eval()
        finally:torch.set_default_dtype(old)
        for p in self.vae.parameters():p.requires_grad_(False)
    def decode(self,u):
        from eval_utils import _decode_arch,decode_hp_by_mode
        u=np.clip(np.asarray(u,dtype=np.float64),0.,1.);z=np.r_[-2.5+5*u[:12],u[12:]].astype('<f4');key=z[:12].tobytes()
        dseed=int.from_bytes(hashlib.sha256(b'AC_open_graph_mix_v1_20261012\0'+key).digest()[:4],'little')
        if key in self.old:arch=self.old[key];self.hits+=1;source='frozen_pool'
        elif key in self.cache:arch=self.cache[key];self.hits+=1;source='round_cache'
        else:
            old=torch.get_default_dtype();torch.set_default_dtype(torch.float32)
            try:arch=_decode_arch(self.vae,torch.tensor(z[:12],dtype=torch.float32),torch.device('cpu'),n_trials=5,decoder_seed=dseed)
            finally:torch.set_default_dtype(old)
            self.calls+=1;self.cache[key]=arch;source='VAE5'
        if arch is None:return None,{'reason':'all_Identity_or_invalid_decode','decoder_seed':dseed,'source':source}
        hp=decode_hp_by_mode(torch.tensor(z,dtype=torch.float32),12,arch,'global4',-4.,-1.5,.1,.6)
        cfg={**arch,**graphs.FIXED,'lr':hp['lr'],'dropout':hp['dropout'],'hidden_dim':hp['hidden_dim'],'l2':hp['weight_decay']}
        cfg['edges']=sorted([list(map(int,e)) for e in cfg['edges']])
        try:graphs.valid(cfg)
        except ValueError as exc:return None,{'reason':str(exc),'decoder_seed':dseed,'source':source}
        return {'candidate_id':digest(cfg),'architecture_id':graphs.aid(cfg),'configuration':cfg,'latent':z[:12].astype(float).tolist(),
                 'latent_source':'generated_float32','proposal_u_float64':u.tolist()}, {'decoder_seed':dseed,'source':source}

class Round:
    def __init__(self,root,method,g,obs,p):
        self.root,self.method,self.g,self.obs,self.p=root,method,g,obs,p;self.step=len(obs)-49
        self.fd=folder(root,method,g)/'acquisition'/f'{self.step:03d}';self.fd.mkdir(parents=True,exist_ok=True)
        self.sp=self.fd/'state.json';self.archive=self.fd/'archive.jsonl';self.seen={o['candidate_id'] for o in obs}
        self.mc=read(inputs(root,g)/'seeds.json')['qlogei_mc'][self.step-1]
        self.salt={'A_raw':proposal_seed(g,11,self.step),'A_fallback':proposal_seed(g,12,self.step),
          'C_random':proposal_seed(g,21,self.step),'C_mutation':proposal_seed(g,22,self.step),'C_HP':proposal_seed(g,23,self.step),'C_fallback':proposal_seed(g,24,self.step)}
        self.state=read(self.sp) if self.sp.exists() else {'method':method,'group':g,'step':self.step,'requests':0,'main_requests':0,'post_requests':0,'fallback_requests':0,
          'consumed_seconds':0.,'phase':'main','raw_done':0,'completed_starts':[],'start_calls':{},'start_vectors':{},'post_keys':[],
          'fallback_done':0,'decoder_calls':0,'decoder_hits':0,'encoder_calls':0,'events':[],'times':{'proposal':0.,'decoder':0.,'landing_dedup':0.,'file_IO':0.},'seeds':self.salt}
        if self.state['method']!=method or self.state['step']!=self.step:raise ValueError('Round resume identity changed')
        self.prior_seconds=self.state['consumed_seconds']
        if 'recovery_unix' in self.state:
            self.prior_seconds+=time.time()-self.state.pop('recovery_unix')
            self.state['events'].append({'event':'recovery_import_and_restore_time_counted','cumulative_seconds':self.prior_seconds})
        self.begin=time.monotonic();self.state['active_unix']=time.time();self.state['segment_prior_seconds']=self.prior_seconds;self.rows=[]
        if self.archive.exists():
            for line in self.archive.read_text().splitlines():
                try:self.rows.append(__import__('json').loads(line))
                except __import__('json').JSONDecodeError:raise ValueError('Incomplete archive record; preserve and repair torn final line explicitly')
        self.prior_meters=dict(self.state.get('meters',{}));self.decoder=None
        self.state['prior_decoder_calls']=self.state['decoder_calls'];self.state['prior_decoder_hits']=self.state['decoder_hits'];self.save()
    def elapsed(self):return self.prior_seconds+time.monotonic()-self.begin
    def save(self):
        self.state['consumed_seconds']=self.elapsed()
        self.state['meters']={k:self.prior_meters.get(k,0)+getattr(self.p,k,0) for k in ['posterior_points','gradients','forward_seconds','backward_seconds','WL_seconds']}
        start=time.perf_counter();write(self.sp,self.state);self.state['times']['file_IO']+=time.perf_counter()-start
    def log(self,row):
        start=time.perf_counter();append(self.archive,row);self.rows.append(row);self.state['times']['file_IO']+=time.perf_counter()-start
    def count(self,phase,count=1,persist=True):
        cap=1856 if self.method=='A' else 1920;fall=128 if self.method=='A' else 64
        if self.state['requests']+count>2048:raise BudgetStop('2048 round request cap')
        if phase=='main' and self.state['main_requests']+count>cap:raise BudgetStop('normal main phase cap')
        if phase=='post' and self.state['post_requests']+count>64:raise BudgetStop('64 postprocessing cap')
        if phase=='fallback' and self.state['fallback_requests']+count>fall:raise BudgetStop('fallback cap')
        self.state['requests']+=count;self.state[{'main':'main_requests','post':'post_requests','fallback':'fallback_requests'}[phase]]+=count
        if persist:self.save()
    def request(self,u,gradient=False,graph=None,hidden=None,l2=None,start_id=None):
        self.count('main',persist=False);u=np.clip(np.asarray(u,dtype=np.float64),0,1)
        if start_id is not None:
            self.state['start_calls'][start_id]=self.state['start_calls'].get(start_id,0)+1;self.state['start_vectors'][start_id]=u.tolist()
        self.save()
        pending={'event':'requested','request':self.state['requests'],'phase':'main','u':u.tolist(),'gradient':gradient,'start_id':start_id,'elapsed':self.elapsed()}
        self.log(pending)
        if self.method=='A':score,derivative=self.p.evaluate(u,gradient);key=np.r_[-2.5+5*u[:12],u[12:]].astype('<f4').tobytes().hex();extra={}
        else:
            cfg=graphs.config(graph['architecture'],hidden,l2,u);index=self.p.graph_index(cfg)
            score,derivative=self.p.evaluate(u,gradient,index,hidden,l2);key=digest(cfg);extra={'graph':graph,'hidden':hidden,'l2':l2,'configuration_float64':cfg}
        row={'event':'scored','request':self.state['requests'],'phase':'main','u':u.tolist(),'score':score,'key':key,'start_id':start_id,
             'elapsed':self.elapsed(),**extra};self.log(row);return row,derivative
    def refine(self,starts,maxcalls):
        for sid,start in starts:
            if sid in self.state['completed_starts']:continue
            prior=self.state['start_calls'].get(sid,0)
            if prior>=maxcalls:self.state['completed_starts'].append(sid);self.save();continue
            initial=np.asarray(self.state['start_vectors'].get(sid,start['u']))
            if prior:self.state['events'].append({'event':'resume_LBFGSB_from_saved_vector','start':sid,'used':prior,'remaining':maxcalls-prior,'same_iteration_trajectory_claimed':False})
            def fun(u):
                if self.state['start_calls'].get(sid,0)>=maxcalls:raise BudgetStop('per-start request cap')
                if self.elapsed()>=270:raise BudgetStop('main 270s phase deadline')
                row,grad=self.request(u,True,start.get('graph'),start.get('hidden'),start.get('l2'),sid)
                return -row['score'],-grad
            try:
                out=minimize(fun,initial,jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*len(initial),
                    options={'maxiter':maxcalls,'maxls':20,'ftol':1e-9,'gtol':1e-6})
                self.state['events'].append({'event':'optimizer_end','start':sid,'status':int(out.status),'message':str(out.message),'success':bool(out.success)})
            except BudgetStop as exc:self.state['events'].append({'event':'budget_stop','start':sid,'message':str(exc)})
            except Exception as exc:self.state['events'].append({'event':'optimizer_exception','start':sid,'type':type(exc).__name__,'message':str(exc)})
            self.state['completed_starts'].append(sid);self.save()
            if self.elapsed()>=270:break
    def main_A(self):
        engine=torch.quasirandom.SobolEngine(16,scramble=True,seed=self.salt['A_raw']);design=engine.draw(256,dtype=torch.float64).numpy()
        for i in range(self.state['raw_done'],256):
            if self.elapsed()>=270:break
            self.state['raw_done']=i+1;self.save();self.request(design[i])
        sp=self.fd/'starts.json'
        if sp.exists():starts=read(sp)
        else:
            raw=[r for r in self.rows if r['event']=='scored' and r.get('start_id') is None];unique={}
            for r in sorted(raw,key=lambda r:(-r['score'],r['key'])):unique.setdefault(r['key'],r)
            starts=[[f'A_{i}',r] for i,r in enumerate(list(unique.values())[:8])];write(sp,starts)
        self.refine(starts,200)
    def main_C(self):
        path=self.fd/'graphs.json'
        if path.exists():proposals=read(path)['graphs']
        else:
            start=time.perf_counter();proposals,trace,rng=graphs.propose(self.obs,{'random':self.salt['C_random'],'mutation':self.salt['C_mutation']})
            self.state['times']['proposal']+=time.perf_counter()-start;write(path,{'graphs':proposals,'attempts':trace,'RNG_state':rng})
        if not proposals:raise RuntimeError('No legal graph in bounded C graph streams')
        design=torch.quasirandom.SobolEngine(2,scramble=True,seed=self.salt['C_HP']).draw(2,dtype=torch.float64).numpy()
        combos=list(itertools.product(graphs.HIDDEN,graphs.L2));total=len(proposals)*60
        for i in range(self.state['raw_done'],total):
            if self.elapsed()>=270:break
            graph=proposals[i//60];hidden,l2=combos[(i%60)//2];v=design[i%2]
            self.state['raw_done']=i+1;self.save();self.request(v,graph=graph,hidden=hidden,l2=l2)
        sp=self.fd/'starts.json'
        if sp.exists():starts=read(sp)
        else:
            starts=[]
            for index,graph in enumerate(proposals):
                rows=[r for r in self.rows if r['event']=='scored' and r.get('start_id') is None and r['graph']['architecture_id']==graph['architecture_id']];unique={}
                for r in sorted(rows,key=lambda r:(-r['score'],r['key'])):unique.setdefault(r['key'],r)
                starts.extend([[f'C_{index}_{j}',r] for j,r in enumerate(list(unique.values())[:2])])
            write(sp,starts)
        self.refine(starts,30)
    def land(self,row,phase):
        self.count(phase);self.log({'event':'requested','phase':phase,'request':self.state['requests'],'source_key':row['key'],'elapsed':self.elapsed()})
        start=time.perf_counter()
        if self.method=='A':
            if self.decoder is None:self.decoder=OpenDecoder(self.root,self.g)
            selected,meta=self.decoder.decode(row['u']);self.state['times']['decoder']+=time.perf_counter()-start
            self.state['decoder_calls']=self.decoder.calls+self.state.get('prior_decoder_calls',0);self.state['decoder_hits']=self.decoder.hits+self.state.get('prior_decoder_hits',0)
            if selected is None:self.log({'event':'rejected','phase':phase,'reason':meta,'source_key':row['key']});return
            cfg=selected['configuration'];score=self.p.actual_A(cfg,selected['latent'])
        else:
            cfg=graphs.config(row['graph']['architecture'],row['hidden'],row['l2'],row['u'],True);index=self.p.graph_index(cfg)
            v=np.asarray(row['u'],dtype='<f4').astype(np.float64);score,_=self.p.evaluate(v,False,index,cfg['hidden_dim'],cfg['l2'])
            selected={'candidate_id':digest(cfg),'architecture_id':graphs.aid(cfg),'configuration':cfg,'latent':None,'latent_source':'none_direct_graph','proposal_v_float64':row['u']};meta=row['graph'].get('source',{})
        if selected['candidate_id'] in self.seen:self.log({'event':'rejected','phase':phase,'reason':'seen_complete_configuration','source_key':row['key']});return
        selected.update(realized_qLogEI=score,inner_qLogEI=row.get('score'),proposal_source=meta,source_key=row['key'],
           relaxed_minus_realized=None if row.get('score') is None else row['score']-score)
        self.log({'event':'landed','phase':phase,'selected':selected,'elapsed':self.elapsed()});self.save()
    def post(self):
        unique={}
        scored=[r for r in self.rows if r['event']=='scored']
        for row in sorted(scored,key=lambda r:(-r['score'],r['key'])):unique.setdefault(row['key'],row)
        for row in list(unique.values())[:64]:
            if row['key'] in self.state['post_keys']:continue
            if self.state['post_requests']>=64 or self.elapsed()>=300:break
            # Mark before native scoring: an interrupted request still consumes its slot.
            self.state['post_keys'].append(row['key']);self.save();self.land(row,'post')
        if any(r['event']=='landed' for r in self.rows):return
        if self.method=='A':
            design=torch.quasirandom.SobolEngine(16,scramble=True,seed=self.salt['A_fallback']).draw(128,dtype=torch.float64).numpy()
            for i in range(self.state['fallback_done'],128):
                if self.state['fallback_requests']>=128 or self.elapsed()>=300:break
                self.state['fallback_done']=i+1;self.save();self.land({'u':design[i].tolist(),'key':'fallback_A_'+str(i)},'fallback')
        else:
            rng=np.random.default_rng(self.salt['C_fallback']);accepted=0
            for attempt in range(128):
                m=int(rng.integers(1,6));ops=[graphs.OPS[int(i)] for i in rng.integers(0,6,m)];edges=[(u,v) for u in range(m+1) for v in range(u+1,m+1) if rng.random()<.5]
                try:arch=graphs.rebuild(ops,edges)
                except ValueError:continue
                hidden=graphs.HIDDEN[int(rng.integers(6))];l2=graphs.L2[int(rng.integers(5))];v=rng.random(2)
                accepted+=1
                if accepted<=self.state['fallback_done']:continue
                if self.state['fallback_requests']>=64 or self.elapsed()>=300:break
                self.state['fallback_done']=accepted;self.save()
                self.land({'graph':{'architecture':arch,'source':{'flow':'fallback','attempt':attempt}},'hidden':hidden,'l2':l2,'u':v.tolist(),'key':'fallback_C_'+str(attempt)},'fallback')
    def best(self):
        landed=[r['selected'] for r in self.rows if r['event']=='landed']
        if not landed:raise RuntimeError('NO_VALID_RETURN: bounded acquisition has no legitimate unseen configuration')
        return min(landed,key=lambda r:(-r['realized_qLogEI'],r['candidate_id'],r['source_key']))
    def execute(self,post_only=False):
        try:
            if not post_only and self.state['phase']=='main' and self.elapsed()<270:
                self.main_A() if self.method=='A' else self.main_C()
        except Exception as exc:self.state['events'].append({'event':'main_exception','type':type(exc).__name__,'error':str(exc)})
        self.state['phase']='post';self.save();self.post()
        if self.elapsed()>300:raise TimeoutError('Hard round deadline exceeded; external supervisor must return earlier legitimate archive')
        chosen=self.best();self.p.unchanged()
        self.state['phase']='complete';self.save()
        if self.p.sampler.base_samples is not None:gp_path=self.fd/'sampler.pt';torch.save({'mc_seed':self.mc,'base_samples':self.p.sampler.base_samples.detach().clone()},gp_path)
        write(self.fd/'selection.json',chosen);self.save();return chosen
    def diagnostics(self):
        meter=self.state['meters']
        return {'posterior_points':meter['posterior_points'],'gradient_requests':meter['gradients'],'forward_seconds':meter['forward_seconds'],
          'backward_seconds':meter['backward_seconds'],'WL_seconds':meter['WL_seconds'],'times':self.state['times'],'events':self.state['events'],
          'decoder_calls':self.state['decoder_calls'],'decoder_hits':self.state['decoder_hits'],'encoder_calls':0,'full_training_tokens':len(self.p.training['tokens']),
          'graph_streams':read(self.fd/'graphs.json') if (self.fd/'graphs.json').exists() else None,'fallback_used':self.state['fallback_requests']>0,
          'normal_requests':self.state['main_requests'],'starts':self.state['start_calls'],'total_cap':2048,'time_cap':300}
