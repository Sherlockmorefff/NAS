"""Stable recursive WL semantics and point-only deterministic CPU decoding."""
from collections import Counter
import hashlib
import numpy as np
import torch
from .common import B,read,canonical,digest
from analyse.structured_kernel_phase1.features import build_operator_graph

SALT=b'acq_optimizer_offline_v1_decoder_2026101207\0'
def norm_u(u):
    a=np.asarray(u,dtype=np.float64)
    if a.shape!=(16,) or not np.isfinite(a).all():raise ValueError('Malformed unit vector')
    return np.clip(a,0.,1.)
def point_key(u): return norm_u(u).astype('<f8').tobytes().hex()
def z_vector(u):
    u=norm_u(u);return np.r_[-2.5+5*u[:12],u[12:]].astype('<f4')
def semantic(label): return hashlib.sha256(canonical(label).encode()).hexdigest()

class WL:
    def __init__(self,vocabulary):
        if vocabulary['n_max']!=7 or vocabulary['e_max']!=16:raise ValueError('Frozen graph scales changed')
        levels=[];columns={};offset=0
        for h,d in enumerate(vocabulary['dictionaries']):
            labels=[]
            for raw in d['labels']:
                import json
                value=json.loads(raw)
                if h:
                    c,inc,out=value;p=levels[-1]
                    value=[p[c],sorted(p[i] for i in inc),sorted(p[i] for i in out)]
                labels.append(semantic(value))
            if len(set(labels))!=len(labels):raise ValueError('WL semantic migration collision')
            for i,label in enumerate(labels):columns[(h,label)]=offset+i
            levels.append(labels);offset+=len(labels)
        self.columns=columns;self.dimension=offset
    def counts(self,config):
        graph=build_operator_graph(config);nodes=graph['nodes'];edges=graph['edges']
        inc=[[] for _ in nodes];out=[[] for _ in nodes]
        for u,v in edges:inc[v].append(u);out[u].append(v)
        labels=[semantic(node) for node in nodes];counts=Counter()
        for h in range(3):
            counts.update((h,x) for x in labels)
            if h<2:labels=[semantic([labels[v],sorted(labels[i] for i in inc[v]),sorted(labels[i] for i in out[v])]) for v in range(len(nodes))]
        return counts,len(nodes),len(edges)
    def feature(self,config):
        counts,n,e=self.counts(config);norm=np.sqrt(sum(v*v for v in counts.values()))
        known=np.zeros(self.dimension+2,dtype=np.float64);unknown=0.;unknown_symbols=[]
        for key,v in counts.items():
            if key in self.columns:known[self.columns[key]]=v/norm
            else:unknown+=(v/norm)**2;unknown_symbols.append([key[0],key[1],v])
        known[-2:]=[n/7.,e/16.]
        return known,float(unknown),unknown_symbols

class Decoder:
    def __init__(self):
        self.pool=read(B/'groups/0/candidates.json');self.pool_ids={r['candidate_id'] for r in self.pool}
        self.old={};self.decode_calls=0;self.map_hits=0;self.cache_hits=0;self.new_cache={}
        for row in self.pool:
            cfg=row['configuration'];arch={k:cfg[k] for k in ['operations','edges','effective_layers']}
            key=np.asarray(row['z_search'][:12],dtype='<f4').tobytes()
            if key in self.old and digest(self.old[key])!=digest(arch):raise ValueError('Conflicting frozen architecture for same float32 latent')
            self.old[key]=arch
        from architecture_diagnostics import load_joint_vae
        old=torch.get_default_dtype();torch.set_default_dtype(torch.float32)
        try:self.vae=load_joint_vae(B.parent.parent/'checkpoints/joint_model_pipeline_global4_best.pth','global4',device='cpu').float().eval()
        finally:torch.set_default_dtype(old)
        for p in self.vae.parameters():p.requires_grad_(False)
        self.weight_hash=digest({k:str(tuple(v.shape)) for k,v in self.vae.state_dict().items()})
    def reset(self):self.new_cache={};self.decode_calls=self.map_hits=self.cache_hits=0
    def decode(self,u): return self.decode_z(z_vector(u))
    def decode_z(self,z):
        from eval_utils import _decode_arch,decode_hp_by_mode
        z=np.asarray(z,dtype='<f4').copy();z[:12]=np.clip(z[:12],-2.5,2.5);z[12:]=np.clip(z[12:],0.,1.)
        key=z[:12].tobytes();seed=int.from_bytes(hashlib.sha256(SALT+key).digest()[:4],'little')
        if key in self.old:arch=self.old[key];self.map_hits+=1;source='frozen768'
        elif key in self.new_cache:arch=self.new_cache[key];self.cache_hits+=1;source='run-cache'
        else:
            old=torch.get_default_dtype();torch.set_default_dtype(torch.float32)
            try:arch=_decode_arch(self.vae,torch.from_numpy(z[:12].copy()),torch.device('cpu'),n_trials=5,decoder_seed=seed)
            finally:torch.set_default_dtype(old)
            self.decode_calls+=1;self.new_cache[key]=arch;source='new-decode'
        if arch is None:return None,{'decoder_seed':seed,'decoder_source':source,'z_float32':z.tolist(),'reason':'no_valid_architecture'}
        hp=decode_hp_by_mode(torch.from_numpy(z),12,arch,'global4',-4.,-1.5,.1,.6)
        cfg={'operations':arch['operations'],'edges':sorted([list(map(int,e)) for e in arch['edges']]),
             'normalization':'none','hp_mode':'global4','effective_layers':arch['effective_layers'],
             'lr':hp['lr'],'dropout':hp['dropout'],'hidden_dim':hp['hidden_dim'],'l2':hp['weight_decay'],
             'gat_heads':hp['gat_heads'],'sage_aggr':hp['sage_aggr'],'gin_eps':hp['gin_eps'],
             'gat_heads_by_layer':None,'sage_aggr_by_layer':None,'gin_eps_by_layer':None,'gcnii_alpha':.1,'gcnii_theta':.5}
        try:build_operator_graph(cfg)
        except ValueError as exc:return None,{'decoder_seed':seed,'reason':'invalid_recorded_graph','error':str(exc),'z_float32':z.tolist(),'decoder_source':source}
        return cfg,{'decoder_seed':seed,'decoder_source':source,'z_float32':z.tolist(),'candidate_id':digest(cfg)}
