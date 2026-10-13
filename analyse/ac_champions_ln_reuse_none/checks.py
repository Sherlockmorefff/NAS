import hashlib
import random
import time
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import eval_utils as original
from .common import digest

def construct(config,in_ch,out_ch,seed,device,variant='layer_norm'):
    torch.set_default_dtype(torch.float32)
    with original.isolated_rng(seed,device):
        model=original.DynamicGNN(config,in_ch,out_ch,dropout=config['dropout'],hidden_dim=config['hidden_dim'],
            gcnii_alpha=config['gcnii_alpha'],gcnii_theta=config['gcnii_theta'],gat_heads=config['gat_heads'],
            sage_aggr=config['sage_aggr'],gin_eps=config['gin_eps'],gat_heads_by_layer=config.get('gat_heads_by_layer'),
            sage_aggr_by_layer=config.get('sage_aggr_by_layer'),gin_eps_by_layer=config.get('gin_eps_by_layer'),
            component_variant=variant,activation_variant='relu').to(device)
    return model

def rng_hash(device):
    n=np.random.get_state()
    value={'python':digest(random.getstate()),'numpy':digest([n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])]),
           'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
           'cuda':[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()] if device.type=='cuda' else []}
    return value

def state_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        x=t.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()

def initialization(config,seed,in_ch,out_ch,device):
    states=[];rng=[];counts=[];extras=None
    for variant in ('baseline','layer_norm'):
        with original.isolated_rng(seed,device):
            # construct() isolates and restores RNG, so use the raw constructor
            # here to capture the actual state immediately before training.
            m=original.DynamicGNN(config,in_ch,out_ch,dropout=config['dropout'],hidden_dim=config['hidden_dim'],
                gcnii_alpha=config['gcnii_alpha'],gcnii_theta=config['gcnii_theta'],gat_heads=config['gat_heads'],
                sage_aggr=config['sage_aggr'],gin_eps=config['gin_eps'],gat_heads_by_layer=config.get('gat_heads_by_layer'),
                sage_aggr_by_layer=config.get('sage_aggr_by_layer'),gin_eps_by_layer=config.get('gin_eps_by_layer'),
                component_variant=variant,activation_variant='relu').to(device)
            sd=m.state_dict();shared={k:v for k,v in sd.items() if not k.startswith('layer_norms.')}
            states.append(state_hash(shared));rng.append(rng_hash(device));counts.append(sum(p.numel() for p in m.parameters()))
            if variant=='layer_norm':
                extras={k:v for k,v in sd.items() if k.startswith('layer_norms.')}
                if len(m.layer_norms)!=sum(x!='Identity' for x in config['operations']):
                    raise ValueError('LN on wrong layers')
                for ln in m.layer_norms.values():
                    if ln.eps!=1e-5 or not ln.elementwise_affine or tuple(ln.normalized_shape)!=(config['hidden_dim'],):
                        raise ValueError('Wrong LN parameters')
                if not all(torch.equal(v,torch.ones_like(v) if k.endswith('weight') else torch.zeros_like(v)) for k,v in extras.items()):
                    raise ValueError('Wrong LN constant initialization')
            del m
    if states[0]!=states[1] or rng[0]!=rng[1]:
        raise ValueError('LN changed shared initialization or pre-training RNG')
    return {'seed':seed,'device':str(device),'shared_initial_parameters_sha256':states[0],'pretraining_rng':rng[0],
            'shared_parameters_equal':True,'pretraining_rng_equal':True,'none_parameter_count':counts[0],'LN_parameter_count':counts[1],
            'added_LN_parameter_count':counts[1]-counts[0],'initialization_evidence':'reconstructed_from_frozen_none_code',
            'historical_initialization_hash_available':False,'LN_gamma_beta_verified':True,'warm_start':False}

def sanity(champions):
    started=time.perf_counter();torch.set_num_threads(1)
    from .training import epoch_core,training_view
    from analyse.structured_kernel_followup.validation_train import epoch_core as reference_epoch, restore_rng
    initialization_rows=[{'method':c['method'],'group':c['group'],**initialization(c['configuration'],c['final_seeds'][0],1433,7,torch.device('cpu'))} for c in champions]
    torch.manual_seed(123)
    data=SimpleNamespace(x=torch.randn(12,8),edge_index=torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11],[1,2,3,4,5,6,7,8,9,10,11,0]]),
                         y=torch.arange(12)%3,train_mask=torch.arange(12)<4,val_mask=(torch.arange(12)>=4)&(torch.arange(12)<8),test_mask=torch.arange(12)>=8)
    view=training_view(data)
    if hasattr(view,'test_mask') or not torch.all(view.y[~(data.train_mask|data.val_mask)]==-100):
        raise ValueError('Training label redaction failed')
    records=[]
    for ops in [['GCNConv','GATConv','SAGEConv','GINConv','Identity'],['GCNII','Identity','GCNII']]:
        cfg={'operations':ops,'edges':[[i,i+1] for i in range(len(ops)+1)],'hidden_dim':8,'dropout':.2,'lr':.001,'l2':.0005,
             'gat_heads':1,'sage_aggr':'mean','gin_eps':0.,'gcnii_alpha':.1,'gcnii_theta':.5,'effective_layers':sum(x!='Identity' for x in ops)}
        a=construct(cfg,8,3,321,torch.device('cpu'));b=construct(cfg,8,3,321,torch.device('cpu'))
        b.load_state_dict(a.state_dict(),strict=True)
        try:
            construct(cfg,8,3,321,torch.device('cpu'),'baseline').load_state_dict(a.state_dict(),strict=True)
        except RuntimeError:
            strict_rejected=True
        else:
            raise ValueError('Strict checkpoint load accepted wrong variant')
        captured={};handles=[]
        for key,ln in a.layer_norms.items():
            i=int(key)
            handles.append(a.layers[i].register_forward_hook(lambda mod,inp,out,i=i:captured.__setitem__('op'+str(i),out.detach().clone())))
            handles.append(ln.register_forward_pre_hook(lambda mod,inp,i=i:captured.__setitem__('ln'+str(i),inp[0].detach().clone())))
            handles.append(ln.register_forward_hook(lambda mod,inp,out,i=i:captured.__setitem__('out'+str(i),out.detach().clone())))
        dropout_original=original.F.dropout;drop_calls=[]
        def observed_dropout(x,*args,**kwargs):
            # GAT also calls functional dropout internally for attention.
            # Check only the declared post-convolution DynamicGNN branch.
            if sys._getframe(1).f_code is original.DynamicGNN.forward.__code__:
                drop_calls.append(x.detach().clone())
            return dropout_original(x,*args,**kwargs)
        original.F.dropout=observed_dropout
        try:
            a.eval()
            with torch.no_grad():a(view.x,view.edge_index)
        finally:
            original.F.dropout=dropout_original
            for h in handles:h.remove()
        nonids=[i for i,x in enumerate(ops) if x!='Identity']
        if len(drop_calls)!=len(nonids):raise ValueError('Unexpected dropout/LN count')
        for j,i in enumerate(nonids):
            if not torch.equal(captured['op'+str(i)],captured['ln'+str(i)]) or not torch.equal(drop_calls[j],torch.relu(captured['out'+str(i)])):
                raise ValueError('LN is not between convolution and ReLU/dropout')
        oa=torch.optim.Adam(a.parameters(),lr=.001,weight_decay=.0005);ob=torch.optim.Adam(b.parameters(),lr=.001,weight_decay=.0005)
        sa=torch.optim.lr_scheduler.CosineAnnealingLR(oa,300,eta_min=.00001);sb=torch.optim.lr_scheduler.CosineAnnealingLR(ob,300,eta_min=.00001)
        for _ in range(2):
            rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':[]}
            la,va=reference_epoch(a,view,oa,sa);restore_rng(rng);lb,vb,_=epoch_core(b,view,ob,sb)
            if la!=lb or va!=vb or state_hash(a.state_dict())!=state_hash(b.state_dict()):
                raise ValueError('Timing wrapper changed original numerical epoch')
        records.append({'ops':ops,'LN_placement':'PASS','strict_wrong_variant_rejected':strict_rejected,'two_synthetic_LN_epochs_match_original_kernel_exactly':True})
    from .training import train_validation_only
    with tempfile.TemporaryDirectory(prefix='ac_ln_synthetic_') as temp:
        f=Path(temp)
        v=train_validation_only(cfg,data,8,3,321,torch.device('cpu'),f,'synthetic_ln_only',max_epochs=3,patience=2)
        repeated=train_validation_only(cfg,data,8,3,321,torch.device('cpu'),f,'synthetic_ln_only',max_epochs=3,patience=2)
        if v!=repeated or v['test_metric_computations']!=0 or v['component_variant']!='layer_norm':
            raise ValueError('Synthetic completed same-task checkpoint reuse failed')
        try:
            train_validation_only(cfg,data,8,3,321,torch.device('cpu'),f,'wrong_identity',max_epochs=3,patience=2)
        except ValueError:
            identity_rejected=True
        else:
            raise ValueError('Completed LN checkpoint reused under wrong identity')
        records.append({'full_synthetic_LN_training_and_completed_same_task_reuse':'PASS','wrong_identity_rejected':identity_rejected,'synthetic_epochs':v['stopped_epoch']})
    return {'checks':'PASS','CPU_seconds':time.perf_counter()-started,'initialization_first_seed_each_configuration':initialization_rows,
            'synthetic':records,'Cora_training_calls':0,'None_training_calls':0,'test_scoring_calls':0,
            'all_300_device_initializations_checked_by_formal_workers_before_each_LN_training':True}
