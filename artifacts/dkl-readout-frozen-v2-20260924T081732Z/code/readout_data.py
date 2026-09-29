"""Frozen selected-context boundary. Never opens a scoring label source.

Source records are projected into small exact-schema objects; query observations
are rejected, not silently retained. The caller charges all numerical work.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random

SOURCE_TRAIN_KEYS={'hp','identity','ids','known_variance','known_variance_standardized','measurement_se',
    'metadata','scaler','validation_seeds','x_raw','x_standardized','y','y_standardized','z'}
SCALER_KEYS={'dtype','fit_ids','frozen','std_correction','x_mean','x_scale','y_mean','y_scale'}
QUERY_KEYS={'ids','architectures','x_raw'}
TRAIN_KEYS={'ids','h0','s0','y_standardized','known_variance_standardized','scaler'}
CACHED_QUERY_KEYS={'ids','architectures','h0','s0'}


def _keys(record,allowed,name):
    if not isinstance(record,dict) or set(record)!=set(allowed):
        extra=set(record)-set(allowed) if isinstance(record,dict) else set()
        missing=set(allowed)-set(record) if isinstance(record,dict) else set(allowed)
        raise ValueError(f'{name} violates exact whitelist: extra={sorted(extra)}, missing={sorted(missing)}')


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def query_projection(query):
    """Strict public-only query boundary; poisoning y/SE produces an error."""
    import numpy as np
    _keys(query,QUERY_KEYS,'Query')
    ids,arch=list(query['ids']),list(query['architectures'])
    x=np.asarray(query['x_raw'],dtype=np.float64)
    if len(ids)<5 or len(set(ids))!=len(ids) or len(arch)!=len(ids) or x.shape!=(len(ids),16):
        raise ValueError('Query identity/feature shape')
    if not all(isinstance(v,str) and v for v in ids+arch) or not np.isfinite(x).all():
        raise ValueError('Query contains invalid public values')
    return {'ids':ids,'architectures':arch,'x_raw':x.tolist()}


def training_projection(training,expected_ids=None):
    import numpy as np
    _keys(training,SOURCE_TRAIN_KEYS,'Selected source training')
    _keys(training['scaler'],SCALER_KEYS,'Selected scaler')
    ids=list(training['ids']);sc=copy.deepcopy(training['scaler'])
    if training['metadata'].get('query_labels_released') != 0 or training['metadata'].get('query_SE_released') != 0:
        raise ValueError('Source training snapshot is not explicitly isolated from query observations')
    if len(ids) not in (20,50) or len(set(ids))!=len(ids) or (expected_ids is not None and ids!=list(expected_ids)):
        raise ValueError('Selected IDs or count changed')
    if sc['fit_ids']!=ids or sc['frozen'] is not True or sc['std_correction']!=0 or sc['dtype']!='float64':
        raise ValueError('Scaler was not fitted/frozen on the same selected subset')
    x=np.asarray(training['x_standardized'],dtype=np.float64)
    y=np.asarray(training['y_standardized'],dtype=np.float64)
    known=np.asarray(training['known_variance_standardized'],dtype=np.float64)
    xr=np.asarray(training['x_raw'],dtype=np.float64);yr=np.asarray(training['y'],dtype=np.float64)
    se=np.asarray(training['measurement_se'],dtype=np.float64)
    xm,xs=np.asarray(sc['x_mean'],dtype=np.float64),np.asarray(sc['x_scale'],dtype=np.float64)
    yc,ys=float(sc['y_mean']),float(sc['y_scale'])
    if x.shape!=(len(ids),16) or xr.shape!=x.shape or any(v.shape!=(len(ids),) for v in (y,known,yr,se)) or xm.shape!=(16,) or xs.shape!=(16,):
        raise ValueError('Selected tensor/scaler shape')
    if not all(np.isfinite(v).all() for v in (x,y,known,xr,yr,se,xm,xs)) or not all(math.isfinite(v) for v in (yc,ys)):
        raise ValueError('Nonfinite selected training inputs')
    if np.any(se<0) or np.any(known<0) or np.any(xs<1e-6) or ys<1e-8:
        raise ValueError('Selected noise/scaler lower bound')
    if not (np.allclose(x,(xr-xm)/xs,atol=1e-12,rtol=1e-12)
            and np.allclose(y,(yr-yc)/ys,atol=1e-12,rtol=1e-12)
            and np.allclose(known,se**2/ys**2,atol=1e-14,rtol=1e-12)):
        raise ValueError('Selected standardization or mean-SE variance differs')
    return {'ids':ids,'x_standardized':x.tolist(),'y_standardized':y.tolist(),
            'known_variance_standardized':known.tolist(),'scaler':sc}


def prepare_context(common_state,training,query,context,saved_diagnostic,
                    source_sha256=None,diagnostic_sha256=None):
    """Restore exactly COMMON_gp300, freeze mapper/head and cache h0/s0 once.

    Returns independent CPU float64 tensors. Only selected training y/noise are
    present. The live historical network is deliberately not returned, so no new
    GP optimizer can acquire network/head parameters by accident.
    """
    import numpy as np
    import torch
    from stage01_core import tensor_hash
    from residual_core import restore
    train=training_projection(training)
    public=query_projection(query)
    cid,sid=context['context_id'],context['subset_id']
    if len(train['ids'])!=context['n'] or set(train['ids'])&set(public['ids']):
        raise ValueError('Selected context scope/order mismatch')
    if 'selected_ids_sha256' in context and digest(train['ids'])!=context['selected_ids_sha256']:
        raise ValueError('Selected candidate order differs from the frozen worklist')
    if training['metadata']['subset_id']!=sid or training['metadata']['fold']!=context['fold']:
        raise ValueError('Source snapshot context identity')
    schedule=common_state.get('scheduler',{})
    if (schedule.get('block'),schedule.get('mode'),schedule.get('planned_steps'),schedule.get('completed'))!=(cid+'_COMMON_gp','gp',300,300):
        raise ValueError('Only the exact saved COMMON_gp300 checkpoint is permitted')
    if common_state.get('model_type')!='RESIDUAL16' or common_state.get('constraint_mode')!='bounded':
        raise ValueError('Source is not the fixed residual common state')
    rng=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    try:
        model=restore(common_state)
        for key,attr in [('x_standardized','x'),('y_standardized','y'),('known_variance_standardized','known')]:
            if not torch.equal(getattr(model,attr),torch.tensor(train[key],dtype=torch.float64)):
                raise ValueError('Common saved selected tensors differ: '+key)
        original_hash=tensor_hash(model.mapper.state_dict())
        model.mapper.eval()
        for p in model.mapper.parameters():p.requires_grad_(False);p.grad=None
        sc=train['scaler'];qx=torch.tensor(public['x_raw'],dtype=torch.float64)
        qx=(qx-torch.tensor(sc['x_mean'],dtype=torch.float64))/torch.tensor(sc['x_scale'],dtype=torch.float64)
        with torch.no_grad():
            ht,st=model.encode(model.x);hq,sq=model.encode(qx)
        ht,st,hq,sq=(v.detach().clone() for v in (ht,st,hq,sq))
        if ht.shape!=(context['n'],16) or hq.shape!=(len(public['ids']),16) or st.shape!=(context['n'],) or sq.shape!=(len(public['ids']),):
            raise ValueError('Frozen h0/s0 shape')
        if not all(bool(torch.isfinite(t).all()) and t.dtype==torch.float64 and not t.requires_grad for t in (ht,st,hq,sq)):
            raise ValueError('Invalid/nonfrozen h0/s0')
        if not torch.equal(ht,common_state['history_embedding']):
            raise ValueError('Cached h0 differs from saved COMMON_gp history')
        if tensor_hash(model.mapper.state_dict())!=original_hash or model.mapper.training or any(p.requires_grad for p in model.mapper.parameters()):
            raise RuntimeError('Mapping/head freeze or identity failed')
        # This source record contains unlabelled predictions only; project an
        # explicit reference whitelist, without reading a scoring directory.
        if (saved_diagnostic['block_key'],saved_diagnostic['block_step'],saved_diagnostic['mode'])!=(cid+'_COMMON_gp',300,'gp'):
            raise ValueError('Historical diagnostic is not COMMON_gp300')
        if saved_diagnostic['state_hash']!=model.fingerprint() or saved_diagnostic['query_ids']!=public['ids'] or saved_diagnostic['training_ids']!=train['ids']:
            raise ValueError('Historical diagnostic state/ID order differs')
        if saved_diagnostic.get('query_labels_or_SE_present') is not False:
            raise ValueError('Historical reference is not explicitly unlabelled')
        old_mean=np.asarray(saved_diagnostic['query_mean_standardized'],dtype=np.float64)
        old_var=np.asarray(saved_diagnostic['query_latent_variance_standardized'],dtype=np.float64)
        if old_mean.shape!=(len(public['ids']),) or old_var.shape!=old_mean.shape or not np.isfinite(old_mean).all() or not np.isfinite(old_var).all() or np.any(old_var < -1e-8):
            raise ValueError('Historical reference posterior invalid')
        if saved_diagnostic['y_center']!=sc['y_mean'] or saved_diagnostic['y_scale']!=sc['y_scale']:
            raise ValueError('Historical reference scaler differs')
        tensors={'train_h0':ht,'train_s0':st,'query_h0':hq,'query_s0':sq}
        identity={'context_id':cid,'subset_id':sid,'source_state_sha256':source_sha256,
            'source_diagnostic_sha256':diagnostic_sha256,'source_fingerprint':model.fingerprint(),
            'mapper_and_head_sha256':original_hash,
            'mapping_sha256':tensor_hash({k:v for k,v in model.mapper.state_dict().items() if not k.startswith('head.')}),
            'head_sha256':tensor_hash(model.mapper.head.state_dict()),
            'cache_sha256':tensor_hash(tensors),'cache_component_sha256':{k:tensor_hash({k:v}) for k,v in tensors.items()},
            'selected_tensor_sha256':tensor_hash({'h0':ht,'s0':st,'y':model.y,'known':model.known}),
            'training_ids_sha256':digest(train['ids']),'query_ids_sha256':digest(public['ids']),
            'query_architectures_sha256':digest(public['architectures']),
            'scaler_sha256':digest(sc),'common_gp_state_sha256':tensor_hash(model.gp.state_dict()),
            'frozen_parameter_names':[name for name,_ in model.mapper.named_parameters()],
            'mapper_head_eval':True,'mapper_head_requires_grad':False,'s0_transform':'none; original standardized-y head output',
            'query_labels_or_SE_present':False}
        result={'training':{'ids':train['ids'],'h0':ht,'s0':st,'y_standardized':model.y.detach().clone(),
                'known_variance_standardized':model.known.detach().clone(),'scaler':sc},
            'query':{'ids':public['ids'],'architectures':public['architectures'],'h0':hq,'s0':sq},
            'common_gp_state':copy.deepcopy(model.gp.state_dict()),'identity':identity,
            'old_reference':{'query_ids':public['ids'],'query_mean_standardized':old_mean.tolist(),
                'query_latent_variance_standardized':old_var.tolist(),'state_hash':model.fingerprint(),
                'block_step':300,'reference':'original unlabelled COMMON_gp300 diagnostic'}}
        verify_cached_context(result)
        return result
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])


def verify_cached_context(cache):
    import torch
    from stage01_core import tensor_hash
    _keys(cache,{'training','query','common_gp_state','identity','old_reference'},'Frozen context cache')
    train,query=cache['training'],cache['query']
    _keys(train,TRAIN_KEYS,'Readout training');_keys(query,CACHED_QUERY_KEYS,'Readout query')
    _keys(train['scaler'],SCALER_KEYS,'Readout scaler')
    identity=cache['identity']
    nt,nq=len(train['ids']),len(query['ids'])
    if nt not in (20,50) or nq<5 or len(set(train['ids']))!=nt or len(set(query['ids']))!=nq or set(train['ids'])&set(query['ids']):
        raise ValueError('Cached context identity')
    if len(query['architectures'])!=nq or train['scaler']['fit_ids']!=train['ids']:
        raise ValueError('Cached metadata shape/scope')
    tensors={'train_h0':train['h0'],'train_s0':train['s0'],'query_h0':query['h0'],'query_s0':query['s0']}
    for key,shape in [('h0',(nt,16)),('s0',(nt,)),('y_standardized',(nt,)),('known_variance_standardized',(nt,))]:
        value=train[key]
        if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=torch.float64 or value.device.type!='cpu' or value.requires_grad or not bool(torch.isfinite(value).all()):
            raise ValueError('Cached selected tensor invalid: '+key)
    for key,shape in [('h0',(nq,16)),('s0',(nq,))]:
        value=query[key]
        if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=torch.float64 or value.device.type!='cpu' or value.requires_grad or not bool(torch.isfinite(value).all()):
            raise ValueError('Cached public query tensor invalid: '+key)
    if (tensor_hash(tensors)!=identity['cache_sha256'] or tensor_hash(cache['common_gp_state'])!=identity['common_gp_state_sha256']
            or tensor_hash({'h0':train['h0'],'s0':train['s0'],'y':train['y_standardized'],
                'known':train['known_variance_standardized']})!=identity['selected_tensor_sha256']
            or digest(train['ids'])!=identity['training_ids_sha256'] or digest(query['ids'])!=identity['query_ids_sha256']
            or digest(query['architectures'])!=identity['query_architectures_sha256']
            or digest(train['scaler'])!=identity['scaler_sha256']):
        raise ValueError('Frozen cache identity changed')
    if bool((train['known_variance_standardized']<0).any()):raise ValueError('Negative selected variance')
    return True


def verify_gp_const_zero(record,reference):
    """Acceptance uses standardized outputs, atol=rtol=1e-7; no observations."""
    import numpy as np
    if record['step']!=0 or record['method']!='GP_CONST' or record['query_ids']!=reference['query_ids']:
        raise ValueError('Zero-step GP_CONST reference identity')
    differences={}
    for key in ('query_mean_standardized','query_latent_variance_standardized'):
        a,b=np.asarray(record[key],dtype=np.float64),np.asarray(reference[key],dtype=np.float64)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Zero-step reference shape/finite')
        differences[key]={'max_abs':float(np.max(np.abs(a-b))),'pass':bool(np.allclose(a,b,atol=1e-7,rtol=1e-7))}
    result={'pass':all(v['pass'] for v in differences.values()),'atol':1e-7,'rtol':1e-7,
            'unit':'standardized label mean and latent variance','differences':differences,'query_labels_or_SE_present':False}
    if not result['pass']:raise ValueError('GP_CONST0 does not reproduce original COMMON_gp300: '+json.dumps(result))
    return result
