"""LayerNorm wrapper of frozen validation-only training, with segment timing.

The numerical epoch body follows eval_utils.train_and_eval_arch exactly. Only
train/validation labels are retained by the training view. Synthetic acceptance
compares every loss/validation epoch and final model against the original core.
"""
import copy
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from deterministic_runtime import prepare_deterministic_environment
prepare_deterministic_environment()
import torch
import torch.nn.functional as F
import eval_utils as original
from .common import read_json,write_json,append_jsonl,sha256,digest,now


def save_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def training_view(data):
    keep=data.train_mask|data.val_mask
    labels=torch.full_like(data.y,-100);labels[keep]=data.y[keep]
    return SimpleNamespace(x=data.x,edge_index=data.edge_index,y=labels,
        train_mask=data.train_mask,val_mask=data.val_mask)


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def epoch_core(gnn,data,optimizer,scheduler):
    begin=time.perf_counter()
    cuda=data.x.device.type=='cuda'
    events=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if cuda else None
    marks=[time.perf_counter()]
    gnn.train();optimizer.zero_grad()
    if cuda:events[0].record()
    out=gnn(data.x,data.edge_index)
    if not bool(torch.isfinite(out).all().detach().cpu().item()):raise FloatingPointError('non-finite DynamicGNN training output')
    loss=F.cross_entropy(out[data.train_mask],data.y[data.train_mask])
    if not bool(torch.isfinite(loss).detach().cpu().item()):raise FloatingPointError('non-finite DynamicGNN training loss')
    if cuda:events[1].record()
    marks.append(time.perf_counter())
    loss.backward();optimizer.step();scheduler.step()
    if cuda:events[2].record()
    marks.append(time.perf_counter())
    gnn.eval()
    with torch.no_grad():
        eval_out=gnn(data.x,data.edge_index)
        if not bool(torch.isfinite(eval_out).all().detach().cpu().item()):raise FloatingPointError('non-finite DynamicGNN validation output')
        pred=eval_out.argmax(dim=1);count=int(data.val_mask.sum().item())
        if count<=0:raise ValueError('validation mask is empty')
        accuracy=pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()/count
    if cuda:
        events[3].record();events[3].synchronize()
    marks.append(time.perf_counter())
    values=[events[i].elapsed_time(events[i+1])/1000 for i in range(3)] if cuda else [marks[i+1]-marks[i] for i in range(3)]
    timings=dict(zip(['forward_loss_seconds','backward_Adam_scheduler_seconds','validation_seconds'],values))
    timings['epoch_wall_seconds']=time.perf_counter()-begin
    timings['timing_clock']='CUDA events, epoch-end metric synchronization' if cuda else 'CPU perf_counter'
    return float(loss.detach().cpu().item()),float(accuracy),timings


def train_validation_only(config,data,in_ch,out_ch,seed,device,folder,identity,*,max_epochs=300,patience=80):
    torch.set_default_dtype(torch.float32)
    original.assert_strict_torch_determinism(torch)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'resume.pt';result_path=folder/'result.json'
    if result_path.exists():
        result=read_json(result_path)
        if result['identity']!=identity:raise ValueError('Completed training identity mismatch')
        for name,h in result.get('checkpoint_hashes',{}).items():
            if sha256(folder/name)!=h:raise ValueError('Saved checkpoint changed')
        return result
    old=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint.exists() else None
    if old and old['identity']!=identity:raise ValueError('Resume identity mismatch')
    if old and old.get('result'):
        result=old['result'];result['checkpoint_hashes']={'resume.pt':sha256(checkpoint),'best_model.pt':sha256(folder/'best_model.pt')}
        write_json(result_path,result);return result
    started=time.perf_counter();previous=old.get('elapsed_seconds',0.) if old else 0.
    checkpoint_io=[old.get('checkpoint_io_seconds',0.) if old else 0.]
    def timed_save(path,value):
        before=time.perf_counter();save_torch(path,value);checkpoint_io[0]+=time.perf_counter()-before
    data=training_view(data)
    assert data.x.dtype==torch.float32 and not hasattr(data,'test_mask')
    with original.isolated_rng(seed,device):
        model=original.DynamicGNN(config,in_ch,out_ch,dropout=float(config['dropout']),hidden_dim=int(config['hidden_dim']),
            gcnii_alpha=float(config.get('gcnii_alpha',.1)),gcnii_theta=float(config.get('gcnii_theta',.5)),
            gat_heads=int(config['gat_heads']),sage_aggr=config['sage_aggr'],gin_eps=float(config['gin_eps']),
            gat_heads_by_layer=config.get('gat_heads_by_layer'),sage_aggr_by_layer=config.get('sage_aggr_by_layer'),
            gin_eps_by_layer=config.get('gin_eps_by_layer'),component_variant='layer_norm',activation_variant='relu').to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=float(config['lr']),weight_decay=float(config['l2']))
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs,eta_min=float(config['lr'])*.01)
        best_val=0.;best_epoch=None;no_improve=0;trajectory=[];completed=0;best_state=None
        if old:
            model.load_state_dict(old['model'],strict=True);optimizer.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler'])
            best_val=old['best_val'];best_epoch=old['best_epoch'];no_improve=old['no_improve'];trajectory=old['trajectory'];completed=old['epoch'];best_state=old['best_state']
            restore_rng(old['rng'])
        append_jsonl(folder/'attempts.jsonl',{'started_at':now(),'seed':seed,'identity':identity,'resume_epoch':completed,
            'same_task_retry':bool(old),'checkpoint_interval':5,'interrupted_updates_after_saved_epoch_may_be_recomputed':bool(old)})
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        def snapshot(result=None):
            return {'identity':identity,'epoch':completed,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'best_val':best_val,'best_epoch':best_epoch,'no_improve':no_improve,
                'best_state':best_state,'rng':capture_rng(),'trajectory':trajectory,
                'elapsed_seconds':previous+time.perf_counter()-started,'checkpoint_io_seconds':checkpoint_io[0],'result':result}
        if not old:timed_save(checkpoint,snapshot())
        for epoch in range(completed,max_epochs):
            if no_improve>=patience:break
            loss,val,timing=epoch_core(model,data,optimizer,scheduler);completed=epoch+1
            trajectory.append({'epoch_1based':completed,'loss':loss,'validation_accuracy':val,**timing})
            improved=val>best_val
            if improved:
                best_val=val;best_epoch=completed;no_improve=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:no_improve+=1
            append_jsonl(folder/'epoch_events.jsonl',{'epoch_1based':completed,'loss':loss,'validation_accuracy':val,
                'best_epoch':best_epoch,'best_validation':best_val,'lr_next':optimizer.param_groups[0]['lr'],'time':now(),**timing})
            if improved or completed%5==0 or no_improve>=patience or completed==max_epochs:timed_save(checkpoint,snapshot())
        if best_state is None:raise RuntimeError('Original best_val never exceeded zero; no selected checkpoint')
        final_hash=original._model_state_hash(model)
        # Restoring the exact selected validation state is a verification, not
        # a new training/update or changed selection criterion.
        model.load_state_dict(best_state,strict=True);model.eval()
        with torch.no_grad():
            predictions=model(data.x,data.edge_index).argmax(dim=1)
            restored=float(predictions[data.val_mask].eq(data.y[data.val_mask]).sum().item()/int(data.val_mask.sum().item()))
        if restored!=best_val:raise ValueError('Restored best-state validation differs from selected epoch')
        timed_save(folder/'best_model.pt',{'identity':identity,'model':best_state,'best_epoch':best_epoch,'validation_accuracy':best_val})
        result={'identity':identity,'state':'COMPLETE','valid':True,'seed':seed,'best_validation_accuracy':best_val,
            'best_epoch':best_epoch,'stopped_epoch':completed,'stop_reason':'patience' if no_improve>=patience else 'max_epochs',
            'best_restored_validation_accuracy':restored,'model_final_before_restore_sha256':final_hash,
            'trajectory':trajectory,'parameter_count':sum(p.numel() for p in model.parameters()),'dtype':'float32',
            'elapsed_seconds':previous+time.perf_counter()-started,'device':str(device),
            'peak_gpu_memory_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'test_metric_computations':0,'test_score_reads':0,'completed_at':now(),'component_variant':'layer_norm',
            'warm_start':False,'checkpoint_io_seconds':checkpoint_io[0],
            'stage_timing':{k:sum(x.get(k,0.) for x in trajectory) for k in ['forward_loss_seconds','backward_Adam_scheduler_seconds','validation_seconds','epoch_wall_seconds']}}
        timed_save(checkpoint,snapshot(result))
        result['checkpoint_hashes']={'resume.pt':sha256(checkpoint),'best_model.pt':sha256(folder/'best_model.pt')}
        result['checkpoint_io_seconds']=checkpoint_io[0];write_json(result_path,result)
        return result

