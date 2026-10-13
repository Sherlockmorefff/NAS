import os
import time
import torch
from .common import *
from .checks import construct
from .prepare import bundle_data

def score(root):
    verify(root)
    from deterministic_runtime import configure_torch_determinism
    configure_torch_determinism(torch);torch.set_default_dtype(torch.float32);torch.set_num_threads(1)
    barrier=read_json(root/'LN_TRAINING_SEAL.json');decision=read_json(root/'VALIDATION_DECISION.json')
    if len(barrier['models'])!=300 or barrier['validation_decision_sha256']!=sha256(root/'VALIDATION_DECISION.json') or decision['LN_test_available_at_decision']:
        raise ValueError('LN validation/300-model barrier incomplete')
    bundle,context=bundle_data()
    if context!=read_json(root/'PROTOCOL.json')['dataset_context']:raise ValueError('Scoring data context changed')
    device=torch.device('cuda:0');data=bundle.data.to(device);output=[]
    write_json(root/'scoring_process.json',{'pid':os.getpid(),'at':now(),'variant':'layer_norm','strict':True,'dtype':'float32','None_scoring_calls':0})
    for task in tasks(root):
        t=next(x for x in barrier['models'] if x['task_id']==task['task_id'])
        checkpoint=Path(t['checkpoint']);out=root/'test_scores'/(t['task_id']+'.json')
        if sha256(checkpoint)!=t['checkpoint_sha256']:raise ValueError('LN sealed checkpoint changed')
        if out.exists():s=read_json(out)
        else:
            begin=time.perf_counter();model=construct(task['configuration'],bundle.num_features,bundle.num_classes,task['seed'],device)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['identity']!=task['task_id'] or saved['validation_accuracy']!=t['validation_accuracy']:
                raise ValueError('LN scoring checkpoint task/validation mismatch')
            model.load_state_dict(saved['model'],strict=True);model.eval()
            with torch.no_grad():
                pred=model(data.x,data.edge_index).argmax(1)
                value=float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()/int(data.test_mask.sum().item()))
            s={'task_id':task['task_id'],'checkpoint_sha256':t['checkpoint_sha256'],'barrier_sha256':sha256(root/'LN_TRAINING_SEAL.json'),
               'validation_decision_sha256':sha256(root/'VALIDATION_DECISION.json'),'test_accuracy':value,'computed_at':now(),
               'elapsed_seconds':time.perf_counter()-begin,'variant':'layer_norm','strict_checkpoint_load':True,'dtype':'float32',
               'training_updates':0,'None_score_computed':False,'scoring_pid':os.getpid()}
            seal(out,s);del model;torch.cuda.empty_cache()
        if s['computed_at']<barrier['sealed_at'] or s['barrier_sha256']!=sha256(root/'LN_TRAINING_SEAL.json') or s['variant']!='layer_norm' or not s['strict_checkpoint_load']:
            raise ValueError('Invalid LN post-seal score')
        v=result(root,task)
        output.append({k:task[k] for k in ('method','group','source_candidate_id','seed','seed_index','task_id')} | {
            'condition':'layer_norm','historical':False,'validation_accuracy':v['best_validation_accuracy'],**s,
            'best_epoch':v['best_epoch'],'stopped_epoch':v['stopped_epoch'],'parameter_count':v['parameter_count'],
            'peak_gpu_memory_allocated_bytes':v['peak_gpu_memory_allocated_bytes'],'elapsed_seconds_training':v['elapsed_seconds'],
            'stage_timing':v['stage_timing'],'checkpoint_io_seconds':v['checkpoint_io_seconds'],'checkpoint':str(checkpoint)})
    if len(output)!=300:raise ValueError('LN final scores incomplete')
    seal(root/'LN_SEED_RESULTS.json',output);csv_write(root/'LN_SEED_RESULTS.csv',output)
    print(json.dumps({'scores':len(output),'variant':'layer_norm','strict':True}),flush=True)
