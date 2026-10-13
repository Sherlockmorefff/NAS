"""Fresh float32 scoring process after the ten-champion/300-model barrier."""
import os
import time
from pathlib import Path
from .common import *

def construct_model(config,in_ch,out_ch,train_seed,device):
    import torch
    import eval_utils as original
    torch.set_default_dtype(torch.float32)
    with original.isolated_rng(train_seed,device):
        model=original.DynamicGNN(config,in_ch,out_ch,dropout=config['dropout'],hidden_dim=config['hidden_dim'],
           gcnii_alpha=config['gcnii_alpha'],gcnii_theta=config['gcnii_theta'],gat_heads=config['gat_heads'],
           sage_aggr=config['sage_aggr'],gin_eps=config['gin_eps'],component_variant='baseline',activation_variant='relu').to(device)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Final model parameters must be float32')
    return model

def score(root):
    import torch
    import numpy as np
    from deterministic_runtime import configure_torch_determinism
    from analyse.wi_schur_B_pool300.prepare import bundle_data
    from .prepare import verify
    verify(root);torch.set_default_dtype(torch.float32);configure_torch_determinism(torch);torch.set_num_threads(1)
    barrier=root/'AC_all_final_models_sealed.json';b=read(barrier)
    if b['logical_models']!=300 or len(b['trajectories'])!=10:raise ValueError('New AC test barrier incomplete')
    bundle,context=bundle_data()
    if context!=read(root/'PROTOCOL.json')['dataset_context']:raise ValueError('Scoring dataset changed')
    device=torch.device('cuda:0');data=bundle.data.to(device)
    if data.x.dtype!=torch.float32:raise ValueError('Data features must be float32')
    output=[];scoring_root=root/'new_test_scores';scoring_root.mkdir(exist_ok=True)
    for method,g in TRAJECTORIES:
        f=folder(root,method,g);champ=read(f/'champion.json');manifest=read(f/'final_models.json')
        check=next(t for t in b['trajectories'] if t['method']==method and t['group']==g)
        if sha(f/'champion.json')!=check['champion_sha256'] or sha(f/'final_models.json')!=check['models_manifest_sha256']:raise ValueError('AC barrier input changed')
        for m in manifest['models']:
            h=m['checkpoint_sha256'];p=Path(m['checkpoint']);out=scoring_root/(h+'.json')
            if sha(p)!=h:raise ValueError('Final checkpoint changed')
            if out.exists():value=read(out)
            else:
                started=time.perf_counter();model=construct_model(champ['configuration'],bundle.num_features,bundle.num_classes,m['seed'],device)
                checkpoint=torch.load(p,map_location='cpu',weights_only=False)
                if checkpoint['identity']!=m['task_id']:raise ValueError('Final model task identity changed')
                model.load_state_dict(checkpoint['model']);model.eval()
                with torch.no_grad():
                    prediction=model(data.x,data.edge_index).argmax(1)
                    accuracy=float(prediction[data.test_mask].eq(data.y[data.test_mask]).sum().item()/int(data.test_mask.sum()))
                value={'checkpoint_sha256':h,'barrier_sha256':sha(barrier),'test_accuracy':accuracy,'computed_at':now(),
                    'elapsed_seconds':time.perf_counter()-started,'scoring_pid':os.getpid(),'dtype':'float32','training_updates':0,
                    'prior_test_exposure':m['prior_test_exposure'],'B_score_value_copied':False}
                seal(out,value);del model;torch.cuda.empty_cache()
            if value['barrier_sha256']!=sha(barrier) or value['computed_at']<b['sealed_at'] or not np.isfinite(value['test_accuracy']):raise ValueError('Invalid post-barrier score')
            output.append({'method':method,'group':g,'candidate_id':champ['candidate_id'],**m,**value})
    if len(output)!=300:raise ValueError('AC final score mappings incomplete')
    seal(root/'AC_final_scores.json',output)
    print({'score':'COMPLETE','logical_models':300,'unique_scores':len(list(scoring_root.glob('*.json')))},flush=True)
