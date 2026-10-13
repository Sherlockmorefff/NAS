"""Freeze all five label-free pools and the complete B protocol before training."""
import importlib.metadata
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import qmc, norm
from .common import BASE, seed, read, write, sha, digest, now, rbf_schur, init_db

def bundle_data():
    from dataset_utils import resolve_dataset_request,load_dataset_from_request,dataset_context_from_manifest
    request=resolve_dataset_request('cora',None,cora_root='Cora')
    bundle=load_dataset_from_request(request);context=dataset_context_from_manifest(bundle.manifest)
    if context['dataset_content_fingerprint']!='373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7' or context['split_fingerprint']!='bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09':
        raise ValueError('Unexpected Cora data/split identity')
    return bundle,context

def verify(root):
    frozen=read(root/'freeze.json')
    if frozen['identity']!=digest({k:v for k,v in frozen.items() if k!='identity'}):raise ValueError('Freeze identity changed')
    for rel,h in frozen['inputs'].items():
        if sha(root/rel)!=h:raise ValueError('Frozen input changed: '+rel)
    for rel,h in frozen['source'].items():
        if sha(Path.cwd()/rel)!=h:raise ValueError('Code changed; record engineering recovery before resume: '+rel)
    if sha(Path(frozen['vae']['path']))!=frozen['vae']['sha256']:raise ValueError('VAE weight changed')
    return frozen

def prepare(root, prompts):
    if (root/'freeze.json').exists(): return verify(root)
    if (root/'tasks.sqlite').exists():
        from .common import counts
        if sum(counts(root)['actual'].values()):raise ValueError('Cannot prepare after any training task exists')
    import torch
    from deterministic_runtime import configure_torch_determinism
    configure_torch_determinism(torch);torch.set_num_threads(1)
    from architecture_diagnostics import load_joint_vae
    from eval_utils import _decode_arch,decode_hp_by_mode
    from analyse.structured_kernel_phase1.features import build_operator_graph,directed_wl
    from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter
    from analyse.structured_kernel_followup.validation_train import train_validation_only
    from . import gp
    started=time.perf_counter();root.mkdir(parents=True,exist_ok=True)
    checkpoint=Path('checkpoints/joint_model_pipeline_global4_best.pth').resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    weight_hash=sha(checkpoint)
    if weight_hash!='e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4':raise ValueError('Unexpected original VAE weight')
    model=load_joint_vae(checkpoint,'global4',device='cpu')
    for p in model.parameters():p.requires_grad_(False)
    bundle,context=bundle_data()
    recipe={'dataset':'cora','split':'planetoid_public','graph_transform':'NormalizeFeatures','batch':'full_batch',
       'hp_mode':'global4','normalization':'none','dtype':'float32','model':'DynamicGNN baseline/relu',
       'optimizer':'Adam original defaults; lr=config.lr, weight_decay=config.l2',
       'scheduler':'CosineAnnealingLR(T_max=max_epochs,eta_min=lr*.01)',
       'checkpoint_selection':'strict validation improvement, first best retained','track_test':False}
    protocol={'scope':'B only; 固定候选池内的完整B搜索','groups':list(range(5)),'pool_size':768,'initial':50,'online':250,
       'budget':{'search':1500,'validation':1500,'final':150,'total_logical_training':3150},
       'search_recipe':dict(recipe,max_epochs=150,patience=40),'final_recipe':dict(recipe,max_epochs=300,patience=80),
       'dataset_context':context,'vae':{'path':str(checkpoint),'sha256':weight_hash,'frozen':True},
       'lhs':{'domain':'normal_ppf(LHS[:12])*0.8; HP=LHS[12:]; float32 clipping','z_bound':2.5,
              'HP_bounds':[-4.,-1.5,.1,.6],'decoder_trials':5,'decoder_policy':'historical_modal3',
              'dedup':'actual complete configuration; first valid generation index retained','supplement_batches':768},
       'initialization':'float64 original global RBF-Schur, ell=sqrt16, epsilon1e-6, no labels',
       'surrogate':{'kernel':'sf2*(.25*kW+.25*kH+.5*kW*kH)','base':'isotropic Matern5/2',
          'likelihood':'GaussianLikelihood learned unknown noise; no known SE',
          'input_scaler':'seen observations only, ddof0; positive pdist median scaling; WL columns unscaled',
          'starts':[.5,1.,2.],'Adam_steps':300,'Adam_lr':.01,'LBFGS':gp.LBFGS,
          'parameters':5,'selection':'finite training NLL only; all starts recorded','matrix_dtype':'float64'},
       'acquisition':{'kind':'qLogEI','q':1,'Sobol_samples':512,'chunk_size':64,'steps':list(range(1,251)),
          'tie':'score descending, frozen candidate ID ascending','terminal_mean_query':False},
       'final_evaluation':{'topk':10,'validation_seeds':30,'final_seeds':30,'champion_rule':'30 validation mean then candidate ID',
          'new_training_seeds_same_node_validation_split':True,'test_barrier':'all 5 champions and 150 final models sealed first',
          'test_role':'historical development benchmark','H_access':False},
       'seed_rule':{'base':BASE,'formula':'uint32(SeedSequence([base,group,stage_code,index]).generate_state(1))',
           'stage_codes':{'lhs':1,'decoder':2,'search':3,'mc':4,'validation':5,'final':6},'method_name_included':False},
       'bootstrap':{'unit':'5 complete search groups','replicates':10000,'seed':2026101102,'percentiles':[2.5,97.5]},
       'authority':'original Schur CLI/validation constraints 150/40, global4; independent entry preserves 50+250',
       'not_original_continuous_BO':True,'three_seed_observation_protocol_reproduced':False,'automatic_AC':False,
       'created_at':now()}
    for g in range(5):
        completed_folder=root/'groups'/str(g)
        completed_files=('candidates.json','generation.json','wl_vocabulary.json','public_features.json','seeds.json','initialization.json')
        if all((completed_folder/name).is_file() for name in completed_files):
            existing=read(completed_folder/'candidates.json')
            if len(existing)!=768 or len({r['candidate_id'] for r in existing})!=768 or any(digest(r['configuration'])!=r['candidate_id'] for r in existing):raise ValueError('Existing unlabelled pool invalid')
            print({'reused_unlabelled_group':g,'candidate_count':len(existing),'no_redecoding':True},flush=True)
            continue
        if completed_folder.exists() and any(completed_folder.iterdir()):raise ValueError('Incomplete pool preparation; retain existing files and repair exact group explicitly')
        rows=[];seen=set();raw=invalid=duplicates=0;batch=0;generation=[]
        while len(rows)<768:
            if batch>=20:raise RuntimeError('Could not fill legal unique pool within frozen 20 batches')
            lhs_seed=seed(g,'lhs',batch);unit=qmc.LatinHypercube(d=16,seed=lhs_seed).random(768)
            vectors=np.concatenate((norm.ppf(np.clip(unit[:,:12],1e-6,1-1e-6))*.8,unit[:,12:]),axis=1)
            vectors[:,:12]=np.clip(vectors[:,:12],-2.5,2.5);vectors[:,12:]=np.clip(vectors[:,12:],0,1)
            vectors=vectors.astype(np.float32)
            generation.append({'batch':batch,'lhs_seed':lhs_seed,'generated_rows':768})
            for z in vectors:
                index=raw;raw+=1;dseed=seed(g,'decoder',index)
                arch=_decode_arch(model,torch.tensor(z[:12]),torch.device('cpu'),n_trials=5,decoder_seed=dseed)
                if arch is None: invalid+=1;continue
                hp=decode_hp_by_mode(torch.tensor(z),12,arch,'global4',-4.,-1.5,.1,.6)
                config={'operations':arch['operations'],'edges':sorted([list(map(int,e)) for e in arch['edges']]),
                    'normalization':'none','hp_mode':'global4','effective_layers':arch['effective_layers'],
                    'lr':hp['lr'],'dropout':hp['dropout'],'hidden_dim':hp['hidden_dim'],'l2':hp['weight_decay'],
                    'gat_heads':hp['gat_heads'],'sage_aggr':hp['sage_aggr'],'gin_eps':hp['gin_eps'],
                    'gat_heads_by_layer':None,'sage_aggr_by_layer':None,'gin_eps_by_layer':None,
                    'gcnii_alpha':.1,'gcnii_theta':.5}
                try:graph=build_operator_graph(config)
                except ValueError:invalid+=1;continue
                cid=digest(config)
                if cid in seen:duplicates+=1;continue
                seen.add(cid);aid=digest(graph)
                actual_hp=[float(np.log10(config['lr'])),config['dropout'],float(np.log2(config['hidden_dim'])),float(np.log10(config['l2']))]
                rows.append({'candidate_id':cid,'architecture_id':aid,'configuration':config,'z_search':z.tolist(),
                    'x_raw':z[:12].astype(float).tolist()+actual_hp,'generation_index':index,'decoder_seed':dseed})
                if len(rows)==768:break
            batch+=1
        folder=root/'groups'/str(g);folder.mkdir(parents=True)
        write(folder/'candidates.json',rows)
        write(folder/'generation.json',{'raw_decoded':raw,'invalid':invalid,'duplicates':duplicates,'valid_unique':768,
             'LHS_generated':768*batch,'LHS_unused':768*batch-raw,'batches':generation,'performance_reads':0})
        graphs={r['architecture_id']:build_operator_graph(r['configuration']) for r in rows}
        features,manifest=directed_wl(graphs);manifest['pool']='this group frozen768 unlabeled architectures';manifest['identity']=digest({k:v for k,v in manifest.items() if k!='identity'})
        write(folder/'wl_vocabulary.json',manifest)
        write(folder/'public_features.json',{'ids':[r['candidate_id'] for r in rows],
            'x_raw':[r['x_raw'] for r in rows],'wl':[features[r['architecture_id']] for r in rows]})
        seeds={'group':g,'search_training_seed':seed(g,'search'),'qlogei_mc':[seed(g,'mc',i) for i in range(1,251)],
            'validation_training_seeds':[seed(g,'validation',i) for i in range(30)],
            'final_training_seeds':[seed(g,'final',i) for i in range(30)],'lhs_batches':generation}
        training=[seeds['search_training_seed']]+seeds['validation_training_seeds']+seeds['final_training_seeds']
        if len(training)!=len(set(training)):raise ValueError('Overlapping stage training seeds')
        write(folder/'seeds.json',seeds)
        initial=rbf_schur(rows);initial['training_seed']=seeds['search_training_seed'];write(folder/'initialization.json',initial)
        print({'frozen_group':g,'raw':raw,'invalid':invalid,'duplicates':duplicates,'wl_dimension':manifest['psi_dimension']},flush=True)
    files={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        p=Path(filename).resolve()
        if p.suffix=='.py' and p.is_file() and p.is_relative_to(Path.cwd()):files[str(p.relative_to(Path.cwd()))]=sha(p)
    for p in Path('analyse/wi_schur_B_pool300').glob('*.py'):files[str(p)]=sha(p)
    # Explicitly include indirect atomic checkpoint IO, and every actual imported project dependency.
    files['surrogate/checkpoint_io.py']=sha('surrogate/checkpoint_io.py')
    protocol['evaluation_code_identity']=digest({k:v for k,v in files.items() if k in ['eval_utils.py','deterministic_runtime.py','dataset_utils.py','analyse/structured_kernel_followup/validation_train.py','hp_modes.py','evaluation_errors.py']})
    write(root/'protocol.json',protocol);write(root/'source_manifest.json',files)
    for rel in files:
        dest=root/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(rel,dest)
    for i,p in enumerate(prompts):shutil.copyfile(p,root/('PROMPT.txt' if i==0 else 'SUPPLEMENT.txt'))
    inputs={str(p.relative_to(root)):sha(p) for p in root.rglob('*.json') if 'code' not in p.parts}
    frozen={'source':files,'inputs':inputs,'vae':protocol['vae'],'dataset_context':context,'sealed_at':now(),
        'preparation_seconds':time.perf_counter()-started,'labels_produced_before_freeze':0,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'environment':{name:importlib.metadata.version(name) for name in ('torch','gpytorch','botorch','scipy','numpy','torch-geometric')}}
    frozen['identity']=digest(frozen);write(root/'freeze.json',frozen);init_db(root)
    write(root/'status.json',{'state':'PREPARED','updated_at':now(),'pid':None,'training_started':False})
    return frozen
