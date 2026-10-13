"""Reuse frozen unlabeled initialization, verify dependencies, seal before labels."""
import importlib.metadata
import shutil
import sys
import time
from pathlib import Path
from .common import *
from . import graphs

BROOT=Path('/home/fangxy/D-VAE/results/wi_schur_B_pool300_20261011v1')
OLDAC=Path('/home/fangxy/D-VAE/results/wi_schur_AC_pool300_20261012v1')
TRAIN_FILES=('eval_utils.py','deterministic_runtime.py','dataset_utils.py','analyse/structured_kernel_followup/validation_train.py','hp_modes.py','evaluation_errors.py')

def verify(root):
    from analyse.wi_schur_B_pool300.prepare import verify as old_b
    from analyse.wi_schur_AC_pool300.prepare import verify as old_ac
    f=read(root/'freeze.json');old_b(BROOT);old_ac(OLDAC)
    if f['identity']!=digest({k:v for k,v in f.items() if k!='identity'}):raise ValueError('Open freeze changed')
    for rel,h in f['inputs'].items():
        if sha(root/rel)!=h:raise ValueError('Open frozen input changed: '+rel)
    for rel,h in f['source'].items():
        if sha(rel)!=h:raise ValueError('Open source changed; explicit engineering recovery record required: '+rel)
    return f

def prepare(root,prompt):
    if (root/'freeze.json').exists():return verify(root)
    from analyse.wi_schur_B_pool300.prepare import verify as old_b,bundle_data
    from analyse.wi_schur_AC_pool300.prepare import verify as old_ac
    from analyse.structured_kernel_followup.validation_train import train_validation_only
    from .initialization import select
    from . import gp,search,runtime,report,scoring,optimize,checks
    started=time.perf_counter();bf=old_b(BROOT);af=old_ac(OLDAC)
    for old in [BROOT,OLDAC]:
        if read(old/'COMPLETE.json')['state']!='COMPLETE':raise ValueError('Historical source not completed')
    bp=read(BROOT/'protocol.json');actual=digest({k:sha(k) for k in TRAIN_FILES})
    if actual!=bp['evaluation_code_identity']:raise ValueError('Training execution identity does not match historical sources')
    for name,version in bf['environment'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Training dependency changed: '+name)
    if (root/'tasks.sqlite').exists():
        with connect(root) as d:
            if d.execute('SELECT COUNT(*) FROM logical').fetchone()[0]:raise ValueError('No preparation after observation release')
    bundle,context=bundle_data()
    if context!=bp['dataset_context']:raise ValueError('Dataset or split changed')
    root.mkdir(parents=True,exist_ok=True);initial=[];seeds_all={};representation={}
    for g in range(5):
        dst=inputs(root,g);dst.mkdir(parents=True,exist_ok=True)
        for name in ['candidates.json','public_features.json','wl_vocabulary.json','seeds.json','initialization.json']:
            src=BROOT/'groups'/str(g)/name
            if (dst/name).exists() and sha(dst/name)!=sha(src):raise ValueError('Existing input differs from source')
            shutil.copyfile(src,dst/name)
        rows=read(dst/'candidates.json');vocab=read(dst/'wl_vocabulary.json');s=read(dst/'seeds.json');binit=read(dst/'initialization.json')
        if not (1<=vocab['n_max']<=7 and 1<=vocab['e_max']<=16):raise ValueError('Invalid source WL graph scales')
        if len(rows)!=768:raise ValueError('Source pool size changed')
        for row in rows:
            graphs.valid(row['configuration'])
            if digest(row['configuration'])!=row['candidate_id'] or graphs.aid(row['configuration'])!=row['architecture_id']:raise ValueError('Source canonical identity differs')
        if s['search_training_seed']!=seed(g,'search') or s['qlogei_mc']!=[seed(g,'mc',i) for i in range(1,251)]:raise ValueError('Search/MC seed mismatch')
        for stage,key in [('validation','validation_training_seeds'),('final','final_training_seeds')]:
            if s[key]!=[seed(g,stage,i) for i in range(30)]:raise ValueError('Validation/final seed mismatch')
        all_train=[s['search_training_seed']]+s['validation_training_seeds']+s['final_training_seeds']
        if len(all_train)!=len(set(all_train)):raise ValueError('Training phase seed overlap')
        csrc=OLDAC/'trajectories/C'/str(g)/'initialization.json';cinit=read(csrc)
        rebuilt=select(read(dst/'public_features.json'),binit['first_point'])
        if rebuilt['ids']!=cinit['ids'] or rebuilt['transform']!=cinit['transform']:raise ValueError('C unlabeled initialization differs from sealed rule')
        for method,obj,source in [('A',binit,BROOT/'groups'/str(g)/'initialization.json'),('C',cinit,csrc)]:
            f=folder(root,method,g);f.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,f/'initialization.json')
            initial.append({'method':method,'group':g,'ids':obj['ids'],'source':str(source),'sha256':sha(source),'unlabelled_rule_verified':True})
        proposals={str(phase):[proposal_seed(g,phase,i) for i in range(1,251)] for phase in [11,12,21,22,23,24]}
        seeds_all[str(g)]={'old':s,'proposals':proposals}
        representation[str(g)]={'n_max':vocab['n_max'],'e_max':vocab['e_max'],'fallback_ids':[r['candidate_id'] for r in rows if any(i not in {e[1] for e in r['configuration']['edges']} for i in range(1,len(r['configuration']['operations'])+1))]}
        print({'group':g,'initializations_verified':True,'fixed_attributes_verified':True},flush=True)
    write(root/'INITIALIZATION_MANIFEST.json',initial);write(root/'SEEDS.json',{'groups':seeds_all,'bootstrap':proposal_seed(0,31,0)})
    protocol={'version':'AC_open_graph_mix_v1_20261012','frozen_at':now(),'B_root':str(BROOT),'old_AC_root':str(OLDAC),
      'cache_sources':[str(OLDAC),str(BROOT)],'B_new_search_or_training_trajectories':0,'groups':list(range(5)),'methods':['A','C'],
      'method_definitions':{'A':'RBF-Schur50; RAW16; 256 Sobol +8 L-BFGS-B; float32 VAE landing','C':'old unlabeled WI-Schur50; full stable-token WI; 8 random+8 mutated graphs; discrete enumeration +continuous HP gradients'},
      'logical_budget':{'search':3000,'validation':3000,'final':300,'total':6300,'each_trajectory':[50,250,300,30]},
      'search_recipe':bp['search_recipe'],'final_recipe':bp['final_recipe'],'dataset_context':context,'vae':bp['vae'],'evaluation_code_identity':actual,
      'fixed_configuration':graphs.FIXED,'graph_family':{'operators':graphs.OPS,'m':[1,2,3,4,5],'output':'all and only internal loose ends connect END','max_nodes':7,'max_edges':16,'all_Identity_rejected':True,'missing_incoming_fallback_preserved':True},
      'representation':representation,'C_full_history_tokens':'recursive H0/H1/H2 semantic hashes; all seen tokens as explicit axes, T-union-Q expansion; complete norms and shared new-token inner products',
      'A_generated_latent':'original12 for initialization, generated little-endian float32 after landing; no encoder_mu replacement','C_latent':None,
      'GP':{'A':'isotropic Matern5/2 RAW16 four parameters','C':'sf2*(.25*kW+.25*kH+.5*kW*kH), two isotropic Matern5/2, five parameters',
       'scales':'own seen observations only; ddof0; positive pair medians; zero y scale blocks','cold_starts':[.5,1.,2.],'Adam_steps':300,'Adam_lr':.01,'LBFGS':gp.LBFGS,
       'initial_mean':0.,'initial_signal':1.,'initial_noise':.01,'floors':[1e-4,1e-8,1e-6],'jitter':[1e-8,1e-7,1e-6],'parameters_fixed_during_acquisition':True},
      'acquisition':{'q':1,'MC':512,'same_old_step_seed':True,'fat':True,'tau_max':.01,'tau_relu':1e-6,'eta':.001,'posterior':'latent no added likelihood noise',
        'candidate_requests':2048,'main_seconds':270,'total_seconds':300,'external_native_call_supervisor':True,'requests_counted_before_call':True},
      'A_optimizer':{'Sobol16':256,'scramble':True,'starts':8,'max_requests_each':200,'maxiter':200,'maxls':20,'ftol':1e-9,'gtol':1e-6,'post':64,'fallback_Sobol':128,
         'decoder_salt':'AC_open_graph_mix_v1_20261012\\0','seed':'first4 little endian SHA256(salt+clipped latent12 float32 bytes)','decoder_trials':5},
      'C_optimizer':{'random_graphs':8,'mutations':8,'attempts_per_flow':128,'parents':'top5 different own seen structures by best search validation then structureID',
         'mutation':'uniform parent; .5 operator replacement, .5 possible internal edge toggle; rebuild outputs','hidden':graphs.HIDDEN,'l2':graphs.L2,
         'shared_Sobol2_points':2,'coarse_per_graph':60,'refine_starts_per_graph':2,'refine_requests_each':30,'maxiter':30,'maxls':20,'ftol':1e-9,'gtol':1e-6,'post':64,'fallback':64,'fallback_graph_attempts':128,
         'online_encoder_calls':0,'online_decoder_calls':0},
      'proposal_seed':{'base':2026101223,'formula':'uint32(SeedSequence([base,group,phase,index]).generate_state(1))','phases':[11,12,21,22,23,24,31]},
      'training_seed_rule':bp['seed_rule'],'validation_selection':'Top10 own search scores then30seed means/CID; 30 disjoint final seeds; strict first-best checkpoint',
      'test_gate':'all10 champions/300 final logical models sealed before separate float32 unique-checkpoint scoring','historical_development_benchmark':True,'historical_final_seeds_previously_used':True,'H_access':False,
      'statistics':{'unit':'5 paired group means; pergroup30finalseed mean/SD','comparison':'C-A','bootstrap_replicates':10000,'bootstrap_seed':proposal_seed(0,31,0),'percentiles':[2.5,97.5],'confirmatory_claim':False},
      'scheduler':{'trajectories':TRAJECTORIES,'CPU_numerical_workers':2,'threads_each':1,'GPU_workers_max':2,'score_priority':False},
      'ETA':'first measured after A0/C0 each10online; current-run stage timings; initial due=ETA+900s, mutable until first trigger; healthy recheck300s',
      'CPU_sanity_tolerances':{'absolute':1e-7,'relative':1e-6,'finite_difference_step':1e-6,'fixed_points':[.43,.37]},'prompt_sha256':sha(prompt)}
    write(root/'PROTOCOL.json',protocol);shutil.copyfile(prompt,root/'PROMPT.txt')
    checks.sanity(root)
    source={}
    for m in list(sys.modules.values()):
        fn=getattr(m,'__file__',None)
        if fn:
            p=Path(fn).resolve()
            if p.is_file() and p.suffix=='.py' and p.is_relative_to(Path.cwd()):source[str(p.relative_to(Path.cwd()))]=sha(p)
    for p in Path('analyse/ac_open_graph_mix').glob('*'):
        if p.suffix in ['.py','.sh']:source[str(p)]=sha(p)
    source['surrogate/checkpoint_io.py']=sha('surrogate/checkpoint_io.py');write(root/'SOURCE_MANIFEST.json',source)
    for name in source:
        dest=root/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    immutable=[root/n for n in ['PROTOCOL.json','SOURCE_MANIFEST.json','INITIALIZATION_MANIFEST.json','SEEDS.json','SANITY.json','PROMPT.txt']]
    immutable+=list((root/'inputs').rglob('*.json'))+[folder(root,m,g)/'initialization.json' for m,g in TRAJECTORIES]
    frozen={'source':source,'inputs':{str(p.relative_to(root)):sha(p) for p in immutable},
      'old_B_identity':bf['identity'],'old_AC_identity':af['identity'],'before_new_observation_release':True,'preparation_seconds':time.perf_counter()-started,'sealed_at':now()}
    frozen['identity']=digest(frozen);write(root/'freeze.json',frozen);init_db(root)
    write(root/'STATUS.json',{'state':'PREPARED','pid':None,'updated_at':now()});print({'open_preparation':'PASS','freeze':frozen['identity']},flush=True)
