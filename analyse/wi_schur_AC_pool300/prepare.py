import importlib.metadata
import shutil
import sqlite3
import sys
from pathlib import Path
from .common import *

BROOT=Path('/home/fangxy/D-VAE/results/wi_schur_B_pool300_20261011v1')
TRAIN_FILES=('eval_utils.py','deterministic_runtime.py','dataset_utils.py','analyse/structured_kernel_followup/validation_train.py','hp_modes.py','evaluation_errors.py')

def fallback_ids(rows):
    return [r['candidate_id'] for r in rows if any(i not in {e[1] for e in r['configuration']['edges']} for i in range(1,len(r['configuration']['operations'])+1))]

def verify(root):
    from analyse.wi_schur_B_pool300.prepare import verify as verify_B
    f=read(root/'freeze.json');b=verify_B(Path(f['B_root']))
    if b['identity']!=f['B_identity']:raise ValueError('B freeze changed')
    for rel,h in f['inputs'].items():
        if sha(root/rel)!=h:raise ValueError('AC frozen input changed: '+rel)
    for rel,h in f['source'].items():
        if sha(rel)!=h:raise ValueError('AC source changed: '+rel)
    if f['identity']!=digest({k:v for k,v in f.items() if k!='identity'}):raise ValueError('AC freeze identity changed')
    return f

def prepare(root,prompt):
    if (root/'freeze.json').exists():return verify(root)
    from analyse.wi_schur_B_pool300.prepare import verify as verify_B,bundle_data
    from analyse.structured_kernel_followup.validation_train import train_validation_only
    from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter
    from .initialization import select
    from . import gp,search,runtime,report,scoring
    b=verify_B(BROOT);bp=read(BROOT/'protocol.json');done=read(BROOT/'COMPLETE.json')
    if done['state']!='COMPLETE' or done['accepted']['checks']!='PASS':raise ValueError('B not complete/accepted')
    if (root/'tasks.sqlite').exists():
        with connect(root) as db:
            if db.execute('SELECT count(*) FROM logical').fetchone()[0]:raise ValueError('Cannot prepare after new observation requests')
    for name,version in b['environment'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Dependency version differs from B: '+name)
    actual=digest({k:sha(k) for k in TRAIN_FILES})
    if actual!=bp['evaluation_code_identity']:raise ValueError('Actual evaluator identity does not match B')
    # B verify above also checks models.py, nas_space.py and every frozen imported model dependency.
    bundle,context=bundle_data()
    if context!=bp['dataset_context']:raise ValueError('Dataset context differs from B')
    with sqlite3.connect(BROOT.joinpath('tasks.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        if db.execute('SELECT count(*) FROM tasks WHERE status=?',('succeeded',)).fetchone()[0]!=3150:raise ValueError('Missing B physical tasks')
    root.mkdir(parents=True,exist_ok=True);fallback={};references={}
    for g in range(5):
        dst=inputs(root,g);dst.mkdir(parents=True,exist_ok=True)
        for name in ('candidates.json','public_features.json','wl_vocabulary.json','seeds.json','initialization.json','generation.json'):
            source=BROOT/'groups'/str(g)/name;target=dst/name
            if target.exists() and sha(target)!=sha(source):raise ValueError('Existing copied B input differs')
            if not target.exists():shutil.copyfile(source,target)
            references[str(target.relative_to(root))]={'source':str(source),'sha256':sha(source)}
        rows=read(dst/'candidates.json');seeds=read(dst/'seeds.json');initial=read(dst/'initialization.json')
        if len(rows)!=768 or any(digest(r['configuration'])!=r['candidate_id'] for r in rows):raise ValueError('Invalid B complete configurations')
        expected=[950174653,3648874041,1886438676,1976660120,2383257882][g]
        if seeds['search_training_seed']!=expected or expected!=seed(g,'search'):raise ValueError('Group search seed mismatch')
        if seeds['qlogei_mc']!=[seed(g,'mc',i) for i in range(1,251)]:raise ValueError('MC seeds differ')
        for stage,field in [('validation','validation_training_seeds'),('final','final_training_seeds')]:
            if seeds[field]!=[seed(g,stage,i) for i in range(30)]:raise ValueError('Final seed list differs')
        used=[expected]+seeds['validation_training_seeds']+seeds['final_training_seeds']
        if len(used)!=len(set(used)):raise ValueError('Stage training seeds overlap')
        ap=folder(root,'A',g);cp=folder(root,'C',g);ap.mkdir(parents=True,exist_ok=True);cp.mkdir(parents=True,exist_ok=True)
        seal(ap/'initialization.json',{'method':'A','ids':initial['ids'],'first_point':initial['first_point'],
            'source':str(BROOT/'groups'/str(g)/'initialization.json'),'source_sha256':sha(dst/'initialization.json'),'labels_used_for_selection':False})
        c=select(read(dst/'public_features.json'),initial['first_point']);c['overlap_with_B_initial50']=len(set(c['ids'])&set(initial['ids']))
        seal(cp/'initialization.json',c)
        fallback[str(g)]=fallback_ids(rows)
        print({'group':g,'C_B_initial_overlap':c['overlap_with_B_initial50'],'fallback_configurations':len(fallback[str(g)]),'no_performance_labels_read':True},flush=True)
    protocol={'scope':'matched A/C extension on frozen B768 pools','B_root':str(BROOT),'B_freeze_identity':b['identity'],
      'extension_after_Btest':True,'historical_development_benchmark':True,'H_access':False,'groups':list(range(5)),
      'methods':{'A':'RBF-Schur50 + single-seed RAW16 GP qLogEI250','C':'fixed unlabelled W-I-Schur50 + single-seed W-I GP qLogEI250'},
      'budgets':{'each_method':{'search':1500,'validation':1500,'final':150,'total':3150},'AC_logical':6300,'ABC_logical':9450},
      'B_training_read_only':True,'seed_rule':bp['seed_rule'],'search_recipe':bp['search_recipe'],'final_recipe':bp['final_recipe'],
      'dataset_context':bp['dataset_context'],'vae':bp['vae'],'evaluation_code_identity':actual,'training_files':list(TRAIN_FILES),
      'A_kernel':{'base':'isotropic Matern5/2 RAW16','parameters':4,'ARD':False,'input':'B x_raw latent12+actual transformedHP4',
        'standardization':'seen mean/std ddof0, constants1; seen strictly-positive16D pdist median'},
      'C_initialization':{'base':'isotropic Matern5/2 W-I','ellW':1.,'ellH':1.,'signal':1.,'weights':[.25,.25,.5],
        'epsilon':1e-6,'scales':'all768 unlabelled HP zscore/ddof0 and separate positive WL/HP distance medians',
        'first_point':'B shared first point','B_GP_parameters_read':False,'B_observations_read':False},
      'C_GP':bp['surrogate'],'GP_common':{'starts':[.5,1.,2.],'Adam_steps':300,'Adam_lr':.01,'LBFGS':gp.LBFGS,
        'mean_initial':0.,'outputscale_initial':1.,'noise_initial':.01,'noise_lower':1e-6,'ell_lower':1e-4,'signal_lower':1e-8,
        'likelihood':'GaussianLikelihood unknown noise, no known SE','fit_selection':'finite training NLL only; all starts preserved'},
      'acquisition':bp['acquisition'],'final_evaluation':{'top10':10,'validation_seeds':30,'final_seeds':30,
        'champion':'30 validation mean then CID','test_barrier':'all10 AC champions and300 final logical models first sealed',
        'scoring':'fresh isolated float32 process; each unique checkpoint newly rescored after new barrier'},
      'cache_key':'digest(configuration,dataset,recipe,seed,evaluation_code_identity); no method',
      'logic_primary_key':['method','group','phase','slot'],'fit_identity_includes':['method','group','kernel','transform','observations','freeze'],
      'scheduler':{'order':TRAJECTORIES,'training_workers':2,'GP_processes':2,'GP_threads_each':1,'score_dependent_priority':False},
      'statistics':{'unit':'5 matched groups','final_training_seed_statistics':'30 final-seed validation/test mean and SD, then5 group means',
        'main_comparisons':['B-A','C-B'],'auxiliary':'C-A','bootstrap':{'replicates':10000,'seed':2026101102,'common_group_indices':True,'percentiles':[2.5,97.5]},
        'multiplicity_correction':False,'confirmatory_significance_claim':False},
      'representation_limitation':{'missing_incoming_edge_fallback_ids':fallback,'counts':{g:len(ids) for g,ids in fallback.items()},
        'retain_B_recorded_graph_and_DynamicGNN_fallback':True,'no_edge_repair_or_candidate_removal':True},'frozen_at':now()}
    write(root/'AC_PROTOCOL.json',protocol);write(root/'B_input_references.json',references)
    shutil.copyfile(prompt,root/'PROMPT.txt')
    sources={}
    for m in list(sys.modules.values()):
        filename=getattr(m,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if p.is_file() and p.suffix=='.py' and p.is_relative_to(Path.cwd()):sources[str(p.relative_to(Path.cwd()))]=sha(p)
    for p in Path('analyse/wi_schur_AC_pool300').glob('*'):
        if p.suffix in ('.py','.sh'):sources[str(p)]=sha(p)
    sources['surrogate/checkpoint_io.py']=sha('surrogate/checkpoint_io.py')
    for rel in sources:
        target=root/'code'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(rel,target)
    frozen={'B_root':str(BROOT),'B_identity':b['identity'],'source':sources,
        'inputs':{str(p.relative_to(root)):sha(p) for p in root.rglob('*.json') if 'code' not in p.parts},
        'before_AC_observation_release':True,'sealed_at':now()};frozen['identity']=digest(frozen)
    write(root/'freeze.json',frozen);init_db(root)
    write(root/'status.json',{'state':'PREPARED','updated_at':now(),'pid':None})
    print({'preparation':'PASS','identity':frozen['identity']},flush=True)
