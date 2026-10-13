import math
import shutil
import subprocess
import time
from pathlib import Path
from .common import *

CORE = ['eval_utils.py','deterministic_runtime.py','dataset_utils.py','evaluation_errors.py','hp_modes.py',
        'analyse/structured_kernel_followup/validation_train.py','analyse/structured_kernel_followup/common.py',
        'analyse/structured_kernel_phase1/io.py','analyse/ac_open_graph_mix/scoring.py']

def bundle_data():
    from dataset_utils import resolve_dataset_request, load_dataset_from_request, dataset_context_from_manifest
    b = load_dataset_from_request(resolve_dataset_request('cora',None,cora_root='Cora'))
    c = dataset_context_from_manifest(b.manifest)
    if c['dataset_content_fingerprint'] != '373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7' or c['split_fingerprint'] != 'bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09':
        raise ValueError('Cora data/split fingerprint mismatch')
    return b,c

def prepare(root, prompt):
    if (root/'freeze.json').exists():
        return verify(root)
    if (root/'tasks.sqlite').exists():
        raise ValueError('Unfrozen tasks exist; do not overwrite')
    start=time.perf_counter()
    root.mkdir(parents=True,exist_ok=True)
    origin=read_json(SOURCE/'PROTOCOL.json')
    if read_json(SOURCE/'ACCEPTANCE.json')['checks'] != 'PASS' or read_json(SOURCE/'STATUS.json')['state'] != 'COMPLETE':
        raise ValueError('Historical open AC source not accepted/complete')
    source_manifest=read_json(SOURCE/'SOURCE_MANIFEST.json')
    for p in CORE:
        if sha256(p) != source_manifest[p]:
            raise ValueError('Frozen historical implementation changed: '+p)
    recipe=origin['final_recipe']
    if recipe['max_epochs']!=300 or recipe['patience']!=80 or recipe['track_test'] or recipe['normalization']!='none':
        raise ValueError('Unexpected historical final recipe')
    bundle,context=bundle_data()
    if context!=origin['dataset_context'] or context['torch_version']!='2.5.1+cu121' or context['pyg_version']!='2.7.0':
        raise ValueError('Dataset/dependency context differs')
    source_files={}
    def historical(path):
        path=Path(path);source_files[str(path)]=sha256(path);return read_json(path)
    for n in ['PROTOCOL.json','SOURCE_MANIFEST.json','AC_all_final_models_sealed.json','resources.json']:
        historical(SOURCE/n)
    source_files[str(SOURCE/'FINAL_SEED_RESULTS.csv')]=sha256(SOURCE/'FINAL_SEED_RESULTS.csv')
    barrier=read_json(SOURCE/'AC_all_final_models_sealed.json')
    rows=csv_read(SOURCE/'FINAL_SEED_RESULTS.csv')
    lookup={}
    for row in rows:
        key=(row['method'],int(row['group']),row['candidate_id'],int(row['seed']))
        if key in lookup:
            raise ValueError('Duplicate baseline pair key')
        lookup[key]=row
    if len(lookup)!=300:
        raise ValueError('Historical None must have exactly 300 rows')
    champions=[];baselines=[];ln_tasks=[];pairs=[]
    seeds_by_group={g:historical(SOURCE/'inputs/groups'/str(g)/'seeds.json')['final_training_seeds'] for g in range(5)}
    for m,g in TRAJECTORIES:
        f=SOURCE/'trajectories'/m/str(g)
        champ=historical(f/'champion.json');models=historical(f/'final_models.json')
        cid=champ['candidate_id'];config=champ['configuration']
        if cid!=EXPECTED_CIDS[(m,g)] or digest(config)!=cid or config['normalization']!='none':
            raise ValueError('Frozen champion CID/config differs')
        seal_row=next(x for x in barrier['trajectories'] if x['method']==m and x['group']==g)
        if sha256(f/'champion.json')!=seal_row['champion_sha256'] or sha256(f/'final_models.json')!=seal_row['models_manifest_sha256']:
            raise ValueError('Historical champion/model barrier mismatch')
        seeds=seeds_by_group[g]
        if len(seeds)!=30 or len(set(seeds))!=30 or len(models['models'])!=30:
            raise ValueError('Expected 30 distinct final seeds')
        champions.append({'method':m,'group':g,'source_candidate_id':cid,'configuration':config,'final_seeds':seeds})
        mapped={x['seed']:x for x in models['models']}
        for i,seed in enumerate(seeds):
            row=lookup[(m,g,cid,seed)];model=mapped[seed]
            task_id=row['task_id'];checkpoint=Path(row['checkpoint']);weights_ok=checkpoint.is_file()
            if task_id!=model['task_id'] or str(checkpoint)!=model['checkpoint'] or row['checkpoint_sha256']!=model['checkpoint_sha256'] or row['barrier_sha256']!=sha256(SOURCE/'AC_all_final_models_sealed.json') or row['computed_at']<barrier['sealed_at'] or row['dtype']!='float32' or int(row['training_updates'])!=0:
                raise ValueError('Baseline CSV/model/score barrier identity mismatch')
            expected_task=digest({'configuration':config,'dataset':context,'recipe':recipe,'seed':seed,'evaluation_code_identity':origin['evaluation_code_identity']})
            if task_id!=expected_task:
                raise ValueError('Baseline final training identity mismatch')
            tf=checkpoint.parent;old=historical(tf/'result.json');execution=historical(tf/'execution.json')
            if old['identity']!=task_id or old['seed']!=seed or old['test_metric_computations']!=0 or float(row['validation_accuracy'])!=old['best_validation_accuracy'] or float(row['validation_accuracy'])!=model['validation_accuracy']:
                raise ValueError('Baseline validation must match original final result')
            if weights_ok and sha256(checkpoint)!=row['checkpoint_sha256']:
                raise ValueError('Historical None weights changed')
            for metric in ['validation_accuracy','test_accuracy']:
                if not math.isfinite(float(row[metric])):
                    raise ValueError('Non-finite baseline score')
            base={'method':m,'group':g,'source_candidate_id':cid,'seed':seed,'seed_index':i,'task_id':task_id,
                  'checkpoint':str(checkpoint),'checkpoint_sha256':row['checkpoint_sha256'],
                  'validation_accuracy':float(row['validation_accuracy']),'test_accuracy':float(row['test_accuracy']),
                  'validation_source':str(SOURCE/'FINAL_SEED_RESULTS.csv')+':validation_accuracy',
                  'test_source':str(SOURCE/'FINAL_SEED_RESULTS.csv')+':test_accuracy','historical':True,
                  'weights_verified_from_source_manifest_only':not weights_ok,'best_epoch':old.get('best_epoch'),
                  'stopped_epoch':old.get('stopped_epoch'),'elapsed_seconds':old.get('elapsed_seconds'),
                  'parameter_count':old.get('parameter_count'),'peak_gpu_memory_allocated_bytes':old.get('peak_gpu_memory_allocated_bytes'),
                  'stage_timing':old.get('stage_timing'),'execution':execution,'original_barrier_sha256':row['barrier_sha256']}
            baselines.append(base)
            ln_config={**config,'normalization':'layer_norm'}
            ln_recipe={**recipe,'normalization':'layer_norm','model':'DynamicGNN layer_norm/relu'}
            new_key={'run_id':root.name,'protocol_version':VERSION,'component_variant':'layer_norm','configuration':ln_config,
                     'seed':seed,'dataset':context,'recipe':ln_recipe,'source':{p:sha256(p) for p in CORE},
                     'new_source':{str(p):sha256(p) for p in sorted(Path('analyse/ac_champions_ln_reuse_none').glob('*.py'))}}
            tid=digest(new_key)
            task={'task_id':tid,'method':m,'group':g,'source_candidate_id':cid,'seed':seed,'seed_index':i,
                  'configuration':ln_config,'component_variant':'layer_norm','from_scratch':True,'ordinal':i*10+TRAJECTORIES.index((m,g))}
            ln_tasks.append(task)
            pairs.append({'method':m,'group':g,'source_candidate_id':cid,'seed':seed,'none_task_id':task_id,'ln_task_id':tid})
    if len({x['task_id'] for x in ln_tasks})!=300:
        raise ValueError('LN task identities are not unique')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    admitted=[]
    for line in gpu.stdout.splitlines():
        index,uuid,util,mem=[x.strip() for x in line.split(',')]
        if float(util)<=5 and float(mem)<128:
            admitted.append({'physical_index':int(index),'uuid':uuid,'admission_utilization':float(util),'admission_memory_MiB':float(mem)})
    admitted=admitted[:2]
    if not admitted:
        raise RuntimeError('No free GPU; do not occupy another experiment GPU')
    protocol={'version':VERSION,'source_run':str(SOURCE),'frozen_at':now(),'dataset_context':context,
              'source_final_recipe':recipe,'ln_final_recipe':ln_recipe,'component_variant':'layer_norm','activation_variant':'relu',
              'LN':{'position':'nonIdentity convolution output -> LN -> original ReLU -> original dropout',
                    'eps':1e-5,'elementwise_affine':True,'gamma':1,'beta':0,'per_node_hidden_feature_statistics':True,
                    'Identity_output_and_GIN_internal_MLP_LN':False,'optimizer_weight_decay_all_parameters':True},
              'new_training_budget':300,'historical_none_results':300,'paired_results':300,'conditions':600,
              'None_new_training':0,'None_new_test_scoring':0,'historical_LN_cache':False,'LN_warm_start':False,
              'NAS_GP_VAE_calls':0,'H_accessed':False,'first_batch':'first original final seed of all 10 configurations; part of 300',
              'initialization_evidence':'reconstructed_from_frozen_none_code; no saved historical initialization hashes',
              'test_gate':'all 300 LN models and validation decision sealed before independent strict layer_norm scoring',
              'historical_development_benchmark':True,'baseline_test_previously_observed':True,'seeds_previously_used':True,
              'bootstrap':{'seed':2026101309,'replicates':10000,'unit':'5 group blocks, resample A/C jointly with None/LN pairs retained'},
              'direction_tolerance_pp':1e-12,'validation_recommendation':'overall, A and C means each >1e-12 pp; before LN test',
              'ETA':{'first_after':10,'acceptance_buffer_seconds':900,'healthy_recheck_seconds':300,'status_seconds':60,'progress_ETA_seconds':300},
              'prompt_sha256':sha256(prompt),'automatic_fixed_LN_NAS':False}
    write_json(root/'PROTOCOL.json',protocol)
    write_json(root/'BASELINE_MANIFEST.json',{'source_files':source_files,'rows':baselines,'exact_key_join':True,'validation_from_final_CSV_only':True,'None_rescored':False})
    write_json(root/'BASELINE_SOURCE_FILES.json',source_files)
    write_json(root/'INPUT_MANIFEST.json',{'champions':champions,'source_root':str(SOURCE)})
    write_json(root/'PAIR_MANIFEST.json',{'pairs':pairs})
    write_json(root/'LN_TASK_MANIFEST.json',{'tasks':sorted(ln_tasks,key=lambda x:x['ordinal'])})
    csv_write(root/'BASELINE_VALIDATION.csv',[{k:v for k,v in b.items() if k not in ('test_accuracy','test_source','execution')} for b in baselines])
    csv_write(root/'BASELINE_TEST.csv',[{k:b[k] for k in ('method','group','source_candidate_id','seed','task_id','checkpoint','checkpoint_sha256','test_accuracy','test_source','historical')} for b in baselines])
    (root/'PROMPT.txt').write_bytes(Path(prompt).read_bytes())
    (root/'ERRORS.jsonl').touch(exist_ok=True)
    write_json(root/'resources.json',{'python':str(Path('/home/fangxy/.conda/envs/nas/bin/python')),'GPUs':admitted,'threads_per_worker':1,'original':read_json(SOURCE/'resources.json')})
    from .checks import sanity
    write_json(root/'SANITY.json',sanity(champions))
    source={str(p):sha256(p) for p in sorted(Path('analyse/ac_champions_ln_reuse_none').glob('*')) if p.suffix in ('.py','.sh')}
    source.update({p:sha256(p) for p in CORE})
    write_json(root/'SOURCE_MANIFEST.json',source)
    for p in Path('analyse/ac_champions_ln_reuse_none').glob('*'):
        if p.suffix in ('.py','.sh'):
            (root/'code').mkdir(exist_ok=True);shutil.copy2(p,root/'code'/p.name)
    inputs={p.name:sha256(p) for p in root.iterdir() if p.name in ['PROTOCOL.json','BASELINE_MANIFEST.json','BASELINE_SOURCE_FILES.json','INPUT_MANIFEST.json','PAIR_MANIFEST.json','LN_TASK_MANIFEST.json','BASELINE_VALIDATION.csv','BASELINE_TEST.csv','PROMPT.txt','SANITY.json','SOURCE_MANIFEST.json']}
    frozen={'source':source,'inputs':inputs,'frozen_at':now(),'preparation_seconds':time.perf_counter()-start}
    frozen['identity']=digest(frozen);write_json(root/'freeze.json',frozen)
    with connect(root) as db:
        db.executescript('CREATE TABLE tasks(id TEXT PRIMARY KEY, ordinal INTEGER, status TEXT, worker INTEGER, pid INTEGER, started REAL, finished REAL, elapsed REAL, attempts INTEGER DEFAULT 0, error TEXT); CREATE TABLE attempts(task TEXT, attempt INTEGER, worker INTEGER, started REAL, finished REAL, exit_code INTEGER, error TEXT, PRIMARY KEY(task,attempt));')
        db.executemany('INSERT INTO tasks(id,ordinal,status) VALUES(?,?,?)',[(t['task_id'],t['ordinal'],'queued') for t in ln_tasks])
    write_json(root/'QUEUE_ENQUEUED.json',{'all_300_enqueued_unix':time.time(),'at':now()})
    write_json(root/'STATUS.json',{'state':'PREPARED','LN_completed':0,'LN_total':300,'None_new_training':0,'updated_at':now()})
    return frozen
