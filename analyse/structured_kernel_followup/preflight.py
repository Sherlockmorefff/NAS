import ast
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from .common import *


def random_manifest(ids,initial):
    result={}
    for seed in SEEDS:
        start=initial[str(seed)]['ids'];remaining=sorted(set(ids)-set(start))
        query=np.random.default_rng(np.random.SeedSequence([seed,20261005,2])).permutation(remaining).tolist()[:21]
        assert len(start)==50 and len(set(start+query))==71
        result[str(seed)]={'initial_ids':start,'query_ids':query,'query_kind':['random']*21,
            'pool_ids_sha256':digest(sorted(ids)),'initial_ids_sha256':digest(start),'query_ids_sha256':digest(query),
            'rng':'default_rng(SeedSequence([search_seed,20261005,2]))','budget':{'initial':50,'new_queries':21,'terminal_n':71}}
    return result


def selected_mapping(returns):
    if len(returns)!=20:raise ValueError('Expected 20 return decisions')
    configs={};mapping={}
    for key,row in returns.items():
        cid=row['candidate_id'];config=row['configuration']
        if cid!=config['candidate_id']:raise ValueError('Returned configuration ID mismatch')
        if cid in configs and configs[cid]!=config:raise ValueError('Different configuration reuses ID')
        configs[cid]=config;mapping[key]=cid
    if sorted(configs)!=CONFIG_IDS:raise ValueError('Expected exact frozen five configurations')
    for suffix in (3,6,8):
        s=2026100200+suffix
        if mapping[f'{s}_W-I']!=mapping[f'{s}_RAW16']:raise ValueError('Historical shared return mismatch')
    if mapping['2026100205_W-I']==mapping['2026100205_RAW16']:raise ValueError('Seed205 configurations must differ')
    return {'mapping':mapping,'configurations':configs,'unique_configurations':5,'return_decisions':20}


def protocol():
    return {'purpose':'development_diagnosis_and_selected_configuration_reproducibility','independent_confirmation':False,
        'old_state_preserved':'DEVELOPMENT_REPLAY_NO_STABLE_GAIN','modules_independent':True,
        'M1':{'models':86,'WI_seeds':M1_SEEDS,'n':list(range(50,71)),'RAW16_n50':M1_SEEDS[:2],
            'champion_id':CHAMPION,'new_GP_fits':0,'NLL_atol':1e-8,'posterior_raw_atol':1e-8,
            'decomposition_n':[50,60,70],'kernel_weights':{'architecture':.25,'hp':.25,'interaction':.5},
            'candidate_contribution':'100*y_scale*(C^-1 k_s^T)_i*(y_std_i-constant_mean)',
            'prospective_variance':'latent_std_variance+extra_std_variance+median(current T known_variance_std)',
            'rank_ties':'descending score then full candidate_id; ranks over remaining candidates only',
            'interpretation':'algebraic association, not causal attribution'},
        'M2':{'trajectories':10,'initial':50,'queries':21,'logical_releases':710,'new_GP_fits':0,
            'rng':'default_rng(SeedSequence([search_seed,20261005,2])) over sorted(A_ids-initial_ids)',
            'bootstrap':{'replicates':2000,'seed':2026100531,'quantiles':[.025,.975],'resampling':'same paired indices for all comparisons'},
            'primary':'n71 WI complete strategy minus Random','AUC':'trapezoid cached regret queries0..20 /20',
            'new_gate':None,'score_units':'percentage points'},
        'M3':{'candidate_ids':CONFIG_IDS,'seeds':[3,4,5],'tasks':15,'order':'candidate_id ascending, then seed',
            'dataset':'Cora','split':'public','transform':'NormalizeFeatures','batch':'full_batch','hp_mode':'global4',
            'normalization':'none','max_epochs':300,'patience':80,'dtype':'original float32','worker':1,
            'optimizer':'torch.optim.Adam(lr=config.lr,weight_decay=config.l2); original defaults',
            'scheduler':'CosineAnnealingLR(T_max=300,eta_min=lr*.01); step after each optimizer step',
            'early_stop':'validation accuracy > best; min_delta=0; ties count nonimprovement; first best retained',
            'original_evaluator':'eval_utils.train_and_eval_arch','model':'eval_utils.DynamicGNN baseline/relu',
            'best_state':'original returns maximum validation without restoration; new entry saves and restores corresponding best state and verifies same validation',
            'checkpoint':'every 5 epochs, each improvement, and terminal; model/optimizer/scheduler/early-stop/RNG/trajectory',
            'resume':'same task from saved epoch; completed tasks never retrained; any interrupted recomputation separately logged',
            'test_boundary':'loader loads full graph labels for dataset identity; trainer receives only train/val labels and no test_mask; zero test metrics',
            'analysis':'configuration mean/SE(ddof1)/sqrt3; D_r over all10 mapped pairs; mean/range of three D_r; no bootstrap on repeated configurations'},
        'M4':{'pool':'original768 unlabeled configurations','performance_reads':0,'training':0,
            'classification':['ELIGIBLE_WITH_EVIDENCE','DEVELOPMENT_USED','PROVENANCE_UNRESOLVED'],
            'architecture_relation':'exact directed typed graph isomorphism against confirmed historical development graphs',
            'unknown_history':'unresolved, never interpreted unused'},
        'forbidden':['B performance','C performance','node test scoring','KG','new GP optimization','new kernels/scans','C training','additional GNN seeds/tasks']}


def preflight(root,repo,prompt,acceptance):
    if root.exists() and any(root.iterdir()):raise ValueError('Require a new empty run directory')
    from analyse.structured_kernel_development_replay.data import verify as verify_replay
    old=verify_replay(repo/REPLAY,repo)
    summary=read_json(repo/REPLAY/'summary.json')
    c=summary['counts']
    assert (c['trajectories'],c['fits'],c['starts'])==(20,420,1260)
    assert abs(summary['n71_mean_paired_gain_pp']-.6)<1e-12 and summary['positive_n71_seeds']==6
    assert summary['pool_optimum_ids']==[CHAMPION] and abs(summary['pool_optimum_validation_mean']-.8153333333333332)<1e-15
    assert summary['state']=='DEVELOPMENT_REPLAY_NO_STABLE_GAIN' and read_json(repo/BLOCKED/'status.json')['state']=='BLOCKED'
    public=read_json(repo/REPLAY/'public_features.json');assert len(public['ids'])==179 and len(set(public['architecture_ids']))==96
    mapping=selected_mapping(read_json(repo/REPLAY/'returned_candidates.json'))
    initial=read_json(repo/REPLAY/'initial_observation_manifest.json')
    import xml.etree.ElementTree as ET
    tests=ET.parse(acceptance/'pytest.xml').getroot()
    suites=list(tests.iter('testsuite'))
    if not suites or sum(int(s.attrib.get('tests',0)) for s in suites)<6 or any(int(s.attrib.get(k,0)) for s in suites for k in ('failures','errors','skipped')):raise ValueError('Synthetic acceptance incomplete or failed')
    root.mkdir(parents=True)
    shutil.copytree(acceptance,root/'acceptance')
    shutil.copyfile(prompt,root/'PROMPT.txt')
    write_json(root/'protocol.json',protocol());write_json(root/'selected_mapping.json',mapping)
    write_json(root/'random_query_manifest.json',random_manifest(public['ids'],initial))
    tasks=[{'task_id':f'{cid}_seed{s}','candidate_id':cid,'seed':s,'configuration':mapping['configurations'][cid]} for cid in CONFIG_IDS for s in (3,4,5)]
    write_json(root/'fresh_training_manifest.json',{'tasks':tasks,'formal_tasks':15,'unique_configurations':5})
    from analyse.structured_kernel_phase1.runner import capture_baseline
    write_json(root/'baseline.json',capture_baseline(repo))
    origin=repo/ORIGIN
    dependencies=read_json(origin/'source_dependency_manifest.json')['files']
    dependency_paths=[]
    for row in dependencies:
        p=repo/row['relative_path']
        if p.suffix=='.py' and p.exists():
            if sha256(p)!=row['sha256']:raise ValueError('Historical training dependency changed: '+str(p))
            dependency_paths.append(p)
    paths=[repo/REPLAY/name for name in ('README_RESULTS.md','protocol.json','source_manifest.json','summary.json','paired_comparisons.csv','curves.csv','trajectories.csv','returned_candidates.json','initial_observation_manifest.json','acquisition_audit.md','actual_commands.md','public_features.json','run_identity.json')]
    paths+=[repo/BLOCKED/'status.json',repo/PHASE1/'frozen_manifest.json',repo/PREP/'candidates.json',repo/PREP/'data_use.json',repo/PREP/'provenance.json',repo/'results/posthoc/fuzzy-metric-stage2-20260910-v1/partition.json']
    paths += [origin/n for n in ('frozen_candidate_configs.json','selected_union_tasks.json','preregistration.json','source_dependency_manifest.json','pre_label_immutable_manifest.json','stabilization/stabilization_tasks.json','stabilization/stabilization_immutable_manifest.json')]
    paths += list(origin.glob('phase_*/**/progress.json'))+dependency_paths
    paths += [repo/p for p in old['code']]+[repo/p for p in old['sources'] if p.startswith('analyse/structured_kernel_phase1/')]
    model_sources=[]
    for seed,method,n in models():
        base=repo/REPLAY/'trajectories'/f'{seed}_{method}';folder=base/'fits'/f'n{n:03d}'
        fit=read_json(folder/'fit.json');model=folder/fit['selected_model']
        if sha256(model)!=fit['selected_model_sha256']:raise ValueError('Historical selected model changed')
        paths += [model,folder/'fit.json',folder/'training.json',base/'decisions'/f'{n-49:02d}.json',base/'observations.json',base/'complete.json']
        model_sources.append({'seed':seed,'method':method,'n':n,'checkpoint':str(model.relative_to(repo)),'sha256':sha256(model)})
    write_json(root/'M1_model_manifest.json',model_sources)
    # Dataset bytes participate in the freeze without decoding any labels here.
    dataset_paths=[p for p in (repo/'Cora').rglob('*') if p.is_file() and ('raw' in p.parts or 'processed' in p.parts) and p.suffix not in ('.py','.pyc')]
    if not dataset_paths:raise ValueError('Existing Cora data absent; no automatic download')
    paths+=dataset_paths
    sources={str(p.relative_to(repo)):sha256(p) for p in sorted(set(paths))}
    write_json(root/'source_manifest.json',{'files':sources,'replay_identity':old['run_identity'],
        'A_only_labels':'pinned read_A_labels; no mixed cache','B_performance_reads':0,'C_performance_reads':0,
        'data_file_hashing':'opaque bytes only during preflight; M3 loader boundary separately recorded'})
    write_json(root/'training_core_audit.json',{'historical_evaluator_sha256':sha256(repo/'eval_utils.py'),
        'original_training_loop_lines':[861,1061],'optimizer':'Adam','scheduler':'CosineAnnealingLR',
        'best_checkpoint_originally_restored':False,'strict_improvement':True,'min_delta':0,'no_test_branch':True})
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    write_json(root/'resource_preflight.json',{'gpu_inventory':gpu.stdout,'returncode':gpu.returncode,'worker':1,'selected_device':'cuda:0 if available else CPU','no_extra_benchmark':True,'python':sys.executable})
    for p in (repo/'analyse/structured_kernel_followup').glob('*.py'):ast.parse(p.read_text())
    subprocess.run(['git','diff','--check'],cwd=repo,check=True)
    inputs={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
    frozen={'code':code_manifest(repo),'sources':sources,'inputs':inputs,'created_at':now()};frozen['identity']=digest(frozen)
    write_json(root/'run_identity.json',frozen)
    write_json(root/'module_status.json',{'identity':frozen['identity'],'independent_confirmation':False,'modules':{m:{'state':'PENDING'} for m in ('M1','M2','M3','M4')}})
    print(json.dumps({'preflight':'PASS','run_identity':frozen['identity'],'root':str(root)}),flush=True)
