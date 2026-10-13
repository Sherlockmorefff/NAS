import ast
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .common import *
from .protocol import protocol
from analyse.structured_kernel_phase1.data import SOURCE,A_ID_HASH,A_LABEL_HASH,RECIPE_HASH,PUBLIC_HASHES,VAE,VAE_HASH
from analyse.structured_kernel_phase1.features import build_operator_graph,directed_wl,graph_hash

C_CONFIG_SOURCES=[ORIGIN+'/frozen_candidate_configs.json',
 'results/posthoc/original-head-deterministic-batch-pilot-global4-v1/frozen_candidate_pools.json',
 'results/posthoc/original-head-confirmatory-attribution-global4-v2/frozen_candidate_pools.json',PREP+'/candidates.json']
C_ID_SOURCES=[ORIGIN+'/exposure_registry.json',ORIGIN+'/exposure_source_inventory.json',
 'results/posthoc/h-vs-head-model-blind-prospective-global4-v1/phase_a/exposure_registry.json',
 'results/posthoc/h-vs-head-model-blind-prospective-global4-v1/phase_a/exposure_source_inventory.json',FOLLOWUP+'/historical_use_registry.json',
 ORIGIN+'/candidate_generation_audit.json',ORIGIN+'/preregistration.json']


def audit_A(root,repo):
    from analyse.structured_kernel_development_replay.data import verify as old_verify,initial_ids
    from analyse.structured_kernel_development_replay.learner import validate_complete
    frozen=old_verify(repo/REPLAY,repo)
    public=read_json(repo/REPLAY/'public_features.json');ids=public['ids']
    assert len(ids)==179 and digest(sorted(ids))==A_ID_HASH and len(set(public['architecture_ids']))==96
    configs=read_json(repo/REPLAY/'public_configurations.json')
    graphs={c['architecture_fingerprint']:build_operator_graph(c) for c in configs.values()}
    vectors,feature=directed_wl(graphs)
    for cid,aid,x,v in zip(ids,public['architecture_ids'],public['x_raw'],public['wl']):
        c=configs[cid];assert np.array_equal(vectors[aid],v)
        assert np.allclose([np.log10(c['lr']),c['dropout'],np.log2(c['hidden_dim']),np.log10(c['l2'])],x[12:],atol=1e-14,rtol=0)
        assert graph_hash(graphs[aid])==public['graph_hashes'][aid]
    for name,h in PUBLIC_HASHES.items():
        if sha256(repo/SOURCE/name)!=h:raise ValueError('A recipe/config/features changed: '+name)
    if not (repo/SOURCE/'private/evaluation_A.json').is_file():raise ValueError('A-only cache missing')
    summary=read_json(repo/REPLAY/'summary.json');assert summary['counts']['fits']==420 and summary['counts']['starts']==1260
    assert summary['state']=='DEVELOPMENT_REPLAY_NO_STABLE_GAIN' and abs(summary['n71_mean_paired_gain_pp']-.6)<1e-12
    followup=read_json(repo/FOLLOWUP/'summary.json');assert followup['counts']['M1_verified_states']==86 and followup['counts']['M3_valid_tasks']==15
    assert abs(followup['modules']['M2']['comparisons']['WI_minus_Random_n71']['mean_pp']-.76)<1e-12
    initial=read_json(repo/REPLAY/'initial_observation_manifest.json')
    for seed in SEEDS:assert initial[str(seed)]['ids']==initial_ids(ids,seed)
    paths=[repo/REPLAY/x for x in ('public_features.json','public_configurations.json','initial_observation_manifest.json','README_RESULTS.md','protocol.json','source_manifest.json','run_identity.json','summary.json','paired_comparisons.csv','returned_candidates.json','curves.csv')]
    paths += [repo/FOLLOWUP/x for x in ('README_RESULTS.md','protocol.json','source_manifest.json','run_identity.json','summary.json','M2_summary.json','M2_completion.json','random_trajectories.csv','random_curves.csv','random_paired_comparisons.csv')]
    paths += [repo/PHASE1/x for x in ('README_RESULTS.md','protocol.json','source_manifest.json','frozen_manifest.json','feature_manifest.json')]
    paths += [repo/SOURCE/n for n in PUBLIC_HASHES]+[repo/BLOCKED/'status.json']
    diagnostic_gaps=[]
    for s in SEEDS:
        for method in ('W-I','RAW16'):
            folder=repo/REPLAY/'trajectories'/f'{s}_{method}'
            complete=read_json(folder/'complete.json')
            assert (complete['fits'],complete['starts'],complete['observations'])==(21,63,71)
            for key in ('observations.json',):
                if sha256(folder/key)!=complete['files'][key]:raise ValueError('Historical observed comparison source changed')
            paths+=[folder/x for x in ('complete.json','observations.json')]
            for n in range(50,71):
                f=folder/'fits'/f'n{n:03d}'
                if not (f/'fit.json').exists():
                    diagnostic_gaps.append(str(f/'fit.json'));continue
                fit=read_json(f/'fit.json')
                for path in [f/'fit.json',f/'training.json',f/fit['selected_model'],folder/'decisions'/f'{n-49:02d}.json']:
                    if path.exists():paths.append(path)
                    else:diagnostic_gaps.append(str(path))
    for rel,h in read_json(repo/FOLLOWUP/'M2_completion.json')['files'].items():
        if sha256(repo/FOLLOWUP/rel)!=h:raise ValueError('Random reference changed')
        paths.append(repo/FOLLOWUP/rel)
    paths += [repo/p for p in frozen['code']]+[repo/p for p in frozen['sources']]
    for sub in ('analyse/structured_kernel_phase1','analyse/structured_kernel_followup'):paths += list((repo/sub).glob('*.py'))
    for name,value in [('public_features.json',public),('public_configurations.json',configs),('initial_observation_manifest.json',initial),('A_graphs.json',graphs),('A_feature_manifest.json',read_json(repo/PHASE1/'feature_manifest.json'))]:write_json(root/name,value)
    write_json(root/'historical_posterior_availability.json',{'missing':diagnostic_gaps,'missing_old_models_only_limit_diagnostics':True})
    return {str(p.relative_to(repo)):sha256(p) for p in sorted(set(paths))}


def audit_C(root,repo):
    if sha256(repo/VAE)!=VAE_HASH:raise ValueError('Frozen VAE identity mismatch')
    from .c_features import ExactIndex
    history={};sources={};fingerprints=set()
    for rel in C_CONFIG_SOURCES:
        p=repo/rel;rows=read_json(p);sources[rel]=sha256(p)
        for c in rows:
            aid=c['architecture_fingerprint'];g=build_operator_graph(c)
            key=digest(g)
            history.setdefault(key,{'graph':g,'architecture_fingerprints':set(),'sources':set()})
            history[key]['architecture_fingerprints'].add(aid);history[key]['sources'].add(rel);fingerprints.add(aid)
    registries={}
    for rel in C_ID_SOURCES:
        sources[rel]=sha256(repo/rel)
        d=read_json(repo/rel)
        if rel.endswith('/exposure_registry.json'):
            ids={r['architecture_fingerprint'] for r in d};registries[rel]=len(ids);fingerprints|=ids
    represented={a for r in history.values() for a in r['architecture_fingerprints']};missing=sorted(fingerprints-represented)
    history={k:{'graph':v['graph'],'architecture_fingerprints':sorted(v['architecture_fingerprints']),'sources':sorted(v['sources'])} for k,v in history.items()}
    audit={'known_fingerprint_count':len(fingerprints),'full_graph_identity_count':len(represented),'ordered_typed_graph_count':len(history),
        'fingerprints_without_full_label_free_graphs':missing,'provenance_resolved_for_independent_C':not missing,
        'registry_sources':registries,'scope':'available label-free configurations plus pre-existing metadata; no mixed histories or label caches opened',
        'history_gap':'legacy histories/posthoc truth outputs are fingerprint-only in safe metadata; exact graph equivalence and full subsequent-use history cannot be certified',
        'unlabeled_exposure':'not itself evidence of performance leakage; conservative exclusion is not a claim all exposed graphs were performance-used',
        'original192_is_exhaustive':False,'evidence':ORIGIN+'/candidate_generation_audit.json',
        'reason':'16384 normal latent draws yielded261 fresh unique before selecting192; not exhaustive enumeration',
        'source_hashes':sources}
    cdir=root/'C_preparation';write_json(cdir/'historical_graphs.json',history);write_json(cdir/'historical_use_registry.json',audit)
    write_json(cdir/'historical_fingerprints.json',sorted(fingerprints))
    paths=[repo/'architecture_diagnostics.py',repo/'nas_space.py',repo/'models.py',repo/'hp_modes.py',repo/'eval_utils.py',repo/VAE,
        repo/'analyse/high_regime_prior_model_blind_validation.py',repo/'analyse/original_head_deterministic_batch_pilot.py',
        repo/'analyse/structure_native_surrogate_pilot.py',repo/'analyse/performance_aware_vae_strict_regression.py']
    # Freeze the already-audited local Python import closure as source identity.
    dependencies=read_json(repo/ORIGIN/'source_dependency_manifest.json')['files']
    paths += [repo/r['relative_path'] for r in dependencies if r['relative_path'].endswith('.py') and (repo/r['relative_path']).is_file()]
    sources.update({str(p.relative_to(repo)):sha256(p) for p in set(paths)})
    return sources,audit


def preflight(root,repo,prompt,acceptance):
    if root.exists() and any(root.iterdir()):raise ValueError('New empty run required')
    suites=list(ET.parse(acceptance/'pytest.xml').getroot().iter('testsuite'))
    if not suites or sum(int(s.attrib.get('tests',0)) for s in suites)<8 or any(int(s.attrib.get(k,0)) for s in suites for k in ('failures','errors','skipped')):raise ValueError('Synthetic acceptance not passed')
    root.mkdir(parents=True);shutil.copyfile(prompt,root/'PROMPT.txt');shutil.copytree(acceptance,root/'acceptance')
    write_json(root/'protocol.json',protocol())
    from analyse.structured_kernel_phase1.runner import capture_baseline
    write_json(root/'baseline.json',capture_baseline(repo))
    sources={'A':{},'C':{}};states={};errors={}
    try:
        sources['A']=audit_A(root,repo);states['A']={'state':'PENDING'};states['analysis']={'state':'PENDING'}
    except Exception as exc:
        errors['A']=repr(exc);states['A']={'state':'BLOCKED','reason':repr(exc)};states['analysis']={'state':'BLOCKED','reason':'A identity unavailable'}
    try:
        sources['C'],audit=audit_C(root,repo);states['C']={'state':'PENDING','historical_graph_gaps':len(audit['fingerprints_without_full_label_free_graphs'])}
    except Exception as exc:errors['C']=repr(exc);states['C']={'state':'BLOCKED_C','reason':repr(exc)}
    write_json(root/'source_manifest.json',{'sources':sources,'A_expected_label_sha256':A_LABEL_HASH,'recipe_identity':RECIPE_HASH,
        'label_entrypoint':'trusted read_A_labels only','B_performance_SE_reads':0,'C_performance_reads':0,'preflight_errors':errors})
    import torch,gpytorch,botorch
    write_json(root/'environment.json',{'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,
        'gpytorch':gpytorch.__version__,'botorch':botorch.__version__,'numpy':np.__version__,'A_device':'cpu','A_dtype':'float64','C_encoder_dtype':'float32','threads':1,'workers':1})
    for p in (repo/'analyse/structured_kernel_arch_main_ablation').glob('*.py'):ast.parse(p.read_text())
    subprocess.run(['git','diff','--check'],cwd=repo,check=True)
    frozen={'code':code_files(repo),'sources':sources,'inputs':{str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file()},'created_at':now()}
    frozen['identity']=digest(frozen);write_json(root/'run_identity.json',frozen)
    write_json(root/'module_status.json',{'run_identity':frozen['identity'],'independent_confirmation':False,'modules':states})
    print(json.dumps({'preflight':'PASS' if not errors else 'PARTIAL','identity':frozen['identity'],'modules':states,'errors':errors}),flush=True)
