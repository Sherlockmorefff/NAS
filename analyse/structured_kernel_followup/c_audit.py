"""Configuration and historical-use metadata only; no performance loaders."""
from collections import Counter
from .common import *
from analyse.structured_kernel_phase1.features import build_operator_graph,graph_hash,isomorphic_operator_graph


def run(root,repo):
    if (root/'M4_summary.json').exists() and read_json(root/'M4_summary.json')['state']=='COMPLETE':return read_json(root/'M4_summary.json')
    set_status(root,'M4','RUNNING')
    origin=read_json(repo/ORIGIN/'frozen_candidate_configs.json')
    prepared=read_json(repo/PREP/'candidates.json')
    tasks=read_json(repo/ORIGIN/'selected_union_tasks.json')
    stabilization=read_json(repo/ORIGIN/'stabilization/stabilization_tasks.json')
    assert len(origin)==768
    used={c['candidate_id']:c for c in prepared}
    selected={t['joint_config_fingerprint'] for t in tasks}
    if set(used)!=selected:raise ValueError('Historical development membership differs from selected tasks')
    assert len(used)==336
    development={c['architecture_fingerprint']:build_operator_graph(c) for c in prepared}
    if len(development)!=192:raise ValueError('Historical architecture membership changed')
    graphs={r['architecture_fingerprint']:build_operator_graph(r) for r in origin}
    relations={}
    for aid,g in graphs.items():
        # Fast hash grouping is only a prefilter. Every claimed relationship
        # is checked with exact directed typed graph isomorphism.
        matches=[a for a,h in development.items() if graph_hash(h)==graph_hash(g) and isomorphic_operator_graph(g,h)]
        if not matches:matches=[a for a,h in development.items() if isomorphic_operator_graph(g,h)]
        relations[aid]=sorted(matches)
    entries=[]
    for c in origin:
        cid=c['joint_config_fingerprint'];aid=c['architecture_fingerprint'];direct=cid in used
        entries.append({'candidate_id':cid,'config_id':c['config_id'],'architecture_id':aid,
            'classification':'DEVELOPMENT_USED' if direct else 'PROVENANCE_UNRESOLVED',
            'known_performance_development_use':direct,'unlabeled_generation_exposure':True,
            'architecture_matches_historical_development':relations[aid],
            'eligible_for_unknown_architecture_confirmation':False if relations[aid] else None,
            'configuration_complete':all(k in c for k in ('operations','edges','lr','dropout','hidden_dim','l2','gat_heads','sage_aggr','gin_eps')),
            'reason':'A/B historical development metadata explicitly covers this configuration' if direct else
                'Same exact typed architecture already used in development; configuration-level full historical performance use remains unresolved',
            'evidence':[ORIGIN+'/frozen_candidate_configs.json',ORIGIN+'/selected_union_tasks.json',PREP+'/candidates.json',PREP+'/data_use.json']})
    source=read_json(root/'source_manifest.json')['files']
    registry={'confirmed_use':[{'source':ORIGIN,'type':'performance development','members':sorted(selected),
        'architecture_count':192,'evidence':['selected_union_tasks.json','phase_b/initial/gpu0/progress.json','phase_b/initial/gpu1/progress.json',PREP+'/data_use.json']},
        {'source':PHASE1,'type':'A labels used for method selection','members':read_json(repo/REPLAY/'public_features.json')['ids']},
        {'source':REPLAY,'type':'A cached development search evaluation','members':read_json(repo/REPLAY/'public_features.json')['ids']}],
        'unlabeled_exposure':{'original768_generation':'frozen VAE decoding and predetermined HP; this alone is not performance leakage',
            'VAE_pretraining_membership':'historical exact members unresolved, unsupervised exposure not automatically performance leakage'},
        'historical_directories_inventory':[str(p.relative_to(repo)) for parent in ('results/posthoc','results/search') for p in sorted((repo/parent).iterdir()) if p.is_dir()],
        'evidence_hashes':{p:h for p,h in source.items() if p.startswith(ORIGIN) or p.startswith(PREP)},
        'gaps':['No exhaustive label access/use ledger for all remaining432 configurations across every historical experiment',
                'Mixed seed_results and performance caches deliberately not opened',
                'Unknown-architecture eligibility is already excluded by confirmed development architecture matches; no performance reveal needed'],
        'scope_note':'The verified historical subset suffices to exclude every original768 architecture. The directory inventory is not a claim of exhaustive configuration-level history.'}
    collisions=[{'candidate_id':t['joint_config_fingerprint'],'seed':t['seed'],'task_key':t['task_key'],'assigned_gpu':t['assigned_gpu']} for t in stabilization if t['joint_config_fingerprint'] in CONFIG_IDS and t['seed'] in (3,4,5)]
    progress=[]
    for p in sorted((repo/ORIGIN/'phase_c/stabilization').glob('gpu*/progress.json')):
        x=read_json(p);progress.append({'path':str(p.relative_to(repo)),'sha256':sha256(p),**{k:x[k] for k in ('completed','expected','remaining','terminal_status_counts')}})
    collision={'same_config_seed_tasks':collisions,'count':len(collisions),'completion_metadata':progress,
        'interpretation':'12 prescribed tasks were previously completed in historical stabilization; this run is real retraining, not previously unseen independent measurements. Historical scores not read.'}
    write_json(root/'historical_same_seed_audit.json',collision)
    write_json(root/'historical_use_registry.json',registry)
    write_json(root/'C_eligibility_manifest.json',{'candidates':entries,'architecture_relations':relations,'performance_values_read':0})
    summary={'state':'COMPLETE','scientific_state':'NO_ELIGIBLE_POOL','candidate_count':768,'architecture_count':len(graphs),
        'classifications':dict(Counter(e['classification'] for e in entries)),
        'eligible_candidates_with_evidence':0,'eligible_unknown_architectures':0,'all192_architectures_overlap_development':all(relations.values()),
        'configuration_completeness':sum(e['configuration_complete'] for e in entries),'performance_reads':0,'new_training':0,
        'historical_same_config_seed_tasks':len(collisions)}
    write_json(root/'M4_summary.json',summary);set_status(root,'M4','COMPLETE',scientific_state='NO_ELIGIBLE_POOL',eligible_candidates=0)
    return summary
