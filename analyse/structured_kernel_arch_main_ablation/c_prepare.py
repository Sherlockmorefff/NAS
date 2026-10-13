"""Frozen-decoder sampling and feature preparation only; no surrogate/GNN scores."""
from collections import Counter
from pathlib import Path
import json
import sqlite3
import sys
import time
import numpy as np
import torch
from .common import *
from .c_features import ExactIndex,extend_dictionary
from analyse.structured_kernel_phase1.features import build_operator_graph,graph_hash
from analyse.structured_kernel_phase1.data import VAE,VAE_HASH


def guard(root,repo):
    root=root.resolve();repo=repo.resolve()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0]).resolve()
        if path.is_relative_to(repo/'results') and not path.is_relative_to(root):raise PermissionError('C preparation cannot open historical performance sources')
        if path.is_relative_to(root):
            rel=path.relative_to(root)
            if 'trajectories' in rel.parts or 'analysis' in rel.parts or path.name.startswith('oracle'):raise PermissionError('C preparation cannot read A performance outputs')
    sys.addaudithook(audit)


def run(root,repo):
    folder=root/'C_preparation'
    if (folder/'summary.json').exists():return read_json(folder/'summary.json')
    verify(root,repo,'C');status(root,'C','RUNNING');guard(root,repo)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.manual_seed(2026100601);np.random.seed(2026100601)
    from architecture_diagnostics import load_joint_vae,architecture_fingerprint,encode_architecture_mu
    from analyse.original_head_deterministic_batch_pilot import validate_static_config,config_from_igraph,HP_REGIMES,hp_fingerprint
    from analyse.structure_native_surrogate_pilot import realize_global4_hp
    checkpoint=repo/VAE
    if sha256(checkpoint)!=VAE_HASH:raise ValueError('Frozen encoder identity changed')
    model=load_joint_vae(checkpoint,'global4',device='cpu');model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    history=read_json(folder/'historical_graphs.json');historical=ExactIndex((k,v['graph']) for k,v in history.items())
    fps=set(read_json(folder/'historical_fingerprints.json'));registry=read_json(folder/'historical_use_registry.json')
    provenance=registry['provenance_resolved_for_independent_C']
    latent_path=folder/'latent_draw_manifest.json'
    if not latent_path.exists():
        generator=torch.Generator(device='cpu');generator.manual_seed(2026100601)
        # Draw generation is not a decoder call. Unused draws have no generated
        # graph and no score; each decoder call below is individually persisted.
        draws=[torch.randn(1,12,generator=generator).flatten().tolist() for _ in range(5000)]
        seal(latent_path,{'seed':2026100601,'distribution':'N(0,I12)','dtype':'float32','order':'sequential torch.randn(1,12)','latent_vectors':draws,'identity':digest(draws)})
    latent_manifest=read_json(latent_path);latents=latent_manifest['latent_vectors']
    if digest(latents)!=latent_manifest['identity']:raise ValueError('Latent draw manifest changed')
    db=sqlite3.connect(folder/'generation.sqlite');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY,draw_index INTEGER,status TEXT,payload TEXT,started_at TEXT,completed_at TEXT)');db.commit()
    done={int(i):json.loads(p) for i,p in db.execute('SELECT draw_index,payload FROM calls WHERE status="COMPLETE" ORDER BY id')}
    kept=[];index=ExactIndex();physical_replays=0
    for i,row in sorted(done.items()):
        if row['result']=='PROVISIONAL_NEW' or row['result']=='ELIGIBLE_NEW':
            kept.append(row['architecture']);index.add(row['architecture']['architecture_fingerprint'],row['architecture']['graph'])
    started=time.perf_counter()
    for i,z in enumerate(latents):
        if len(kept)>=45:break
        if i in done:continue
        if db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]>=5000:break
        replay=db.execute('SELECT COUNT(*) FROM calls WHERE draw_index=?',(i,)).fetchone()[0]>0
        with db:cur=db.execute('INSERT INTO calls(draw_index,status,started_at) VALUES(?,?,?)',(i,'RUNNING',now()));call=cur.lastrowid
        record={'draw_index':i,'latent':z,'physical_replay':replay}
        with torch.no_grad():graph=model.arch_vae.decode(torch.tensor([z],dtype=torch.float32),stochastic=False)[0]
        try:
            config=config_from_igraph(model,graph);valid,reason=validate_static_config(config)
            if not valid:record.update(result='INVALID',reason=reason)
            else:
                aid=architecture_fingerprint(config);typed=build_operator_graph(config)
                record['architecture_fingerprint']=aid
                # Fingerprint rejection is conservative; every asserted graph
                # equivalence uses an explicit typed directed isomorphism test.
                matches=historical.match(typed);duplicates=index.match(typed)
                if matches:record.update(result='HISTORICAL_EXACT_ISOMORPHISM',matches=matches)
                elif aid in fps:record.update(result='HISTORICAL_FINGERPRINT_ONLY',reason='Full historical graph missing; reject conservatively, do not assert exact equivalence')
                elif duplicates:record.update(result='DUPLICATE_EXACT_ISOMORPHISM',matches=duplicates)
                else:
                    architecture={**config,'architecture_fingerprint':aid,'generation_index':i,'architecture_index':len(kept),
                        'latent_draw':z,'graph':typed,'graph_hash':graph_hash(typed),'source_checkpoint_sha256':VAE_HASH,
                        'classification':'ELIGIBLE_WITH_EVIDENCE' if provenance else 'PROVENANCE_UNRESOLVED',
                        'exact_isomorphism_exclusion':'passed against available full history graphs; fingerprint-only gaps remain' if not provenance else 'passed'}
                    record.update(result='ELIGIBLE_NEW' if provenance else 'PROVISIONAL_NEW',architecture=architecture)
                    kept.append(architecture);index.add(aid,typed)
        except (TypeError,ValueError,KeyError,IndexError) as exc:record.update(result='INVALID',reason=repr(exc))
        with db:db.execute('UPDATE calls SET status=?,payload=?,completed_at=? WHERE id=?',('COMPLETE',json.dumps(record),now(),call))
        done[i]=record
        if (i+1)%100==0 or len(kept)==45:
            write_json(folder/'progress.json',{'draws_completed':len(done),'kept_provisional_or_eligible_architectures':len(kept),'physical_decoder_calls_started':db.execute('SELECT COUNT(*) FROM calls').fetchone()[0],'elapsed_seconds':time.perf_counter()-started,'updated_at':now()})
    counts=Counter(r['result'] for r in done.values());calls=list(db.execute('SELECT id,draw_index,status,payload,started_at,completed_at FROM calls ORDER BY id'))
    with (folder/'C_generation_log.jsonl').open('w') as out:
        for call,i,state,payload,start,end in calls:
            out.write(json.dumps({'call_id':call,'draw_index':i,'status':state,'started_at':start,'completed_at':end,'outcome':json.loads(payload) if payload else None})+'\n')
    db.close();configs=[];hp_errors=[]
    for architecture in kept:
        ai=architecture['architecture_index'];rng=np.random.default_rng(np.random.SeedSequence([2026100601,ai,1]));used=set()
        base={k:v for k,v in architecture.items() if k not in ('graph','latent_draw')}
        encoded=encode_architecture_mu(model,architecture)
        if len(encoded)!=12 or not np.isfinite(encoded).all():raise ValueError('Invalid frozen encoder mean')
        for regime,(low,high) in HP_REGIMES.items():
            for attempt in range(100):
                raw=rng.uniform(low,high,4).astype(np.float64).tolist();hp=realize_global4_hp({**base,'raw_hp_vector':raw})
                key=tuple(hp[k] for k in ('lr','dropout','hidden_dim','l2'))
                if key not in used:break
            else:hp_errors.append({'architecture_id':base['architecture_fingerprint'],'regime':regime,'reason':'100 repeated realized HP combinations'});continue
            used.add(key);hfp=hp_fingerprint(raw);cid=digest({'architecture_fingerprint':base['architecture_fingerprint'],'hp_fingerprint':hfp})
            configs.append({**base,**hp,'candidate_id':cid,'joint_config_fingerprint':cid,'hp_fingerprint':hfp,'raw_hp_vector':raw,'hp_regime':regime,
                'HP4':[float(np.log10(hp['lr'])),hp['dropout'],float(np.log2(hp['hidden_dim'])),float(np.log10(hp['l2']))],
                'latent12':encoded,'encoder_is_actual_graph_mean':True,'generation_latent_is_not_encoding':True,
                'hp_mode':'global4','normalization':'none','gcnii_alpha':.1,'gcnii_theta':.5,'hp_duplicate_redraws':attempt})
    assert len(configs)<=180 and len({c['candidate_id'] for c in configs})==len(configs)
    seal(folder/'C_unlabeled_manifest.json',{'architectures':kept,'configurations':configs,'configuration_count':len(configs),'architecture_count':len(kept),
        'labels_or_scores_present':False,'eligible_for_independent_confirmation':provenance and not hp_errors,
        'generation_distribution':'original standard-normal latent deterministic greedy decoder','no_performance_filtering':True,'frozen_at':read_json(root/'run_identity.json')['created_at']})
    write_json(folder/'C_eligibility_manifest.json',{'candidates':[{'candidate_id':c['candidate_id'],'architecture_id':c['architecture_fingerprint'],
        'classification':c['classification'],'reason':'Historical fingerprint-only entries and later-use coverage gaps prevent a proof of independent architecture membership' if not provenance else 'Verified fresh exact graph against fully resolved historical graph sources',
        'sources':list(registry['source_hashes']),'no_B_C_performance_access':True} for c in configs],
        'eligible_configuration_count':len(configs) if provenance else 0,'provisional_configuration_count':0 if provenance else len(configs),'historical_gap_count':len(registry['fingerprints_without_full_label_free_graphs'])})
    encoding='MISSING_NO_NEW_GRAPHS';encoding_error=None
    if kept:
        try:
            ag=read_json(root/'A_graphs.json');public=read_json(root/'public_features.json');oldf={a:v for a,v in zip(public['architecture_ids'],public['wl'])}
            cg={a['architecture_fingerprint']:a['graph'] for a in kept};features,manifest=extend_dictionary(ag,cg,read_json(root/'A_feature_manifest.json'),oldf)
            write_json(folder/'extended_feature_manifest.json',manifest)
            write_json(folder/'extended_A_features.json',{a:features[a] for a in ag})
            write_json(folder/'C_features.json',{a:features[a] for a in cg})
            write_json(folder/'frozen_dictionary.json',manifest['dictionaries']);encoding='COMPLETE'
        except (ValueError,KeyError,OSError) as exc:encoding='ENCODING_REVIEW_REQUIRED';encoding_error=repr(exc)
    if sha256(checkpoint)!=VAE_HASH:raise ValueError('Frozen VAE modified')
    scientific='READY_UNLABELED' if provenance and len(configs)==180 and encoding=='COMPLETE' and not hp_errors else ('PARTIAL_UNLABELED' if provenance and configs and encoding=='COMPLETE' else 'BLOCKED_C')
    result={'state':'COMPLETE','scientific_state':scientific,'decoder_calls_started':len(calls),'completed_draws':len(done),
        'physical_repeated_calls':len(calls)-len({c[1] for c in calls}),'draw_outcomes':dict(counts),'generated_architectures':len(kept),'generated_configurations':len(configs),
        'eligible_architectures':len(kept) if provenance else 0,'eligible_configurations':len(configs) if provenance else 0,
        'provenance_unresolved':not provenance,'missing_history_full_graphs':len(registry['fingerprints_without_full_label_free_graphs']),
        'encoding_status':encoding,'encoding_error':encoding_error,'hp_errors':hp_errors,
        'generation_stop_reason':'45 provisional graphs retained; source gap cannot be repaired by more sampling' if len(kept)==45 and not provenance else ('target45' if len(kept)==45 else '5000-call bound'),
        'C_performance_reads':0,'C_performance_predictions':0,'C_acquisition_scores':0,'VAE_updates':0,'new_GNN_training':0,
        'whole_legal_space_exhausted':False,'elapsed_seconds':time.perf_counter()-started}
    write_json(folder/'summary.json',result);status(root,'C',scientific,**{k:v for k,v in result.items() if k not in ('state','scientific_state')})
    return result
