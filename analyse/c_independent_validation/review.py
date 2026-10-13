"""P1 qualification from actual recorded members and bound conservative pools."""
import csv
from collections import Counter,defaultdict
from .common import *
from analyse.c_provenance_evidence_closure.projector import objects
from analyse.c_provenance_evidence_closure.collect import valid_structure
from analyse.c_independence_minimal_audit.identity import fingerprint,config_of

IDENTS=('architecture_fingerprint','architecture_group_id','candidate_id','candidate_fingerprint','joint_config_fingerprint','config_id','configuration_id')
LISTS={'config_ids':'config_id','train_config_ids':'config_id','validation_config_ids':'config_id','source_config_ids':'config_id',
 'candidate_ids':'candidate_id','configuration_ids':'configuration_id','architecture_fingerprints':'architecture_fingerprint',
 'train_architecture_fingerprints':'architecture_fingerprint','validation_architecture_fingerprints':'architecture_fingerprint'}

def decision(cert,orbit):
 if not cert.get('complete_membership') or not cert.get('binding_verified'):return 'PROVENANCE_UNRESOLVED'
 if set(cert['structure_ids'])&set(orbit):return 'HISTORICAL_OVERLAP_POSSIBLE'
 return 'PASS_BOUND_COMPLETE_MEMBERS'

def refs(o):
 return [(k,o[k]) for k in IDENTS if isinstance(o.get(k),str)]

def make_catalogue(repo,root,values):
 lookup=defaultdict(set);basis=defaultdict(set);old=repo/PREVIOUS
 for rel,entry in read(old/'projection_manifest.json').items():
  p=old/entry['path'];assert sha(p)==entry['output_sha256'];s=read(p).get('identity_summary',{})
  for k,vs in s.get('stored_to_concrete',{}).items():
   lookup[('architecture_fingerprint',k)].update(vs);basis[('architecture_fingerprint',k)].add(rel)
 for source,gs in read(repo/MINIMAL/'source_full_graphs.json').items():
  for aid,g in gs.items():
   lookup[('architecture_fingerprint',aid)].add(fingerprint(config_of(g)));basis[('architecture_fingerprint',aid)].add(source)
 for rel,value in values.items():
  for ptr,o in objects(value):
   if valid_structure(o):
    for key in refs(o):lookup[key].add(fingerprint(o));basis[key].add(rel+'#'+ptr)
 # Co-occurring explicitly typed identities provide alias links. Multiple
 # graph values are kept as a conservative union; never choose a convenient one.
 for _ in range(len(IDENTS)+1):
  changes=0
  for rel,value in values.items():
   for ptr,o in objects(value):
    keys=refs(o);known=set().union(*(lookup[k] for k in keys)) if keys else set()
    if known:
     for key in keys:
      before=len(lookup[key]);lookup[key].update(known)
      if len(lookup[key])!=before:basis[key].add(rel+'#'+ptr);changes+=1
  if not changes:break
 write(root/'identity_mapping_catalogue.json',[{'kind':k,'value':v,'structure_ids':sorted(h),'basis_sources':sorted(basis[(k,v)])} for (k,v),h in sorted(lookup.items()) if h])
 return lookup

def members(value,lookup):
 structures=set();unmapped=[];direct=0;member_objects=0;bad=[]
 all_objects=dict(objects(value))
 for pointer,o in all_objects.items():
  keys=refs(o);has_structure='operations' in o or 'edges' in o
  if pointer.endswith('/dataset_identity') and not keys and 'operations' not in o and type(o.get('edges')) is int:
   has_structure=False
  if pointer.endswith('/architecture_summary') and 'edges' not in o and not keys:
   parent=all_objects.get(pointer.rsplit('/',1)[0],{})
   if valid_structure(parent) and o.get('operations')==parent['operations']:has_structure=False
  if has_structure:
   if valid_structure(o):structures.add(fingerprint(o));direct+=1
   else:bad.append(pointer)
  if keys:
   member_objects+=1;mapped=[k for k in keys if lookup[k]]
   if mapped:
    for key in mapped:structures.update(lookup[key])
   elif not (has_structure and valid_structure(o)):unmapped.append({'pointer':pointer,'identities':dict(keys)})
  for field,kind in LISTS.items():
   if isinstance(o.get(field),list):
    for value in o[field]:
     if not isinstance(value,str):raise ValueError('Nonstring identity member')
     member_objects+=1
     if lookup[(kind,value)]:structures.update(lookup[(kind,value)])
     else:unmapped.append({'pointer':pointer+'/'+field,'identities':{kind:value}})
 return {'structure_ids':sorted(structures),'unmapped':unmapped,'unsupported_structure_paths':bad,'direct_structures':direct,'member_objects':member_objects}

def registry_certificate(repo,root,values,source,info,pm):
 hbase='results/posthoc/h-vs-head-model-blind-prospective-global4-v1';hreg=hbase+'/phase_a/exposure_registry.json';hinv=hbase+'/phase_a/exposure_source_inventory.json'
 # These two files are established identity-only registry/inventory material.
 original=read(repo/hreg);unmapped={r['identities']['architecture_fingerprint'] for r in info['unmapped']}
 h_by={r['architecture_fingerprint']:r for r in original}
 group_only=all(a in h_by and h_by[a]['source_groups']==['legacy_histories'] for a in unmapped)
 prior=read(repo/PREVIOUS/'development_member_use_coverage.json');certs={c['id']:c for c in read(repo/PREVIOUS/'producer_schema_certificates.json')['certificates']}
 inventory=read(repo/hinv);records=[x for x in inventory if x['source_group']=='legacy_histories'];complete=[];superset=set();checks=[]
 for row in records:
  p=row['path'].removeprefix(str(repo)+'/');cert=certs.get(prior['covered_sources'].get(p));ok=bool(cert and cert.get('complete_membership') and cert.get('input_binding_verified') and cert['source_sha256']==row['sha256'])
  checks.append({'source':p,'inventory_sha256':row['sha256'],'certificate_id':None if not cert else cert['id'],'passed':ok})
  if ok:superset.update(cert['structure_ids']);complete.append(p)
 artifact=values[hbase+'/artifact_manifest.json'];artifactmap={r['path']:r['sha256'] for r in artifact['files']}
 historical_h_binding=all(artifactmap[name]==sha(repo/hbase/name) for name in ('phase_a/exposure_registry.json','phase_a/exposure_source_inventory.json'))
 source_binding=source==hreg;reference=None
 if source!=hreg:
  inventory_path=str(Path(source).parent/'exposure_source_inventory.json');meta=read(repo/inventory_path)
  matches=[v for v in meta['inventory'] if v['path']==str(repo/hreg)]
  source_binding=len(matches)==1 and matches[0]['sha256']==sha(repo/hreg)
  reference={'path':inventory_path,'sha256':sha(repo/inventory_path),'H_registry_reference':matches}
 # Version-free graph comparison is justified by actual full member payloads,
 # not by assuming the opaque registry IDs use today's fingerprint serializer.
 passed=bool(unmapped) and group_only and len(records)==144 and len(complete)==len(records) and historical_h_binding and source_binding
 return {'source':source,'type':'RUN_BOUND_LEGACY_MEMBER_SUPERSET','complete_membership':passed,'binding_verified':passed,
  'structure_ids':sorted(set(info['structure_ids'])|superset),'unmapped_opaque_ids_covered':len(unmapped),
  'checks':{'all_unmapped_ids_group_only':group_only,'all_144_inputs_complete_and_hash_bound':len(complete)==len(records)==144,
   'H_registry_and_inventory_bound_by_actual_artifact_manifest':historical_h_binding,'actual_inherited_H_reference_hash_matches':source_binding},
  'input_certificates':checks,'inheritance_reference':reference,'registry_projection_sha256':pm[source]['projection_sha256'],
  'derivation':'All unmapped IDs are explicitly attributed only to the 144 legacy input files in the historical H registry/inventory. Each exact byte-bound input has complete recorded member structures. Later registries explicitly hash-reference H. Comparing the entire physical member superset does not require interpreting every opaque historical fingerprint.',
  'does_not_assert_per_ID_graph_recovery':True,'use':'USE_UNRESOLVED'}

def review(root,repo,resume=False):
 if (root/'P1_complete.json').exists():
  if not resume:raise ValueError('P1 already complete')
  verify(root,read(root/'P1_complete.json')['files']);write(root/'P1_resume.json',{'state':'PASS','new_enumerations':0,'new_graph_recoveries':0,'new_model_operations':0,'new_qualification_computations':0,'core_hashes_unchanged':True,'time':now()});return
 acceptance=read(root/'review_acceptance.json');assert acceptance['state']=='PASS' and acceptance['review_sha256']==sha(repo/'analyse/c_independent_validation/review.py')
 ident=read(root/'run_identity.json');verify(repo,ident['safe_inputs'])
 pm=read(root/'projection_manifest.json');values={}
 for rel,info in pm.items():
  assert sha(root/info['path'])==info['projection_sha256'];r=read(root/info['path'])
  if r['state']=='PROJECTED':values[rel]=r['identity_only']
 lookup=make_catalogue(repo,root,values);plan=read(root/'identity_projection_plan.json');certs={};details=[]
 for source,spec in plan.items():
  if not spec['old_obligation']:continue
  if source not in values:
   certs[source]={'source':source,'complete_membership':False,'binding_verified':False,'structure_ids':[],'reason':pm[source]['state']};continue
  member=members(values[source],lookup);details.append({'source':source,**member})
  if source.endswith('/exposure_registry.json') and member['unmapped']:
   cert=registry_certificate(repo,root,values,source,member,pm)
  else:
   complete=bool(member['structure_ids']) and not member['unmapped'] and not member['unsupported_structure_paths']
   cert={'source':source,'type':'ACTUAL_COMPLETE_RECORDED_MEMBERS','complete_membership':complete,'binding_verified':complete,
    'structure_ids':member['structure_ids'],'source_sha256':pm[source]['source_sha256'],'projection_sha256':pm[source]['projection_sha256'],
    'member_objects':member['member_objects'],'direct_structures':member['direct_structures'],'unmapped_objects':len(member['unmapped']),
    'identity_map_evidence':'identity_mapping_catalogue.json','use':'USE_UNRESOLVED',
    'derivation':'Entire actual JSON/JSONL source projected. All recorded structural payloads and member identity fields resolve into concrete graph structures or conservative explicitly co-occurring identity unions; no score values used. Earlier complete development-input pools remain part of the reused source matrix.'}
  certs[source]=cert
 write(root/'new_source_member_certificates.json',certs);write(root/'member_mapping_details.json',details)
 oldmatrix=read(repo/PREVIOUS/'candidate_source_matrix.json');original=read(repo/P4/'C_preparation/C_unlabeled_manifest.json');om=read(repo/MINIMAL/'orbit_manifest.json')
 orbits={aid:set(read(repo/MINIMAL/v['path'])['unique_coarse_hashes']) for aid,v in om['architectures'].items()};matrix=[];changes=[];byarch=defaultdict(list)
 for old in oldmatrix:
  if old['source_id'].startswith('coverage:'):continue
  row=dict(old);aid=row['architecture_id'];source=row['source_id']
  if aid!=EXCLUDED and source in certs:
   row.update(decision=decision(certs[source],orbits[aid]),previous_decision=old['decision'],proof='new_source_member_certificates.json#'+source,evidence_sha256=digest(certs[source]),relation=certs[source].get('type','INCOMPLETE_PROJECTION'))
   if row['decision']!=old['decision']:changes.append(row)
  byarch[aid].append(row);matrix.append(row)
 ar=[];gaps=defaultdict(list)
 for a in original['architectures']:
  aid=a['architecture_fingerprint'];rows=byarch[aid]
  unresolved=[r['source_id'] for r in rows if r['decision'] in ('PROVENANCE_UNRESOLVED','HISTORICAL_OVERLAP_POSSIBLE')]
  if aid==EXCLUDED:cls='PRECAUTIONARY_PROVENANCE_EXCLUSION'
  elif any(r['decision']=='DEVELOPMENT_USED' for r in rows):cls='DEVELOPMENT_USED'
  else:cls='PROVENANCE_UNRESOLVED' if unresolved else 'ELIGIBLE_WITH_EVIDENCE'
  if cls=='PROVENANCE_UNRESOLVED':
   for source in unresolved:gaps[source].append(aid)
  ar.append({'architecture_id':aid,'classification':cls,'candidate_ids':[c['candidate_id'] for c in original['configurations'] if c['architecture_fingerprint']==aid],
   'unresolved_source_count':len(unresolved) if aid!=EXCLUDED else None,'decision_counts':dict(Counter(r['decision'] for r in rows))})
 amap={a['architecture_id']:a for a in ar};config=[{'candidate_id':c['candidate_id'],'architecture_id':c['architecture_fingerprint'],'classification':amap[c['architecture_fingerprint']]['classification']} for c in original['configurations']]
 qualified=[c for c in original['configurations'] if amap[c['architecture_fingerprint']]['classification']=='ELIGIBLE_WITH_EVIDENCE'];U=len(qualified)
 state='NO_QUALIFIED_C' if U==0 else 'QUALIFIED_SUBSET_BUDGET_BLOCKED' if U<71 else 'ENDPOINT_COMPARISON_EXHAUSTIVE_POOL_BLOCKED' if U==71 else 'QUALIFIED_C_READY_FOR_PROTOCOL'
 # Reuse previously accepted ID/HP/feature checks and verify their exact input
 # identities; no VAE invocation or graph/feature regeneration.
 identity_checks=read(repo/MINIMAL/'candidate_identity_checks.json')
 assert identity_checks['source_feature_sha256']==sha(repo/P4/'C_preparation/C_features.json')
 assert identity_checks['dictionary_sha256']==sha(repo/P4/'C_preparation/frozen_dictionary.json')
 assert U<=176 and U%4==0
 eligibility={'state':state,'architectures':ar,'configurations':config,'architecture_counts':dict(Counter(a['classification'] for a in ar)),
  'configuration_counts':dict(Counter(c['classification'] for c in config)),'original_order_retained':True,'old_results_modified':False}
 write(root/'C_eligibility_manifest.json',eligibility)
 write(root/'C_qualified_manifest.json',{'state':state,'architecture_count':U//4,'configuration_count':U,'configurations':qualified,
  'original_manifest_sha256':sha(repo/P4/'C_preparation/C_unlabeled_manifest.json'),'feature_sha256':sha(repo/P4/'C_preparation/C_features.json'),
  'dictionary_sha256':sha(repo/P4/'C_preparation/frozen_dictionary.json'),'coverage_71_over_U':71/U if U else None,
  'transductive_structure_exposure':'Frozen dictionary contains A and all original 45 C structures; no label-driven revision','performance_evidence':False})
 write(root/'candidate_source_matrix.json',matrix)
 with (root/'candidate_source_matrix.csv').open('w',newline='') as f:
  keys=list(dict.fromkeys(k for row in matrix for k in row));writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(matrix)
 write(root/'evidence_delta.json',{'prior_unresolved_obligations':164,'precautionary_architectures':1,'precautionary_configurations':4,'focused_obligations':161,
  'complete_new_source_certificates':sum(c['complete_membership'] for c in certs.values()),'bound_registry_superset_certificates':sum(c.get('type')=='RUN_BOUND_LEGACY_MEMBER_SUPERSET' and c['complete_membership'] for c in certs.values()),
  'changed_candidate_source_decisions':len(changes),'remaining_candidate_affected_sources':len(gaps),'new_permutation_evaluations':0,'prior_permutation_evaluations_reused':5304})
 write(root/'remaining_gaps.json',[{'source':p,'affected_architectures':a,'reason':certs.get(p,{}).get('reason','Member/scope obligation not closed')} for p,a in gaps.items()])
 summary={'state':'COMPLETE','qualification':state,'qualified_architectures':U//4,'qualified_configurations':U,'architecture_counts':eligibility['architecture_counts'],
  'configuration_counts':eligibility['configuration_counts'],'source_certificates':len(certs),'complete_certificates':sum(c['complete_membership'] for c in certs.values()),
  'remaining_sources':len(gaps),'source_projection_access':read(root/'projection_summary.json')['access'],'operations':ZERO,'coverage_71_over_U':71/U if U else None,'new_graph_recoveries':0,'new_orbit_evaluations':0,'cutoff':read(root/'protocol.json')['cutoff']}
 write(root/'P1_summary.json',summary)
 status=read(root/'status.json');status.update(P1='COMPLETE',P2='PENDING' if U>71 else 'NOT_STARTED');write(root/'status.json',status)
 names=['new_source_member_certificates.json','member_mapping_details.json','identity_mapping_catalogue.json','C_eligibility_manifest.json','C_qualified_manifest.json','candidate_source_matrix.json','candidate_source_matrix.csv','evidence_delta.json','remaining_gaps.json','P1_summary.json']
 write(root/'P1_complete.json',{'files':{n:sha(root/n) for n in names},'review_code_sha256':sha(repo/'analyse/c_independent_validation/review.py'),'time':now()})
 print(json.dumps(summary),flush=True)
