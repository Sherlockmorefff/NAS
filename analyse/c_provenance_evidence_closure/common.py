from analyse.c_independence_minimal_audit.common import now,canonical,digest,sha,read,write,seal,csvwrite,log,P4,H,ORIGIN,V1,V2,PREP
from pathlib import Path
OLD='results/C_independence_minimal_audit_20261006T113512Z'
STRICT='results/posthoc/performance-aware-vae-strict-regression-global4-v2'
STRUCT='results/posthoc/structure-native-surrogate-pilot-global4-v1'
ZERO={k:0 for k in ['GP_fits','GP_optimization','GP_predictions','GNN_training','VAE_updates','decoder_calls','new_candidates','C_performance_reads','C_acquisition_scores','B_labels_or_SE_reads','node_test_score_reads','KG']}
def code_files(repo):return {str(p.relative_to(repo)):sha(p) for base in ['analyse/c_provenance_evidence_closure','tests/c_provenance_evidence_closure'] for p in sorted((repo/base).glob('*.py'))}
def verify(repo,files):
 for p,h in files.items():
  if sha(repo/p)!=h:raise ValueError('Identity mismatch: '+p)

