"""Fixed budget admission; saved measurements only, zero model calls."""
import argparse,json,pathlib,hashlib,datetime

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root):
 if (root/'admission.json').exists():raise RuntimeError('Admission already finalized; no retiming or overwrite')
 protocol=read(root/'protocol.json');budget=read(root/'budget.json');pre=read(root/'verification/preflight.json')
 checks={}
 for name in ['data_checks','calibration_checks','scoring_checks']:
  x=read(root/('verification/'+name+'.json'));assert x['pass'],name;checks[name]=sha(root/('verification/'+name+'.json'))
 assert pre['pass'] and pre['actual_optimizer_calls']==276 and budget['synthetic_optimizer_calls']==276
 assert budget['formal_updates']==budget['formal_optimizer_calls']==0
 assert budget['limit']==protocol['budget']['model_seconds']==1500
 for key in ['failed_optimizer_calls','failed_updates','discarded_updates','failed_training_attempts']:assert budget[key]==0,key
 assert len(read(root/'worklist.json')['pipelines'])==90 and read(root/'inner_splits_manifest.json')['subset_count']==10
 fit,setup,pred=(pre[k] for k in ('fit_est','setup_est','pred_est'));used=budget['seconds']
 expected=used+fit+setup+pred;conservative=used+1.5*(fit+setup)+max(30,2*pred)
 admitted=conservative<=1500;status='ADMITTED' if admitted else 'BLOCKED_BUDGET_ADMISSION'
 row={'status':status,'admitted':admitted,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'already_used':used,'fit_est':fit,'setup_est':setup,'pred_est':pred,'expected':expected,'conservative':conservative,'model_limit_seconds':1500,'formula':protocol['budget']['conservative_formula'],'preflight_sha256':sha(root/'verification/preflight.json'),'preflight_checks':checks,'protocol_sha256':sha(root/'protocol.json'),'worklist_sha256':sha(root/'worklist.json'),'source_manifest_sha256':sha(root/'source_manifest.json'),'timing_repetitions':1,'formal_updates_planned':387000,'formal_updates_completed':0,'historical_cost_deducted':False,'model_calls':0}
 (root/'admission.json').write_text(json.dumps(row,indent=2)+'\n')
 (root/'status.json').write_text(json.dumps({'status':status,'completed_pipelines':0,'formal_updates':0,'outer_labels_released':False,'budget_seconds':used,'automatic_resume':False},indent=2)+'\n')
 print(json.dumps(row))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);main(p.parse_args().root)
