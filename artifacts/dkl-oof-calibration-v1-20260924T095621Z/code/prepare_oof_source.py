"""Freeze source identities without decoding any outer scoring observations."""
import argparse,hashlib,json,pathlib,shutil,subprocess,datetime,platform
RID='dkl-oof-calibration-v1-20260924T095621Z'
PARENT=pathlib.Path('/home/fangxy/D-VAE-worktrees/dkl-readout-frozen-v2-20260924T081732Z/artifacts/dkl-readout-frozen-v2-20260924T081732Z')
SHA='d875ca5769e273de10083529d0542a3cf85e71d70ce8283cca957f37896ccedf'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main(root):
 lock=root/'prepare_once.lock';lock.mkdir();write(lock/'entry.json',{'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'permanent':True})
 assert sha(PARENT/'evidence.tar.gz')==SHA
 manifest={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (PARENT/'FILES.sha256').read_text().splitlines()}
 src={};origins={}
 def cp(old,new):
  assert manifest[old]==sha(PARENT/old),old
  target=root/new;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/old,target)
  src[new]=sha(target);origins[new]=str(PARENT/old)
 for f in ['worklist.json','protocol.json','budget.json','source_manifest.json','predictions_manifest.json','FILES.sha256','README.md','REPRODUCE.md','code_manifest.json']:
  if f=='FILES.sha256':
   target=root/'source/parent_FILES.sha256';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/f,target);src['source/parent_FILES.sha256']=sha(target);origins['source/parent_FILES.sha256']=str(PARENT/f)
  else:cp(f,'source/parent_'+f)
 for f in ['public_inputs.json','subsets.json','folds.json']:cp(f,f)
 for f in range(5):cp(f'queries/fold{f}.json',f'queries/fold{f}.json')
 contexts=[c for c in json.loads((PARENT/'worklist.json').read_text())['contexts'] if c['n']==50]
 for sid in sorted({c['subset_id'] for c in contexts}):cp(f'training/{sid}.json',f'training/{sid}.json')
 for c in contexts:
  for m in ['HEAD','GP_CONST','GP_TREND']:cp(f'predictions/{c["context_id"]}_{m}.json',f'source/parent_predictions/{c["context_id"]}_{m}.json')
 for f in ['residual_core.py','stage01_core.py','stage2_core.py','readout_core.py','readout_data.py','readout_diagnostics.py','score_readout.py']:
  cp('code/'+f,'source/frozen_code/'+f)
 cp('source/reference/protocol.json','source/root_recipe_protocol.json')
 for f in ['audit/independent_readout_scoring.json','audit/v2_completion.json','audit/saved_state_checks.json','validation_parent/preflight.json','validation_parent/core_attempt_1/core_checks.json','validation_parent/final_identity_check/core_checks.json','validation_parent/final_identity_check/readout_diagnostics_checks.json','validation_parent/final_identity_check/acceptance.json']:
  if (PARENT/f).exists():cp(f,'source/reused_validation/'+pathlib.Path(f).name)
 old=json.loads((PARENT/'source_manifest.json').read_text())
 write(root/'source_manifest.json',{'parent_run_id':PARENT.name,'parent_root':str(PARENT),'parent_package_sha256':SHA,'root_recipe_run_id':'dkl-residual-constraint-v1-20260924T024011Z','root_recipe_package_sha256':'39d9fc6474e0d9da6f0a456452a25f6b04f51c01442f6c4cf1823ba6ca0416ce','files':src,'origins':origins,'source_label_path':old['source_label_path'],'expected_labels_sha256':old['expected_labels_sha256'],'outer_labels_copied':False,'historical_model_seconds':{'residual':819.0296507184177,'readout_v2_including_v1':521.0264944955707},'new_budget_carryover':0,'math_reuse':'Unchanged residual mapper/ExactGP/readout likelihood and prediction math; adapter only for new counting. Historical validation is source evidence, not new execution.'})
 parent=json.loads((PARENT/'protocol.json').read_text())
 write(root/'protocol.json',{'run_id':RID,'version':'oof-calibration-v1','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'normative_prompt':'PROMPT.md','normative_prompt_sha256':sha(root/'PROMPT.md'),'parent_run_id':PARENT.name,'parent_package_sha256':SHA,'budget':{'model_seconds':1500,'synthetic_optimizer_calls_limit_all_attempts':300,'carryover_seconds':0,'expected_formula':'used + fit_est + setup_est + pred_est','conservative_formula':'used + 1.5*(fit_est+setup_est) + max(30,2*pred_est)','formal_steps_by_class':{'COMMON_MSE':90000,'COMMON_GP':27000,'GP_CONST':135000,'GP_TREND':135000},'timing':{'n_fit':34,'n_holdout':17,'warmup_steps':5,'segments':3,'steps_per_segment':20,'estimator':'max segment mean','total_timing_optimizer_calls':260},'automatic_resume':False},'steps':{'COMMON_MSE':1000,'COMMON_GP':300,'GP_CONST':1500,'GP_TREND':1500},'matrix':{'outer_contexts':30,'inner_pipelines':90,'inner_prediction_sets':270,'oof_rows':4500,'formal_updates':387000,'tau_scalars':60,'outer_refs':90,'outer_derivatives':60,'outer_outputs':150},'numerics':parent['numerics'],'optimizer':{'type':'Adam','betas':[.9,.999],'eps':1e-8,'weight_decay':0,'global_clip':5,'MSE_lr':[.001,800,.0001,200],'COMMON_GP_lr':[.01,240,.001,60],'readout_lr':[.01,1000,.001,500]},'diagnostics':{'COMMON_MSE':[0,800,900,1000],'COMMON_GP':[0,240,300],'readout':parent['diagnostics'],'gp_readout_calls':1080,'common_mse_calls':360,'common_gp_calls':270,'head_train_and_holdout_calls':90},'data':{'inner_split':'oof-arch3-v1','n_outer':50,'fit_sizes':[33,34],'holdout_sizes':[16,17],'scaler':'U only ddof0; x floor1e-6 y floor1e-8','mean_SE':'SE squared once, no extra sqrt3','outer_truth_release':'all90inner pipelines +387000updates +4500OOF rows sealed then60tau +150outeroutputs sealed'},'calibration':{'estimator':'max(0,mean((y-mu_oof)^2-v_model_oof-SE^2))','weights':'50 candidates equally percontext andmethod','unit':'raw accuracy squared','transfer':'no n adjustment','mean_changes':False,'joint_covariance_changes':False},'scoring':{'z90':1.6448536269514722,'champion_tolerance_pp':1e-9,'rank':'descending exact mean then candidate ID ascending','aggregation':'3 model seeds then2sample seeds then5folds equally','flags':'normative PROMPT sections VIII','no_significance':True},'not_authorized':['B/test','joint DKL','GNN/VAE calls','BO/search','SEMI training','hyperparameter changes','mean calibration','automatic next stage']})
 write(root/'status.json',{'status':'PREPARED','formal_updates':0,'completed_pipelines':0,'outer_labels_released':False})
 info={'python':platform.python_version(),'platform':platform.platform(),'git_head':subprocess.check_output(['git','-C',str(root.parents[1]),'rev-parse','HEAD'],text=True).strip(),'device':'cpu','dtype':'float64','threads':1,'deterministic':True}
 import importlib.metadata
 for name in ['torch','gpytorch','numpy']:info[name]=importlib.metadata.version(name)
 write(root/'environment.json',info)
 print(json.dumps({'source_files':len(src),'parent_package_verified':True,'outer_contexts':len(contexts),'new_budget':1500,'new_start':0}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);main(p.parse_args().root)
