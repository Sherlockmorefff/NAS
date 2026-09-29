"""Verify and freeze the minimal immutable parent source; never read Q truth."""
import argparse,hashlib,json,pathlib,shutil,subprocess,datetime,platform,importlib.metadata
RID='gp-block-distance-v1-20260924T125944Z'
PARENT=pathlib.Path('/home/fangxy/D-VAE-worktrees/dkl-oof-calibration-v1-20260924T095621Z/artifacts/dkl-oof-calibration-v1-20260924T095621Z')
SHA='30b14b5e1a179fbaf253708add07f8a7b8187b36b4224467372bab889fdd8d4d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main(root):
 lock=root/'prepare_once.lock';lock.mkdir();write(lock/'entry.json',{'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'permanent':True})
 assert sha(PARENT/'evidence.tar.gz')==SHA,'BLOCKED_SOURCE_MISMATCH package'
 manifest={l.split('  ',1)[1]:l.split('  ',1)[0] for l in (PARENT/'FILES.sha256').read_text().splitlines()};files={};origins={}
 def cp(old,new):
  assert old=='FILES.sha256' or manifest[old]==sha(PARENT/old),old
  p=root/new;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/old,p);files[new]=sha(p);origins[new]=str(PARENT/old)
 for f in ['worklist.json','protocol.json','budget.json','source_manifest.json','outer_predictions_manifest.json','FILES.sha256','README.md','REPRODUCE.md','code_manifest.json']:
  cp(f,'source/parent_'+f)
 for f in ['public_inputs.json','subsets.json','folds.json']:cp(f,f)
 for f in range(5):cp(f'queries/fold{f}.json',f'queries/fold{f}.json')
 contexts=json.loads((PARENT/'worklist.json').read_text())['outer_contexts'];assert len(contexts)==30
 for sid in sorted({c['subset_id'] for c in contexts}):cp(f'training/{sid}.json',f'training/{sid}.json')
 for c in contexts:
  assert c['n']==50
  for m in ['HEAD','GP_CONST','GP_TREND']:
   rel=f'source/parent_predictions/{c["context_id"]}_{m}.json';cp(rel,rel)
 for f in ['stage01_core.py','residual_core.py','readout_core.py','readout_diagnostics.py','score_readout.py','score_oof.py']:
  cp('code/'+f,'source/frozen_code/'+f)
 cp('source/root_recipe_protocol.json','source/root_recipe_protocol.json')
 cp('source/parent_protocol.json','source/readout_v2_protocol.json')
 for f in ['completion.json','independent_oof_scoring.json','source_split_review.json','release_order.json']:
  cp('audit/'+f,'source/parent_audit/'+f)
 old=json.loads((PARENT/'source_manifest.json').read_text())
 assert old['parent_package_sha256']=='d875ca5769e273de10083529d0542a3cf85e71d70ce8283cca957f37896ccedf'
 assert old['root_recipe_package_sha256']=='39d9fc6474e0d9da6f0a456452a25f6b04f51c01442f6c4cf1823ba6ca0416ce'
 source={'run_id':RID,'parent_run_id':PARENT.name,'parent_root':str(PARENT),'parent_package_sha256':SHA,'readout_v2_package_sha256':old['parent_package_sha256'],'root_recipe_package_sha256':old['root_recipe_package_sha256'],'files':files,'origins':origins,'source_label_path':old['source_label_path'],'expected_labels_sha256':old['expected_labels_sha256'],'legal_A_scoring_truth_path':old['source_label_path'],'legal_A_scoring_truth_sha256':old['expected_labels_sha256'],'outer_labels_copied':False,'historical_model_seconds':{'residual':819.0296507184177,'readout_v2_including_v1':521.0264944955707,'oof':656.9550440832973},'new_budget_carryover':0,'historical_reference_bytes_unchanged':True,'algorithmic_isolation':'Per-current-T labels/SE, public Q inputs only; overlapping T allowed. No claim of OS-level byte denial.'}
 write(root/'source_manifest.json',source)
 oldprotocol=json.loads((PARENT/'protocol.json').read_text())
 protocol={'run_id':RID,'version':'block-distance-v1','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'normative_prompt':'PROMPT.md','normative_prompt_sha256':sha(root/'PROMPT.md'),'parent_run_id':PARENT.name,'parent_package_sha256':SHA,'methods':['RAW16_ISO','HP4_ISO','RAW16_BLOCK2'],'budget':{'model_seconds':600,'synthetic_optimizer_calls_limit_all_attempts':300,'carryover_seconds':0,'timing':{'n_train':50,'n_query':38,'warmup_steps':5,'segments':3,'steps_per_segment':20,'estimator':'max segment mean','total_timing_optimizer_calls':195},'formula':'used + 1.5*(18000*(t_A+t_B+t_C)+setup_est) + max(20,2*diagnostic_est)','automatic_resume':False},'steps':{'INIT_GP':300,'FIT_GP':1500},'matrix':{'subsets':10,'real_fits':30,'formal_updates':54000,'INIT_GP_updates':9000,'FIT_GP_updates':45000,'updates_per_method':18000,'new_endpoints':30,'new_candidate_rows':1074,'historical_references':90,'raw_scoring_units':120,'head_masks':30,'saved_states':90,'unique_diagnostics':240},'input':{'A_C':'T-standardized original frozen z12 plus transformed actual HP4','B':'exact same last4 columns','x_floor':1e-6,'y_floor':1e-8,'std_correction':0,'known_noise':'SE_T^2/y_scale_T^2; mean SE already incorporates 3 seeds','query_allowed_keys':['ids','architectures','x_raw'],'model_seed':None,'execution_rng_seed':0},'models':{'kernel':'constant-mean Exact Matern nu=2.5','A':'one lengthscale all16','B':'one lengthscale last4','C':'two raw scalar lengths for z12 and HP4; one fixed unit Matern, no third trainable scale','initial':{'mean':0,'lengthscale':1,'output_variance':1,'extra_noise_variance':.01},'trainable_scalars':{'RAW16_ISO':4,'HP4_ISO':4,'RAW16_BLOCK2':5},'block_dimension_normalization':False},'numerics':oldprotocol['numerics'],'optimizer':{'type':'Adam','betas':[.9,.999],'eps':1e-8,'weight_decay':0,'global_clip':5,'INIT_GP_lr':[[1,240,.01],[241,300,.001]],'FIT_GP_lr':[[1,1000,.01],[1001,1500,.001]],'stage_boundary':'own GP parameters inherited; fresh empty Adam','objective':'negative ExactMarginalLogLikelihood normalized once by n50'},'diagnostics':{'INIT_GP':[0,240,300],'FIT_GP':[0,1000,1200,1300,1400,1500],'FIT_0_alias_INIT_300':True,'unique_per_fit':8,'states_total_steps':[0,300,1800],'stability_windows':[[1200,1300],[1300,1400],[1400,1500]],'thresholds':{'relative_nll':1e-4,'max_mean_raw':.001,'max_latent_sd_raw':.001,'relative_TT_kernel_frobenius':.01},'expanded_model_ambiguity_nll_per_observation':1e-6,'distance_pairs_per_T':1225,'zero_distance_share':None},'prediction':{'full_covariance':True,'unclamped_latent_diagonal':True,'variance':'y_scale^2*(max(latent_diagonal_std,0)+extra_variance_std)','no_query_SE_before_seal':True},'scoring':{'aggregation':'historical 3 model seeds per subset; then 2 samples per fold; then 5 folds equally','head_masks':'3 oldHEAD individual topceil20 percent IDlex; newmethod metric eachmask thenaverage','regions':['all_Q','HEAD_pred20','true_top20','true_bottom20','champions'],'TOL_PP':1e-9,'z':{'50':.6744897501960817,'80':1.2815515655446004,'90':1.6448536269514722,'95':1.959963984540054},'probability':'v_observed=v_model+Q_SE^2 exactly once','flags':{'BLOCK_SCALE_SELECTION_SIGNAL':{'regret_improvement_min_pp':.20,'positive_folds_min':4,'both_samples_positive':True,'gain_improvement_min_pp':-.02},'HISTORICAL_SELECTION_RETAINED':{'baselines':['HEAD','H_TREND'],'regret_increase_max_pp':.10,'gain_fall_max_pp':.02},'ARCHITECTURE_INCREMENT_SIGNAL':'same selection rule for C versus B','PROBABILITY_NO_LARGE_DEGRADATION_C_VS_A':{'nlpd_increase_max_nat':.10,'IS90_increase_max_pp':1,'total_90cov_fall_max':.05,'champion_90cov_fall_max':.05},'OPTIMIZATION_READY':{'RAW16_ISO_stable_min':8,'RAW16_BLOCK2_stable_min':8},'BLOCK_DISTANCE_CANDIDATE_FOR_PROTOCOL_REVIEW':'BLOCK_SCALE_SELECTION_SIGNAL and HISTORICAL_SELECTION_RETAINED and OPTIMIZATION_READY'}},'failure':{'source':'BLOCKED_SOURCE_MISMATCH','budget':'STOPPED_BUDGET_PARTIAL','formal':'FAILED_FORMAL','unknown_tail':'INTERRUPTED_UNVERIFIED'},'interpretation':'Developmental A evidence only, no independent confirmation, no DKL_PASS, no BO authorization','not_authorized':['B/test','GNN/VAE calls','neural mapping','TREND or SEMI new fits','OOF','tau calibration','ARD16','sum/product kernels','extra priors','checkpoint choice','restarts','search','automatic next stage']}
 write(root/'protocol.json',protocol)
 write(root/'status.json',{'status':'PREPARED','formal_updates':0,'completed_fits':0,'outer_labels_released':False})
 env={'python':platform.python_version(),'platform':platform.platform(),'git_head':subprocess.check_output(['git','-C',str(root.parents[1]),'rev-parse','HEAD'],text=True).strip(),'device':'cpu','dtype':'float64','threads':1,'deterministic':True}
 for n in ['torch','gpytorch','numpy']:env[n]=importlib.metadata.version(n)
 write(root/'environment.json',env);print(json.dumps({'verified_source_files':len(files),'new_budget':600,'new_start':0}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);main(p.parse_args().root)
