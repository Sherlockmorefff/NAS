#!/usr/bin/env python3
"""Copy only pinned source states, selected observations and public query inputs."""
import argparse,hashlib,json,shutil,datetime,subprocess,platform
from pathlib import Path
RUN='dkl-readout-frozen-v1-20260924T072004Z'
SOURCE=Path('/home/fangxy/D-VAE/artifacts/dkl-residual-constraint-v1-20260924T024011Z')
PACKAGE='39d9fc6474e0d9da6f0a456452a25f6b04f51c01442f6c4cf1823ba6ca0416ce'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main(root):
 raise RuntimeError('V2 reuses verified parent inputs; use migrate_readout_v2.py in a new directory')

 if (root/'source_manifest.json').exists():raise RuntimeError('Preparation is one-shot')
 expected={l.split('  ',1)[1]:l.split('  ',1)[0] for l in (SOURCE/'FILES.sha256').read_text().splitlines()}
 if sha(SOURCE/'FILES.sha256')!='e4560bf4709d4e4cc289552fe4af4cf7f6590a589b94c78432af7af593aa7afe':raise RuntimeError('Source inventory mismatch')
 files={};origins={}
 def cp(src,dst=None):
  dst=dst or src;p=SOURCE/src;q=root/dst
  if src not in expected or sha(p)!=expected[src]:raise RuntimeError('BLOCKED_SOURCE_INCOMPLETE: '+src)
  q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
  files[dst]=sha(q);origins[dst]=str(p)
 for f in ['worklist.json','subsets.json','folds.json','public_inputs.json','protocol.json','source_manifest.json','resource_ledger.json','README.md','REPRODUCE.md']:
  cp(f,'source/reference/'+f)
 for f in ['subsets.json','folds.json','public_inputs.json']:cp(f)
 old=json.loads((root/'source/reference/worklist.json').read_text());contexts=[];endpoints=[];historical=[];seen=set()
 methods=['GP_CONST','GP_TREND','GP_SEMI']
 for idx,c in enumerate(sorted(old['contexts'],key=lambda x:x['context_id'])):
  c={k:c[k] for k in ['context_id','subset_id','fold','sample_seed','model_seed','n','selected_ids_sha256']}
  c['group_order']=methods[idx%3:]+methods[:idx%3];contexts.append(c);cid,sid=c['context_id'],c['subset_id']
  cp('checkpoints/'+cid+'_COMMON_gp.pt','source/common/'+cid+'_COMMON_gp.pt')
  cp('diagnostics/'+cid+'_COMMON_gp/j300.json','source/common_diagnostics/'+cid+'_j300.json')
  for sourcefile in ['training/'+sid+'.json','queries/fold'+str(c['fold'])+'.json']:
   if sourcefile not in seen:cp(sourcefile);seen.add(sourcefile)
  for method in ['HEAD']+c['group_order']:endpoints.append({**c,'method':method,'unit_id':cid+'_'+method})
  for uid in [sid+'_RAW16',cid+'_R_FROZEN',cid+'_R_REG',cid+'_R_DKL_C']:
   rel='source/historical_predictions/'+uid+'.json'
   if rel not in files:
    cp('predictions/'+uid+'.json',rel);historical.append({'path':rel,'sha256':files[rel],'unit_id':uid})
 assert len(contexts)==60 and len(endpoints)==240 and len(historical)==200
 dump(root/'worklist.json',{'contexts':contexts,'endpoints':endpoints,'context_count':60,'new_gp_fit_count':180,'head_count':60,'expected_endpoint_count':240,'formal_optimizer_updates':270000,'execution_order':'lexical context_id; HEAD then cyclic GP order'})
 labelsha=expected['scoring_B/labels_A.json'];assert labelsha=='5645881e8eac3e27673dafab7d1ffa61c99c58eb4aa6f4c85a195151e6900257'
 dump(root/'source_manifest.json',{'files':files,'origins':origins,'source_root':str(SOURCE),'source_package_sha256':PACKAGE,'historical_predictions':historical,'label_source_path':str(SOURCE/'scoring_B/labels_A.json'),'source_label_path':str(SOURCE/'scoring_B/labels_A.json'),'expected_labels_sha256':labelsha,'scoring_labels_expected_sha256':labelsha,'labels_copied':False,'source_inventory_sha256':sha(SOURCE/'FILES.sha256')})
 protocol={'run_id':RUN,'version':'frozen-readout-v1','normative_prompt_path':'PROMPT.md','normative_prompt_sha256':sha(root/'PROMPT.md'),'primary_n':50,'boundary_n':20,'methods':['HEAD']+methods,'contexts':60,'gp_fits':180,'head_evaluations':60,'steps_per_gp':1500,'formal_updates':270000,
 'source':{'root':str(SOURCE),'package_sha256':PACKAGE,'state':'saved COMMON_gp300 only','new_common_training':False},
 'features':{'h0':'frozen residual16 after common MSE1000+GP300','s0':'original frozen regression head in standardized-y units','refit_scalers':False,'mapper_and_head_trainable':False},
 'models':{'HEAD':'s0 deterministic; uncertainty N/A','GP_CONST':'learned constant + Scale(Matern5/2 isotropic h0)','GP_TREND':'fixed s0 + Scale(Matern5/2 isotropic h0)','GP_SEMI':'fixed s0 + Scale(Matern5/2 isotropic h0) + .0625 + .25*s0*s0prime','beta_prior_mean':[0,1],'beta_prior_covariance':[[.0625,0],[0,.25]],'beta_integrated_not_optimized':True,'shared_initial_gp_parameters':'exact saved source raw parameters; fresh independent Adam'},
 'optimization':{'optimizer':'Adam','betas':[.9,.999],'eps':1e-8,'weight_decay':0,'lr_updates_1_1000':.01,'lr_updates_1001_1500':.001,'gradient_global_norm_clip':5,'objective':'pure negative ExactMLL normalized by n exactly once','priors_or_added_losses':False,'endpoint_selection':False},
 'numerics':{'device':'cpu','dtype':'float64','threads':1,'deterministic':True,'jitter':1e-8,'cholesky_max_tries':1,'max_cholesky_size':10000,'min_fixed_noise':0,'fast_pred_var':False,'length_lower':1e-4,'output_lower':1e-8,'extra_noise_lower':1e-6,'latent_roundoff_lower':-1e-8},
 'diagnostics':{'steps':[0,1000,1200,1300,1400,1500],'stability_windows':[[1200,1300],[1300,1400],[1400,1500]],'nll_delta_limit':'1e-4*max(1,abs(start_nll))','raw_mean_delta':.001,'raw_latent_sd_delta':.001,'matern_and_full_kernel_relative_frobenius':.01,'top5_stability_separate':True},
 'budget':{'model_seconds':1200,'synthetic_optimizer_calls_limit_all_attempts':300,'admission_formula_frozen_before_timing':'used + 1.5*sum(cell_step_seconds*45000 for all 6 cells) + max(30,2*preflight_prediction_seconds*1080/preflight_prediction_calls)','expected_formula':'used + sum(cell_step_seconds*45000) + preflight_prediction_seconds*1080/preflight_prediction_calls','admission_failure':'BLOCKED_BUDGET_ADMISSION; no matrix reduction','old_model_seconds':819.0296507184177,'old_cost_deducted':False,'all_new_model_computation_charged':True,'automatic_resume':False,'formal_lock':'permanent exclusive directory'},
 'scoring':{'release':'uniform seal only; isolated scorer; legal A only','ranking':'mean_raw descending then exact candidate_id ascending','select_k':5,'champion_tolerance_pp':1e-9,'z90':1.6448536269514722,'observed_variance':'yscale^2*(latent_std+extra_std)+query_SE^2; jitter excluded','aggregation':'3 model seeds then 2 sample seeds then 5 folds equally; paired first; RAW once per subset','significance_tests':False,'flag_rules':'exact normative prompt sections; primary n50; n20 separate','no_next_stage':True},
 'excluded':['B_pool','test','GNN/VAE retraining','mapper/head updates','joint DKL','real search','new methods','hyperparameter sweep'],
 'limitations':['Repeated developmental use of A','Head fitted on same selected training labels','GP hyperparameters point estimated','Finite beta prior excludes head/mapper uncertainty','Common GP300 initialization may favor constant mean','Historical comparison unequal budgets']}
 dump(root/'protocol.json',protocol)
 dump(root/'preparation.json',{'status':'PREPARED_NO_MODEL_COMPUTATION','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(files),'query_labels_read':False,'source_states':60,'historical_predictions':200,'optional_DKL_residual_diagnosis_20260924_md':'not found; optional'})
 dump(root/'status.json',{'status':'PREPARED_NO_MODEL_COMPUTATION'})
 print({'prepared':str(root),'source_files':len(files),'contexts':60})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);main(a.parse_args().root)
