"""Freeze the agreed stage-2 protocol and code before releasing any labels."""
import argparse
import datetime
import hashlib
import json
import importlib.metadata
import os
import platform
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    protocol={
        'task_id':root.name,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Stage2 offline diagnostic on repeatedly used A only; no stage3',
        'phase01_protocol_sha256':'494dfb5388a75c6b4ef6468e08b2205d4971ec981675812c9941720f83d9c628',
        'pool':{'candidates':179,'architectures':96,'features_file_sha256':'4aedd1482cdc36c055c4016a25fe4a8158c305321673e16e868b994ab830c797',
                'candidate_identity_file_sha256':'c66c2aa70c4cf2d5fe7d6462bd822a35c4fdd64e950b326cfce8fcf013cab46e',
                'sorted_A_ids_semantic_sha256':'d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2',
                'recipe_semantic_sha256':'40bb99cbc3e33dcc417869a97d969b8e8e1c4ca7c6a5fad66c29ed85edea5978'},
        'recipe':{'dataset':'Cora','split':'public','features':'NormalizeFeatures','hp_mode':'global4','normalization':'none',
                  'batch':'full_batch','max_epochs':300,'patience':80,'training_seeds':[0,1,2]},
        'folds':{'count':5,'query_counts_expected':[38,36,36,34,35],
                 'reuse':'original partition when present; reject wrong identity/overlap/coverage',
                 'only_if_absent':'architecture SHA256 UTF8 sort, fingerprint tiebreak, position mod5'},
        'selection':{'sample_seeds':[20260922,20260923],'model_seeds':[0,1,2],'n':[20,50],
            'algorithm':'rng=default_rng(SeedSequence([sample_seed,fold])); ordered=rng.permutation(sorted(training_pool_ids)); ids20=ordered[:20]; ids50=ordered[:50]',
            'no_label_or_SE_filter':True,'fresh_initialization_for_each_n':True},
        'work_order':{'context_order':['sample_seed','model_seed','fold','n'],'n_order':[20,50],
                      'raw':'once per subset, before common training when model_seed=0',
                      'mapped_branches':['FROZEN','REG','DKL'],'serial':True},
        'matrix':{'common_initializations':60,'mapped_endpoints':180,'raw_endpoints':20,'total_endpoints':200,
                  'common_updates':78000,'branch_updates':240000,'raw_updates':26000,'formal_optimizer_updates':344000},
        'device':'cpu','dtype':'float64','threads':1,'deterministic_algorithms':True,
        'input':{'z_dimensions':12,'hp':['log10(lr)','dropout','log2(hidden_dim)','log10(l2)'],
                 'VAE':'reuse existing frozen codes only, no forward or weight loading',
                 'standardization':'only actual selected20/50, ddof0, fixed thereafter','x_scale_floor':1e-6,'y_scale_floor':1e-8},
        'mapping':{'source':'stage01 exact JointMetricModel','hp':'4->16 SiLU','body':'28->32 SiLU->32 SiLU->8',
                   'head':'8->1','dropout':False,'batchnorm':False,'load_historical_trained_state':False},
        'C_loss':'mean((head(e)-y_std)^2)+0.001*mean(sum(e^2,axis=1))',
        'shared':[{'mode':'regression','updates':1000},{'mode':'gp','updates':300}],
        'branches':{'FROZEN':[{'mode':'gp','updates':1000}],
                    'REG':[{'mode':'regression','updates':1000},{'mode':'gp','updates':1000}],
                    'DKL':[{'mode':'joint','updates':1000}],
                    'RAW16':[{'mode':'gp','updates':300},{'mode':'gp','updates':1000}]},
        'GP':{'type':'ExactGP','mean':'ConstantMean','kernel':'ScaleKernel(MaternKernel(nu=2.5,ard_num_dims=None))',
            'known_noise':'SE^2 / selected_y_scale^2; stored SE already estimates three-seed mean',
            'extra_noise_initial':0.01,'extra_noise_lower':1e-6,'lengthscale_initial':1.0,'lengthscale_lower':1e-4,
            'outputscale_initial':1.0,'outputscale_lower':1e-8,'mean_initial':0.0,
            'loss':'-ExactMarginalLogLikelihood(gp(x),y); already divided by n once',
            'priors':[],'added_losses':[],'jitter':1e-8,'cholesky_max_tries':1,'max_cholesky_size':10000,'min_fixed_noise':0,
            'embedding_standardization':'none','raw16':'standardized16 input directly; constant parameter initialization'},
        'optimizer':{'type':'Adam','mapping_lr':0.001,'head_lr':0.001,'gp_lr':0.01,'betas':[0.9,0.999],'eps':1e-8,
                     'weight_decay':0.0,'new_per_block':True,'schedule':'first80percent base LR, final20percent 0.1base',
                     'gradient_clip_global_norm':5.0,'clip_scope':'all optimized groups jointly','log_preclip_norm':True},
        'diagnostics':{'interval':100,'also_endpoint':True,'sample_inside_current_block':True,
                       'record':['objective','preclip_gradient_norm','representation_scale','GP_parameters','constraint_distance','kernel_change','elapsed_seconds'],
                       'checkpoint_interval':100,'only_endpoint_for_science':True},
        'budget':{'cumulative_fit_predict_seconds':7200,'clock':'time.monotonic/perf_counter',
                  'includes':['all formal training','formal predictions','real verification','synthetic verification separately accounted','failed/retried training'],
                  'excludes':['audit','implementation','packaging'],
                  'estimate':'first formal n20/n50 contexts, retained in matrix; no extra real pilot'},
        'prediction':{'seal_all_before_scoring':True,'include_query_y_or_SE':False,
                      'rank':'GP mean raw accuracy descending, candidate_id lexicographic tie',
                      'select':5,'architecture_diversity_filter':False,'mean_clipping':False,
                      'latent_variance_negative_tolerance_standardized':-1e-8,
                      'small_negative':'preserve raw; clamp to zero only when sqrt needed'},
        'metrics':{'primary':'regret5=max(query_y)-max(selected5_y)',
                   'paired_improvement':'regret5_baseline-regret5_DKL; positive favors DKL',
                   'baselines':['REG','FROZEN','RAW16','exact uniform random5 from same query fold'],
                   'random_best':'sum_{i=5..m} y_sorted[i-1]*comb(i-1,4)/comb(m,5)',
                   'secondary':['top1_regret','MAE','RMSE','Spearman','90pct_coverage','95pct_coverage','interval_width','NLPD'],
                   'variance_observed':'y_scale^2*(latent_variance_std+extra_noise_variance_std)+query_SE^2',
                   'extra_noise_is':'variance, not standard deviation','accuracy_difference_report_unit':'percentage points (raw*100)',
                   'NLPD_unit':'accuracy units','constant_spearman':'undefined with reason'},
        'aggregation':{'separate_n':True,'order':'mean model_seeds within fold/sample; mean sample_seeds within fold; equal mean5folds',
                       'raw_counted_once_per_fold_sample':True,'statistics':'descriptive only; no naive t-test/bootstrap/significance',
                       'pairing':'same fold/sample/n/model seed; RAW16 reference reused, not independently replicated'},
        'classification':{'tie_tolerance_pp':1e-9,'ordered_if_elif_else':[
            '较一致的离线选优信号: mean_REG_improvement>tol AND >=4/5positive_folds AND DKL_mean_regret5<RAW16 AND <random5',
            '局部或混合信号: else if any positive_REG_fold',
            '未见相对继续回归的收益: else'],
            'incomplete':'不完整，暂不分类','four_of_five_is_not_statistical_test':True},
        'failure':'preserve raw error and state; no automatic jitter/constraint/seed change; budget partial; leakage or numerical invalidity INVALID',
        'not_authorized':['B labels','test accuracy','GNN training','VAE forward/training','Fuzzy training','real search','stage3','GPU','commit/push'],
        'next_stage_authorized':False,
    }
    write_new(root/'protocol.json',protocol)
    codefiles=[p for p in sorted((root/'code').rglob('*.py')) if '__pycache__' not in p.parts and not p.name.startswith('._')]
    codes={'files':{str(p.relative_to(root)):sha(p) for p in codefiles},
           'source_head':'64a44c14316f11debaf01cf8583426acfe1410e6','source_branch':'feature/accuracy-bottleneck-diagnostics',
           'stage01_dependency':'exact source bytes, no phase01 trained weights reused'}
    write_new(root/'code_manifest.json',codes)
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                 'packages':{name:importlib.metadata.version(name) for name in ['torch','gpytorch','numpy']},
                 'process_threads':{name:os.environ.get(name) for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
    write_new(root/'environment_frozen.json',environment)
    files=['protocol.json','public_inputs.json','folds.json','subsets.json','worklist.json','code_manifest.json','environment_frozen.json']
    manifest={'status':'FROZEN','frozen_at_utc':protocol['frozen_at_utc'],'labels_released':False,
              'files':{name:sha(root/name) for name in files},'protocol_sha256':sha(root/'protocol.json'),
              'code_manifest_sha256':sha(root/'code_manifest.json')}
    write_new(root/'frozen_manifest.json',manifest)
    print(json.dumps(manifest))


if __name__=='__main__':main()
