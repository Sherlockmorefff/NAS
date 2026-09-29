"""Prepare only frozen selected-training inputs and unlabelled queries."""
import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

OLD = Path('/home/fangxy/D-VAE/artifacts/dkl-stage2-v1-20260922T080157Z')
LENGTH = Path('/home/fangxy/D-VAE/artifacts/dkl-stage2-length-v1-20260922T092924Z')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def hashes(root):
    return {line[66:]:line[:64] for line in (root/'FILES.sha256').read_text().splitlines()}

def prepare(root):
    if (root/'protocol.json').exists(): raise FileExistsError('Already prepared')
    manifests={OLD:hashes(OLD),LENGTH:hashes(LENGTH)}; files={}; origins={}
    def cp(src,rel,dest=None):
        source=src/rel
        if sha(source)!=manifests[src][rel]: raise ValueError('Source hash mismatch: '+str(source))
        dest=dest or ('source/'+('stage2' if src==OLD else 'length')+'/'+rel)
        path=root/dest;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
        files[dest]=sha(path);origins[dest]=str(source)
    for src in (OLD,LENGTH):
        for rel in ['README.md','protocol.json','predictions_manifest.json']:
            cp(src,rel)
    for rel in ['worklist.json','subsets.json','folds.json','public_inputs.json']:
        cp(OLD,rel)
        cp(OLD,rel,rel)
    for p in sorted((OLD/'code').rglob('*.py')):
        if p.name.startswith(('test_','preflight_','freeze_')): continue
        rel=str(p.relative_to(OLD));cp(OLD,rel)
        if p.name not in ['run_stage2.py']: cp(OLD,rel,rel)
    for rel in ['code/length_core.py','code/length_diagnostics.py']:
        cp(LENGTH,rel);cp(LENGTH,rel,rel)
    contexts=read(root/'worklist.json')['contexts']; endpoints=[]
    seen=set()
    for ctx in contexts:
        cid,sid=ctx['context_id'],ctx['subset_id']
        for block in ['COMMON_regression','COMMON_gp','DKL_joint']:
            cp(OLD,f'checkpoints/{cid}_{block}.pt')
        cp(OLD,f'predictions/{cid}_DKL.json')
        cp(LENGTH,f'checkpoints/{cid}_j4000.pt')
        cp(LENGTH,f'predictions/{cid}_j4000.json')
        if sid not in seen:
            seen.add(sid)
            cp(OLD,f'training/{sid}.json');cp(OLD,f'training/{sid}.json',f'training/{sid}.json')
            cp(OLD,f'predictions/{sid}_RAW16.json');cp(OLD,f'states/{sid}_RAW16.pt')
            endpoints.append({**ctx,'unit_id':sid+'_RAW16','method':'RAW16','model_seed':None})
        for method in ['R_FROZEN','R_REG','R_DKL_C','R_DKL_U']:
            endpoints.append({**ctx,'unit_id':cid+'_'+method,'method':method})
    work=read(root/'worklist.json')
    work['source_endpoints']=work.pop('endpoints');work['endpoints']=endpoints
    work.update(total_endpoint_count=260,mapped_endpoint_count=240,formal_optimizer_updates=404000,
                execution_order='unchanged original contexts; RAW first when run_raw; R_FROZEN,R_REG,R_DKL_C,R_DKL_U')
    for ctx in work['contexts']:ctx['branch_order']=['R_FROZEN','R_REG','R_DKL_C','R_DKL_U']
    work.pop('identity',None);write(root/'worklist.json',work)
    # Root copy is intentionally the new B worklist; original remains immutable under source/.
    files.pop('worklist.json');origins.pop('worklist.json')
    public=read(root/'public_inputs.json');folds=read(root/'folds.json')['folds']
    byid={cid:(arch,z+hp) for cid,arch,z,hp in zip(public['ids'],public['architectures'],public['z'],public['hp'])}
    if len(byid)!=179 or len({x[0] for x in byid.values()})!=96:raise ValueError('A identity count')
    if digest(sorted(byid))!='d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2':raise ValueError('A IDs')
    for fold in folds:
        q=fold['query_ids'];t=fold['training_pool_ids']
        if {byid[c][0] for c in q}&{byid[c][0] for c in t}:raise ValueError('Architecture leakage')
        query={'ids':q,'architectures':[byid[c][0] for c in q],'x_raw':[byid[c][1] for c in q]}
        write(root/f'queries/fold{fold["fold"]}.json',query)
    subsets=read(root/'subsets.json')['subsets']
    for sid,subset in subsets.items():
        tr=read(root/f'training/{sid}.json')
        if tr['ids']!=subset['ids'] or tr['scaler']['fit_ids']!=tr['ids']:raise ValueError('Selected scope/order')
        if len(tr['ids'])==20 and tr['ids']!=subsets[sid.replace('_n20','_n50')]['ids'][:20]:raise ValueError('Nested subset')
    source_manifest={'files':files,'origins':origins,'contexts':contexts,'source_roots':{'stage2':str(OLD),'length':str(LENGTH)},
        'label_source_path':str(OLD/'scoring/labels_A.json'),'scoring_labels_expected_sha256':manifests[OLD]['scoring/labels_A.json'],
        'source_protocol_sha256':sha(OLD/'protocol.json'),'length_protocol_sha256':sha(LENGTH/'protocol.json'),
        'source_package_sha256':'3898e1189ae17f9d056a2478e94ab4a7f1b6b7539b6aae5fd26153dd5b642717',
        'length_package_sha256':'d584238341e5af86de190d97afb6f6419420786170130dc57b18fa349738c322',
        'labels_copied':False,'formal_updates':0,'prepared_at_utc':now()}
    write(root/'source_manifest.json',source_manifest)
    p=copy.deepcopy(read(OLD/'protocol.json'))
    p.update(task_id=root.name,frozen_at_utc=now(),scope='A historical cause diagnostic then gated B developmental constrained residual ablation',
      authoritative_specification={'path':'PROMPT.md','sha256':sha(root/'PROMPT.md'),'all_sections_normative':True},
      source_protocol_sha256=sha(OLD/'protocol.json'),length_protocol_sha256=sha(LENGTH/'protocol.json'))
    p.pop('phase01_protocol_sha256',None)
    p['folds'].pop('only_if_absent',None);p['folds']['reuse']='exact original frozen groups; no fallback'
    p['selection']['algorithm']='reuse frozen original subsets and ordering; no resampling'
    p['work_order']['mapped_branches']=['R_FROZEN','R_REG','R_DKL_C','R_DKL_U']
    p['matrix'].update(mapped_endpoints=240,total_endpoints=260,branch_updates=300000,formal_optimizer_updates=404000)
    p['mapping']={'type':'RESIDUAL16','formula':'x+W2*ReLU(W1*x+b1)','W1':[16,8], 'W1_bias':True,'W2':[8,16],
      'W2_bias':False,'head':[16,1],'head_bias':True,'mapping_parameters':264,'head_parameters':17,
      'initialization':'construct float64 CPU W1,W2,head in order with default Linear initialization; zero b1,W2; project W1 once',
      'constraint':{'c':.5,'cap':2**-.5,'method':'exact singular value clipping after every mapping Adam update',
      'tolerance':1e-10,'moments':'unchanged','restore':'check bounded only, never reproject'},'no_other_layers':True}
    p['C_loss']='pure mean((head(h(x))-y_std)^2); no embedding norm or residual penalty'
    p['branches']={'R_FROZEN':[{'mode':'gp','updates':1000}], 'R_REG':[{'mode':'regression','updates':1000},{'mode':'gp','updates':1000}],
       'R_DKL_C':[{'mode':'joint','updates':1000,'constraint_mode':'bounded'}],
       'R_DKL_U':[{'mode':'joint','updates':1000,'constraint_mode':'unbounded'}],'RAW16':p['branches']['RAW16']}
    p['historical_diagnostics']={'states':['RAW_X','COMMON_REG','COMMON_GP','OLD_DKL_1000','OLD_DKL_4000','OLD_RAW16'],
       'shape':'||D_b/s_b-D_a/s_a||F / max(||D_a/s_a||F,1e-12)', 's_h':'median strictly positive off-diagonal TT distance; constant -> null',
       'E1':'100*mean absolute query versus nearest selected train y difference','E5':'100*mean query mean5 absolute y differences',
       'no_scoring_before_unlabelled_seal':True,'COMMON_REG_GP_same_mapping_required':True,
       'rho_auxiliary':'rho>0.99 and absolute truth difference>5 pp; never a gate'}
    p['gate']={'tol_pp':1e-9,'states':['COMMON_REG','OLD_DKL_1000'],'n':[20,50],
       'mapping_support':'fivefold mean(E1mapped-E1raw)>tol and at least4 positive folds',
       'route':'any of four comparisons passes -> full B; none -> DIAGNOSTIC_ONLY_COMPLETE'}
    p['diagnostics'].update(final_blocks='0,100,...1000; postupdate NLL/prediction/kernel; REG interim regression no GP science',
       stability={'windows':[700,800,900,1000],'nll_relative':.001,'TT_and_combined_K_relative':.01,
       'train_query_mean_max_raw':.001,'latent_model_sd_max_raw':.001},K_R_separate=True)
    p['budget'].update(cumulative_fit_predict_seconds=1200,includes=['formal training/projection','A/B new model forward/predict/diagnostics',
      'all synthetic/real model verification','failures/repairs'],excludes=['pure statistics of saved values','IO','implementation','packaging'],
      estimate='first retained n20/n50 contexts; best .8x expected 1x worst 1.25x remaining estimated cost')
    p['GP']['fast_pred_var']=False
    p['metrics']['baselines']=['R_REG','R_FROZEN','RAW16','exact uniform random5 from same query fold']
    p['metrics'].update(main='R_REG regret5 minus R_DKL_C',mechanism='R_DKL_U regret5 minus R_DKL_C',
      gain5='100*max(0,best_selected-best_training)',break_best='best_selected-best_training>1e-11',headroom='100*max(0,best_query-best_training)',
      random_gain='expectation of max(0,order_stat-best_training)',IS90='100*(width+20*(lower-y)Ibelow+20*(y-upper)Iabove)',
      variance_observed='y_scale^2*(max(raw_latent_std,0)+extra_variance_std)+query_SE^2')
    p['classification']={'tol_pp':1e-9,'selection_signal':'mean Delta_REG>tol; >=4/5 positive folds; C better than RAW and exactrandom by tol',
      'otherwise':'any positive REG fold -> mixed; else no benefit','mechanism':'mean Delta_U>tol and >=4/5positive folds',
      'followup_candidate':'selection_signal and gain vs REG/RAW/random not worse (tol_pp); break probability not worse (1e-12); both sample-seed Delta_REG>0 and >=2/3 model-seed Delta_REG>0',
      'incomplete':'not_run or incomplete; no classification'}
    p['prediction']['score_after_uniform_seal']=True
    p['failure']='source missing BLOCKED; invalid math/leak INVALID; budget PARTIAL_BUDGET; numeric failure stop; no score-exposed B retraining'
    write(root/'protocol.json',p)
    write(root/'preparation.json',{'prepared':True,'source_files':len(files),'contexts':60,'endpoints_if_admitted':260,
       'formal_updates':0,'query_counts':[len(f['query_ids']) for f in folds],'A_labels_decoded':False,'at_utc':now()})
    print(json.dumps({'prepared':True,'protocol_sha256':sha(root/'protocol.json'),'source_files':len(files)}))

def freeze(root):
    pre=read(root/'verification/preflight_summary.json')
    if not pre['pass']:raise ValueError('Preflight failed')
    codes={str(p.relative_to(root)):sha(p) for p in sorted((root/'code').rglob('*.py'))}
    write(root/'code_manifest.json',{'files':codes})
    write(root/'environment_frozen.json',{'python':sys.version,'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['torch','gpytorch','numpy']}})
    names=['PROMPT.md','protocol.json','source_manifest.json','code_manifest.json','environment_frozen.json','worklist.json',
      'folds.json','subsets.json','public_inputs.json','verification/preflight_summary.json']
    names += [str(p.relative_to(root)) for folder in ['queries','training'] for p in sorted((root/folder).glob('*.json'))]
    write(root/'frozen_manifest.json',{'files':{n:sha(root/n) for n in names},'frozen_at_utc':now(),
       'new_A_scoring_exposed':False,'formal_updates':0})
    print(json.dumps({'frozen':True,'protocol_sha256':sha(root/'protocol.json'),'code_manifest_sha256':sha(root/'code_manifest.json')}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','freeze']);ap.add_argument('--root',required=True)
    args=ap.parse_args();globals()[args.command](Path(args.root))
