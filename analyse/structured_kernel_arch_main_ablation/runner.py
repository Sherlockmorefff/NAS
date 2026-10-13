import fcntl
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from .common import *


def counters(root):
    fits=list((root/'trajectories').glob('*/fits/n*/fit.json'));starts=list((root/'trajectories').glob('*/fits/n*/start*.json'))
    result={'formal_fits':len(fits),'logical_starts':len(starts),'physical_start_attempts':0,'adam_updates':0,'LBFGS_closures':0,'A_new_queries':0,'A_oracle_calls':0,'C_decoder_calls':0}
    for p in (root/'trajectories').glob('*/fits/n*/start*.jsonl'):
        for line in p.read_text().splitlines():
            e=json.loads(line)
            if e.get('event')=='physical_attempt':result['physical_start_attempts']+=1
            if e.get('stage')=='adam':result['adam_updates']+=1
            if e.get('stage')=='lbfgs':result['LBFGS_closures']+=1
    p=root/'oracle.sqlite'
    if p.exists():
        db=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
        result['A_new_queries']=db.execute('SELECT COUNT(*) FROM observations WHERE slot>=50').fetchone()[0]
        result['A_oracle_calls']=db.execute('SELECT COUNT(*) FROM calls').fetchone()[0];db.close()
    p=root/'C_preparation/generation.sqlite'
    if p.exists():
        db=sqlite3.connect(f'file:{p}?mode=ro',uri=True);result['C_decoder_calls']=db.execute('SELECT COUNT(*) FROM calls').fetchone()[0];db.close()
    return result


def completed(root,module):
    path=root/f'{module}_completion.json'
    if not path.exists():return False
    value=read_json(path)
    if value['run_identity']!=read_json(root/'run_identity.json')['identity']:raise ValueError('Module identity changed')
    for p,h in value['files'].items():
        if sha256(root/p)!=h:raise ValueError('Sealed module changed: '+p)
    return True


def seal_module(root,module,paths):
    write_json(root/f'{module}_completion.json',{'run_identity':read_json(root/'run_identity.json')['identity'],
        'files':{str(p.relative_to(root)):sha256(p) for p in sorted(paths) if p.is_file() and p.suffix not in ('.tmp','.sqlite')},'time':now()})


def child(root,repo,command,extra=(),log=None):
    log=log or root/(command+'.log')
    args=[sys.executable,'-B','-m','analyse.structured_kernel_arch_main_ablation',command,'--run-dir',str(root),*extra]
    with log.open('a') as out:
        p=subprocess.run(args,cwd=repo,stdout=out,stderr=subprocess.STDOUT)
    if p.returncode:
        append_jsonl(root/'errors.jsonl',{'command':args,'exit_code':p.returncode,'log_tail':log.read_text()[-5000:],'time':now()})
    return p.returncode


def run_A(root,repo):
    from .learner import validate_complete
    if completed(root,'A'):
        for seed in SEEDS:validate_complete(root/'trajectories'/f'{seed}_W-HI')
        return
    verify(root,repo,'A');status(root,'A','RUNNING');failed=[];identity=read_json(root/'run_identity.json')['identity'];start=time.perf_counter()
    for seed in SEEDS:
        folder=root/'trajectories'/f'{seed}_W-HI'
        if (folder/'complete.json').exists():validate_complete(folder);continue
        code=child(root,repo,'_learner',('--seed',str(seed)),root/f'learner_{seed}.log')
        if code:failed.append(seed)
        else:validate_complete(folder)
        n=len(list((root/'trajectories').glob('*/complete.json')))
        print(json.dumps({'module':'A','completed_trajectories':n,'failed_seeds':failed,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    if failed:status(root,'A','INCOMPLETE',failed_seeds=failed);return
    files=list((root/'trajectories').glob('*/complete.json'))+[root/'oracle_summary.json',root/'oracle_release_manifest.json',root/'oracle_access_log.jsonl']
    seal_module(root,'A',files);status(root,'A','COMPLETE',trajectories=10,fits=210,starts=630)


def run(root,repo,resume=False):
    with (root/'controller.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An ablation controller is already running')
        before=counters(root);names=['analysis/summary.json','method_selection.json','C_preparation/summary.json','C_preparation/C_unlabeled_manifest.json']
        hashes={n:sha256(root/n) for n in names if (root/n).exists()}
        state=read_json(root/'module_status.json')['modules']
        if state['A']['state']!='BLOCKED':
            try:run_A(root,repo)
            except Exception as exc:status(root,'A','INCOMPLETE',error=repr(exc));append_jsonl(root/'errors.jsonl',{'module':'A','error':repr(exc),'time':now()})
        if completed(root,'A') and not completed(root,'analysis'):
            code=child(root,repo,'analyze')
            if code:status(root,'analysis','INCOMPLETE',worker_exit=code)
            elif (root/'analysis/summary.json').exists() and read_json(root/'analysis/summary.json')['state']=='COMPLETE':
                seal_module(root,'analysis',[p for p in (root/'analysis').rglob('*') if p.is_file()]+[root/'method_selection.json'])
        elif not completed(root,'A'):
            status(root,'analysis','INCOMPLETE',reason='All10 formal trajectories must be sealed before Q truth scoring')
            write_json(root/'method_selection.json',{'method_selection':'INCONCLUSIVE','recommended_method':'W-I','default_only':True,'requires_independent_confirmation':True,'reason':'Incomplete A; no partial mean used'})
        if state['C']['state']!='BLOCKED_C' and not completed(root,'C'):
            code=child(root,repo,'prepare-C')
            if code:status(root,'C','BLOCKED_C',worker_exit=code)
            elif (root/'C_preparation/summary.json').exists():
                seal_module(root,'C',[p for p in (root/'C_preparation').rglob('*') if p.is_file() and p.suffix!='.sqlite'])
        if resume:
            after=counters(root);remaining={n:sha256(root/n) for n in names if (root/n).exists()}
            added={k:after[k]-before[k] for k in before}
            write_json(root/'resume_verification.json',{'before':before,'after':after,'additional':added,'primary_hashes_before':hashes,'primary_hashes_after':remaining,
                'pass':all(v==0 for v in added.values()) and hashes==remaining and len(hashes)==len(names),'time':now()})
