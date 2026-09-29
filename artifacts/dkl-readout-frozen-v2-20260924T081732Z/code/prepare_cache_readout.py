#!/usr/bin/env python3
"""Root-only serial, one-shot frozen-feature preparation and CONST0 validation.

No training or scoring entry exists. Every new numeric operation is charged to
verification in budget.json, including failed calls; file I/O is excluded.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import time
import traceback
from pathlib import Path
import torch
from readout_core import budget_limit
from readout_core import Budget,BudgetExceeded,ReadoutModel,cpu_setup
from readout_data import prepare_context,verify_cached_context,verify_gp_const_zero
from readout_diagnostics import diagnose


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    temp=p.with_suffix('.tmp');torch.save(v,temp);temp.replace(p)


def main(root):
    root=Path(root)
    if (root/'cache_manifest.json').exists() or (root/'verification/source_cache_checks.json').exists():
        raise RuntimeError('Cache preparation is one-shot; completed/failed evidence cannot be overwritten')
    lock=root/'cache_precompute_once.lock';lock.mkdir()
    budget=Budget(root/'budget.json',limit=budget_limit(root));before=budget.seconds;wall=time.perf_counter()
    files={};cases=[];failure=None;current=None;repeat=None;status='PASS';source=None
    def compute(fn):
        budget.check();tick=time.perf_counter()
        try:result=fn()
        finally:
            budget.charge(time.perf_counter()-tick,kind='verification');budget.persist()
        budget.check();return result
    def checked(relative):
        p=root/relative
        if relative not in source['files'] or not p.is_file() or sha(p)!=source['files'][relative]:
            raise ValueError('BLOCKED_SOURCE_INCOMPLETE: '+relative)
        return p
    def make_record(cache):
        verify_cached_context(cache)
        t,q=cache['training'],cache['query'];sc=t['scaler']
        model=ReadoutModel('GP_CONST',t['h0'],t['s0'],t['y_standardized'],t['known_variance_standardized'],
            cache['common_gp_state'],source_fingerprint=cache['identity']['source_fingerprint'])
        record=diagnose(model,q['h0'],q['s0'],q['ids'],sc['y_mean'],sc['y_scale'],0)
        check=verify_gp_const_zero(record,cache['old_reference'])
        verify_cached_context(cache)
        return record,check
    try:
        source=read(root/'source_manifest.json');contexts=read(root/'worklist.json')['contexts']
        if len(contexts)!=60 or [c['context_id'] for c in contexts]!=sorted(c['context_id'] for c in contexts):
            raise ValueError('Fixed 60-context lexical worklist required')
        for rel in source['files']:checked(rel)
        compute(lambda:cpu_setup(0))
        for index,c in enumerate(contexts):
            current=c['context_id'];sid=c['subset_id']
            state_rel=f'source/common/{current}_COMMON_gp.pt';diag_rel=f'source/common_diagnostics/{current}_j300.json'
            state=torch.load(checked(state_rel),map_location='cpu',weights_only=False)
            tr=read(checked(f'training/{sid}.json'));q=read(checked(f'queries/fold{c["fold"]}.json'));old=read(checked(diag_rel))
            start_cost=budget.seconds
            cache=compute(lambda:prepare_context(state,tr,q,c,old,source_sha256=source['files'][state_rel],diagnostic_sha256=source['files'][diag_rel]))
            zero,check=compute(lambda:make_record(cache))
            if index==0:
                # Restore from the saved state again rather than reuse a live
                # mapping instance; no training occurs in either preparation.
                state2=torch.load(checked(state_rel),map_location='cpu',weights_only=False)
                repeated=compute(lambda:prepare_context(state2,tr,q,c,old,source_sha256=source['files'][state_rel],diagnostic_sha256=source['files'][diag_rel]))
                repeated_zero,repeated_check=compute(lambda:make_record(repeated))
                def repeat_check():
                    if repeated['identity']!=cache['identity'] or repeated_zero!=zero or not repeated_check['pass']:
                        raise RuntimeError('Repeated saved-state cache/prediction changed')
                    return {'pass':True,'context_id':current,'cache_identity_exact':True,'diagnostic_exact':True,'optimizer_updates':0}
                repeat=compute(repeat_check)
            cache_rel=f'caches/{current}.pt';identity_rel=f'identity/{current}.json'
            zero_rel=f'verification/common_gp_zero/{current}.json'
            save(root/cache_rel,cache);write(root/identity_rel,cache['identity']);write(root/zero_rel,zero)
            files[cache_rel]=sha(root/cache_rel);files[identity_rel]=sha(root/identity_rel)
            cases.append({'context_id':current,'subset_id':sid,'n':c['n'],'pass':True,
                'const_zero':check,'cache_sha256':cache['identity']['cache_sha256'],
                'cache_path':cache_rel,'cache_file_sha256':files[cache_rel],
                'zero_diagnostic_path':zero_rel,'zero_diagnostic_sha256':sha(root/zero_rel),
                'verification_seconds':budget.seconds-start_cost})
            print(json.dumps({'source_contexts_verified':len(cases),'of':60,'model_seconds':budget.seconds,'limit':budget.limit}),flush=True)
        if len(cases)!=60 or len(files)!=120 or budget.formal_updates!=0:
            raise RuntimeError('Complete source cache/count invariant')
        for rel in source['files']:checked(rel)
        budget.check()
    except Exception as exc:
        status='STOPPED_BUDGET_PARTIAL' if isinstance(exc,BudgetExceeded) else 'FAILED'
        failure={'type':type(exc).__name__,'message':str(exc),'context_id':current,'traceback':traceback.format_exc()}
    finally:
        budget.persist()
    report={'pass':status=='PASS','status':status,'contexts_expected':60,'contexts_passed':len(cases),
        'cases':cases,'repeated_saved_state_check':repeat,'failure':failure,'optimizer_updates':0,
        'query_labels_or_SE_present':False,'charged_verification_seconds':budget.seconds-before,
        'wall_seconds':time.perf_counter()-wall,'budget':budget.as_dict(),'finished_at_utc':now()}
    write(root/'verification/source_cache_checks.json',report)
    if report['pass']:
        write(root/'cache_manifest.json',{'files':files,'context_count':60,'cache_count':60,'identity_count':60,
            'protocol_sha256':sha(root/'protocol.json'),'source_manifest_sha256':sha(root/'source_manifest.json'),
            'worklist_sha256':sha(root/'worklist.json'),'source_cache_checks_sha256':sha(root/'verification/source_cache_checks.json'),
            'frozen_at_utc':now(),'new_optimizer_updates':0,'query_labels_or_SE_present':False})
    print(json.dumps({'source_cache_pass':report['pass'],'contexts':len(cases),'model_seconds':budget.seconds,'failure':failure}),flush=True)
    return 0 if report['pass'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    raise SystemExit(main(parser.parse_args().root))
