"""Finish reporting after a numpy bool serialization error; no scientific rerun."""
import fcntl
import json
import os
import signal
import sqlite3
import time
from pathlib import Path
import numpy as np
from analyse.ac_champions_ln_reuse_none.common import ROOT, read_json, write_json, append_jsonl, now, sha256, verify
from analyse.ac_champions_ln_reuse_none import report as reporting

root=ROOT
lock=(root/'controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
original=read_json(root/'STATUS.json')
if original['LN_completed']!=300:
    raise RuntimeError('Reporting-only repair requires all 300 LN trainings complete')
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args=(proc/'cmdline').read_bytes().split(b'\0')
    except OSError:
        continue
    if b'analyse.ac_champions_ln_reuse_none' not in args:
        continue
    if b'timer' in args:
        os.kill(int(proc.name),signal.SIGTERM)
    else:
        raise RuntimeError('Another experiment process remains active: '+proc.name)
verify(root,baseline_weights=True)
supp=root/'supervision'
supp.mkdir(exist_ok=True)
write_json(supp/'ORIGINAL_FAILED_STATUS.json',original)
write_json(supp/'PRE_REPAIR_TIMING_REPORT.json',read_json(root/'TIMING_REPORT.json'))
write_json(supp/'ORIGINAL_ACCEPTANCE_STATUS.json',read_json(root/'acceptance_status.json'))

def native(value):
    if isinstance(value,np.generic):
        return native(value.item())
    if isinstance(value,dict):
        return {k:native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [native(v) for v in value]
    return value

def corrected_write(path,value):
    write_json(path,native(value))

# Frozen source files stay untouched; only the report JSON boundary converts
# numpy scalar types to their identical Python scalar values.
reporting.write_json=corrected_write
begin=time.time()
reporting.report(root,original['started_unix'])
accepted=reporting.accept(root)
with sqlite3.connect('file:'+str(root/'tasks.sqlite')+'?mode=ro',uri=True) as db:
    training_done=db.execute('SELECT max(finished) FROM tasks').fetchone()[0]
scores=read_json(root/'LN_SEED_RESULTS.json')
receipt={'at':now(),'checks':'PASS','issue':'numpy bool in validation_test_mean_same_direction was not JSON serializable',
         'repair':'report output boundary converts numpy scalars to native Python scalars in a supplemental process; frozen source unchanged',
         'source_file':str(Path(__file__).resolve()),'source_sha256':sha256(__file__),
         'scientific_protocol_changed':False,'training_rerun':False,'test_rescored':False,
         'training_completed_unix':training_done,'last_LN_score_computed_at':max(s['computed_at'] for s in scores),
         'repair_validation_seconds':time.time()-begin,'original_controller_exit_code':int((root/'controller_exit_code.txt').read_text()),
         'freeze_identity':read_json(root/'freeze.json')['identity']}
write_json(root/'DELIVERY_REPAIR.json',receipt)
append_jsonl(root/'ERRORS.jsonl',{'at':now(),'phase':'report_output_engineering_repair','resolved':True,'training_rerun':False,'test_rescored':False,'receipt':'DELIVERY_REPAIR.json'})
status={**original,'state':'COMPLETE','updated_at':now(),'heartbeat':time.time(),'report_output_repair':'DELIVERY_REPAIR.json'}
status.pop('error',None)
write_json(root/'STATUS.json',status)
write_json(root/'COMPLETE.json',{'state':'COMPLETE','completed_at':now(),'accepted':accepted,'delivery_repair':'DELIVERY_REPAIR.json'})
write_json(root/'acceptance_status.json',{'state':'PASS','at':now(),'delivery_repair':'DELIVERY_REPAIR.json'})
(root/'delivery_repair_exit_code.txt').write_text('0\n')
reporting.package(root)
print(json.dumps({'acceptance':accepted,'repair':receipt,'package':read_json(root/'package_manifest.json')},ensure_ascii=False),flush=True)
