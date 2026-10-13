import json
from pathlib import Path
from analyse.structured_kernel_phase1.io import read_json,write_json,write_csv,append_jsonl,sha256,digest,now
from analyse.structured_kernel_followup.common import PHASE1,REPLAY,ORIGIN,PREP,BLOCKED,CHAMPION
FOLLOWUP='results/structured_kernel_followup_20261005T152816Z'
SEEDS=tuple(range(2026100200,2026100210))
METHODS=('W-HI',)
KERNEL_KIND='hp_plus_interaction'


def seal(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Sealed artifact changed: '+str(path))
    else:write_json(path,value)


def code_files(repo):
    paths=sorted((repo/'analyse/structured_kernel_arch_main_ablation').glob('*.py'))+sorted((repo/'tests/structured_kernel_arch_main_ablation').glob('*.py'))
    return {str(p.relative_to(repo)):sha256(p) for p in paths}


def status(root,module,state,**extra):
    d=read_json(root/'module_status.json');d['modules'][module]={'state':state,'updated_at':now(),**extra};write_json(root/'module_status.json',d)


def verify(root,repo,module=None):
    frozen=read_json(root/'run_identity.json')
    if frozen['identity']!=digest({k:v for k,v in frozen.items() if k!='identity'}):raise ValueError('Run identity corruption')
    if frozen['code']!=code_files(repo):raise ValueError('Frozen experiment code changed; cannot mix versions')
    for rel,h in frozen['inputs'].items():
        if sha256(root/rel)!=h:raise ValueError('Frozen input changed: '+rel)
    domains=() if module=='shared' else (('A','C') if module is None else (('A',) if module in ('A','analysis') else ('C',)))
    for domain in domains:
        for rel,h in frozen['sources'][domain].items():
            if sha256(repo/rel)!=h:raise ValueError(domain+' source changed: '+rel)
    return frozen
