import json
from pathlib import Path
from analyse.structured_kernel_phase1.io import read_json,write_json,write_csv,append_jsonl,sha256,digest,now

REPLAY='results/structured_kernel_development_replay_20261002T235614Z'
PHASE1='results/structured_kernel_phase1_20261002T142841Z'
BLOCKED='results/structured_kernel_phase2_20261002T161300Z'
ORIGIN='results/search/high-regime-prior-model-blind-validation-global4-v1'
PREP='results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2'
CHAMPION='30e0bb108c1fdbe8db186152a09c3f2fc6e812be189db920fe4d0d2b0d3cb1a2'
CONFIG_IDS=sorted(['23b5ff1428854b6b957495d7ad3a607c45fdaea741018b861d423315ed7ba8ff',CHAMPION,
    '5f66345ac001edb2df34c99cd1cdb7f528617ee55942bb734200b581f26a34c4',
    '97b65e168894d9d8318fe44e02a46de825600ed6e33aef9f38cde8f0dc198c17',
    'caddc125138445e794152018b31e47e2102e69d83e554fbca50079f3b78a6394'])
SEEDS=list(range(2026100200,2026100210))
M1_SEEDS=[2026100201,2026100205,2026100200,2026100203]


def models():
    return [(s,'W-I',n) for s in M1_SEEDS for n in range(50,71)]+[(s,'RAW16',50) for s in M1_SEEDS[:2]]


def seal(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Refusing to overwrite sealed result: '+str(path))
    else:write_json(path,value)


def set_status(root,module,state,**extra):
    p=root/'module_status.json';status=read_json(p)
    status['modules'][module]={'state':state,'updated_at':now(),**extra}
    write_json(p,status)


def code_manifest(repo):
    paths=sorted((repo/'analyse/structured_kernel_followup').glob('*.py'))
    paths+=sorted((repo/'tests/structured_kernel_followup').glob('*.py'))
    return {str(p.relative_to(repo)):sha256(p) for p in paths}


def verify(root,repo):
    frozen=read_json(root/'run_identity.json')
    if frozen['identity']!=digest({k:v for k,v in frozen.items() if k!='identity'}):raise ValueError('Run identity changed')
    if frozen['code']!=code_manifest(repo):raise ValueError('Code identity changed; use a new run')
    for base,files in [(root,frozen['inputs']),(repo,frozen['sources'])]:
        for rel,h in files.items():
            if sha256(base/rel)!=h:raise ValueError('Source identity changed: '+rel)
    return frozen
