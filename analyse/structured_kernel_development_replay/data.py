"""Public A preparation and immutable provenance; never decode performance."""
import ast
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import gpytorch
import botorch
from analyse.structured_kernel_phase1.io import read_json, write_json, sha256, digest, now
from analyse.structured_kernel_phase1.runner import verify_frozen, capture_baseline
from analyse.structured_kernel_phase1.data import SOURCE, A_ID_HASH, A_LABEL_HASH, RECIPE_HASH
from analyse.structured_kernel_phase1.features import build_operator_graph, graph_hash, directed_wl
from .protocol import protocol, eligibility, METHODS, SEEDS, PHASE1, OLD


def initial_ids(ids, seed):
    return np.random.default_rng(np.random.SeedSequence([seed,0])).permutation(sorted(ids)).tolist()[:50]


def code_files(repo):
    paths = sorted((repo/'analyse/structured_kernel_development_replay').glob('*.py'))
    paths += sorted((repo/'tests/structured_kernel_development_replay').glob('*.py'))
    return {str(p.relative_to(repo)):sha256(p) for p in paths}


def preflight(root, repo, prompt):
    if root.exists() and any(root.iterdir()):
        raise ValueError('New empty run directory required')
    assert eligibility('development_replay','A179') == 'ELIGIBLE'
    frozen = verify_frozen(repo/PHASE1,repo)
    old = read_json(repo/OLD/'status.json')
    if old['state'] != 'BLOCKED' or old['B_revealed']:
        raise ValueError('Old blocked task identity changed')
    public = read_json(repo/PHASE1/'public_features.json')
    ids = public['ids']
    if len(ids)!=179 or digest(sorted(ids))!=A_ID_HASH or len(set(public['architecture_ids']))!=96:
        raise ValueError('A pool identity mismatch')
    cp = repo/SOURCE/'candidate_identity.json'
    if sha256(cp)!='c66c2aa70c4cf2d5fe7d6462bd822a35c4fdd64e950b326cfce8fcf013cab46e':
        raise ValueError('A complete configuration source changed')
    configs = {r['candidate_id']:r for r in read_json(cp)}
    if set(configs)!=set(ids):
        raise ValueError('Configuration ID set mismatch')
    graphs = {}
    for cid,aid,x,wl in zip(ids,public['architecture_ids'],public['x_raw'],public['wl']):
        c=configs[cid]
        if c['architecture_fingerprint']!=aid:
            raise ValueError('Architecture identity mismatch')
        graph=build_operator_graph(c)
        if graph_hash(graph)!=public['graph_hashes'][aid]:
            raise ValueError('Operator graph changed')
        hp=[np.log10(c['lr']),c['dropout'],np.log2(c['hidden_dim']),np.log10(c['l2'])]
        if not np.allclose(hp,x[12:],atol=1e-14,rtol=0):
            raise ValueError('HP transformation mismatch')
        graphs[aid]=graph
    wl,manifest=directed_wl(graphs)
    if not all(np.array_equal(wl[a],v) for a,v in zip(public['architecture_ids'],public['wl'])):
        raise ValueError('Frozen WL representation mismatch')
    root.mkdir(parents=True)
    write_json(root/'protocol.json',protocol())
    shutil.copyfile(prompt,root/'PROMPT.txt')
    write_json(root/'public_features.json',public)
    write_json(root/'public_configurations.json',configs)
    write_json(root/'initial_observation_manifest.json',
        {str(s):{'ids':initial_ids(ids,s),'ids_sha256':digest(initial_ids(ids,s))} for s in SEEDS})
    write_json(root/'baseline.json',capture_baseline(repo))
    source_paths=[repo/PHASE1/'frozen_manifest.json', repo/PHASE1/'public_features.json',
        repo/PHASE1/'feature_manifest.json',cp,repo/SOURCE/'recipe_verified.json',
        repo/OLD/'status.json',repo/OLD/'protocol.json',
        repo/'results/posthoc/fuzzy-metric-stage2-20260910-v1/partition.json',
        repo/'results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2/data_use.json',
        repo/'surrogate/accuracy_gp.py', repo/'bo_phase4.py',repo/'eval_utils.py']
    source={str(p.relative_to(repo)):sha256(p) for p in source_paths}
    source.update(frozen['code_files'])
    # Performance bytes are opened only by the trusted oracle/scorer. Their
    # frozen whitelist hash participates in identity without preflight access.
    source_manifest={'files':source,'A_label_sha256':A_LABEL_HASH,'recipe_identity':RECIPE_HASH,
        'A_label_entrypoint':'analyse.structured_kernel_phase1.data.read_A_labels',
        'phase1_identity':frozen['identity'],'old_independent_confirmation':'BLOCKED',
        'pool':'A179','independent_confirmation':False,'B_label_SE_reads':0,'node_test_reads':0}
    write_json(root/'source_manifest.json',source_manifest)
    write_json(root/'environment.json',{'python':sys.version,'executable':sys.executable,
        'torch':torch.__version__,'gpytorch':gpytorch.__version__,'botorch':botorch.__version__,
        'numpy':np.__version__,'CPU':True,'threads':1,'dtype':'float64'})
    for path in (repo/'analyse/structured_kernel_development_replay').glob('*.py'):
        ast.parse(path.read_text())
    subprocess.run(['git','diff','--check'],cwd=repo,check=True)
    inputs={p.name:sha256(p) for p in sorted(root.iterdir()) if p.is_file()}
    identity={'code':code_files(repo),'inputs':inputs,'sources':source,'created_at':now()}
    identity['run_identity']=digest(identity)
    write_json(root/'run_identity.json',identity)
    write_json(root/'status.json',{'state':'PENDING','run_identity':identity['run_identity'],
        'purpose':'development_replay','independent_confirmation':False,'updated_at':now()})
    (root/'purpose_change.md').write_text('A 开发回放；B 未使用；无独立确认。\n\n'
        '旧协议把历史开发 B 视为独立确认集，前提不成立。用户显式修订研究用途为 A179 开发回放。'
        '原独立确认任务保持 BLOCKED，原结果、协议、恢复命令均不修改。新协议不改写历史身份。\n\n'
        f'新目录：{root}\nrun_identity：{identity["run_identity"]}\n')
    return identity


def verify(root, repo):
    frozen=read_json(root/'run_identity.json')
    if digest({k:v for k,v in frozen.items() if k!='run_identity'})!=frozen['run_identity']:
        raise ValueError('Run identity corruption')
    if frozen['code']!=code_files(repo):
        raise ValueError('Code changed: use a new run, never mix optimization versions')
    for base,files in ((root,frozen['inputs']),(repo,frozen['sources'])):
        for relative,expected in files.items():
            if sha256(base/relative)!=expected:
                raise ValueError('Frozen source/input changed: '+relative)
    return frozen
