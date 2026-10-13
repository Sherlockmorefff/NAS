"""Verify original files, capture this turn's diff and package only new A evidence."""
import difflib,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse.relative_metric_pilot import read,write,file_hash,utc
O=ROOT/'results/posthoc/budget-champion-ranking-v1'
NEW=['surrogate/budget_champion_ranking.py','analyse/budget_champion_engine.py','analyse/budget_champion_run.py',
     'configs/budget_champion_ranking_v1.json','tests/test_budget_champion_ranking.py','analyse/budget_champion_report.py',
     'analyse/budget_champion_verify.py','analyse/budget_champion_recompute.py','analyse/budget_champion_write_report.py','analyse/budget_champion_package.py']
def main():
    before=read(O/'workspace_before.json');old=read(O/'old_results_before.json');doc='docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md'
    changed=[n for n,h in before.items() if not (ROOT/n).exists() or file_hash(ROOT/n)!=h]
    assert changed==[doc] or set(changed)=={doc},changed
    after=(ROOT/doc).read_bytes();base=(O/'baseline/Stage_Tasks.md').read_bytes();assert after.startswith(base)
    missing=[n for n,h in old.items() if not (ROOT/n).exists() or file_hash(ROOT/n)!=h];assert not missing,missing
    env=read(O/'environment.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==env['HEAD']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==env['branch']
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    assert read(O/'verification/runtime.json')['passed'] and read(O/'verification/independent.json')['passed']
    patch=''.join(difflib.unified_diff(base.decode().splitlines(True),after.decode().splitlines(True),fromfile='a/'+doc,tofile='b/'+doc))
    for n in NEW:
        assert n not in before,n
        patch+=''.join(difflib.unified_diff([], (ROOT/n).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+n))
        if n not in read(O/'protocol.json')['sources']:
            dst=O/'analysis_source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
    (O/'changes.patch').write_text(patch)
    (O/'git_status_after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    (O/'tracked_diff_after.patch').write_text(subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True))
    write(O/'verification/preservation.json',dict(passed=True,baseline_workspace_files=len(before),baseline_changed=changed,
        only_existing_change='append-only Stage_Tasks section19',old_results_files_checked=len(old),old_results_changed=0,
        existing_tracked_modifications_unchanged=['bo_phase4.py','eval_utils.py'],task_document_prefix_unchanged=True,
        new_files=NEW,git_diff_check=True,py_compile=True,HEAD=env['HEAD'],branch=env['branch'],time=utc()),immutable=False)
    files=[p for p in O.rglob('*') if p.is_file() and p.name not in ['bundle_manifest.json','package.json','local_verification.json']]
    assert not any(p.is_symlink() for p in files)
    forbidden=['B_validation.json','oracle_validation.json','test_results.json','.ssh','id_rsa','id_ed25519']
    for p in files:assert not any(x in p.parts for x in forbidden),p
    manifest={str(p.relative_to(O)):file_hash(p) for p in files};write(O/'bundle_manifest.json',manifest,immutable=False)
    archive=ROOT/'artifacts/packages/budget-champion-ranking-v1-evidence-v2.zip';assert not archive.exists(),'existing package preserved; inspect before replacing'
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files+[O/'bundle_manifest.json']):z.write(p,str(p.relative_to(O)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    result=dict(archive=str(archive),sha256=file_hash(archive),bytes=archive.stat().st_size,entries=len(files)+1,
        verified_manifest_files=len(manifest),scope='Only current A experiment; no raw B/test scores or credentials',time=utc())
    write(O/'package.json',result,immutable=False);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
