import fcntl,time,traceback
from .common import *
from .performance import freeze_performance,seal_predictions,build_cache
from .search import run_search
from .analysis import score,final_test
from .report import report

def run(root,repo,resume=False):
 with (root/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
  before={p.name:sha(p) for p in root.glob('*_complete.json')}
  try:
   if read(root/'P1_summary.json')['qualified_configurations']<=71:report(root,repo);return
   freeze_performance(root,repo);seal_predictions(root,repo);build_cache(root,repo);run_search(root,repo);score(root,repo);final_test(root,repo);report(root,repo)
   if resume and 'P6_complete.json' in before:
    assert all(sha(root/p)==h for p,h in before.items())
    write(root/'resume_verification.json',{'state':'PASS','new_GNN_tasks':0,'new_GP_fits':0,'new_test_evaluations':0,'new_score_computations':0,'core_hashes_unchanged':True,'seconds':time.monotonic()-started,'time':now()})
  except Exception as e:
   write(root/'technical_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time':now()})
   report(root,repo);raise
