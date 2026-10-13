import argparse,shlex,sys
from pathlib import Path
from .common import log,now
p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','collect','review','freeze-performance','predict','train','search','score','test','all','report','package','_train_one','_oracle','_search_worker']);p.add_argument('--run-dir',required=True);p.add_argument('--prompt');p.add_argument('--resume',action='store_true');p.add_argument('--task-id')
a=p.parse_args();root=Path(a.run_dir).resolve();repo=Path.cwd()
if a.command=='freeze':
 from .scope import freeze
 freeze(root,repo,a.prompt)
elif a.command=='collect':
 from .collect import collect
 collect(root,repo,a.resume)
elif a.command=='review':
 from .review import review
 review(root,repo,a.resume)
elif a.command in ('freeze-performance','predict','train','_train_one'):
 from .performance import freeze_performance,seal_predictions,build_cache,train_one
 if a.command=='_train_one':train_one(root,repo,a.task_id)
 else:{'freeze-performance':freeze_performance,'predict':seal_predictions,'train':build_cache}[a.command](root,repo)
elif a.command=='_oracle':
 from .oracle import serve
 serve(root)
elif a.command in ('search','_search_worker'):
 from .search import run_search,worker
 (worker if a.command=='_search_worker' else run_search)(root,repo)
elif a.command in ('score','test'):
 from .analysis import score,final_test
 (score if a.command=='score' else final_test)(root,repo)
elif a.command=='all':
 from .runner import run
 run(root,repo,a.resume)
else:
 from .report import report,package
 (report if a.command=='report' else package)(root,repo)
if not a.command.startswith('_'):log(root/'actual_commands.jsonl',{'at':now(),'command':shlex.join([sys.executable,'-B','-m','analyse.c_independent_validation',*sys.argv[1:]])})
