import argparse,shlex,sys
from .common import *
p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','collect','review','report','package']);p.add_argument('--run-dir');p.add_argument('--prompt');p.add_argument('--acceptance');p.add_argument('--resume',action='store_true');a=p.parse_args();repo=Path.cwd()
if a.command=='freeze':
 from .scope import freeze
 freeze(repo,a.prompt)
else:
 root=Path(a.run_dir).resolve();log(root/'actual_commands.jsonl',{'at':now(),'command':shlex.join([sys.executable,'-B','-m','analyse.c_provenance_evidence_closure',*sys.argv[1:]])})
 if a.command=='collect':
  from .collect import collect
  collect(root,repo,a.resume,Path(a.acceptance) if a.acceptance else None)
 elif a.command=='review':
  from .review import review
  review(root,repo,a.resume)
 else:
  from .report import report,package
  (report if a.command=='report' else package)(root,repo)
