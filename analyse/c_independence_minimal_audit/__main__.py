import argparse
import shlex,sys
from pathlib import Path
from .scope import freeze
p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','report','package']);p.add_argument('--run-dir',required=True);p.add_argument('--prompt');p.add_argument('--resume',action='store_true');p.add_argument('--acceptance')
a=p.parse_args();repo=Path.cwd();root=Path(a.run_dir).resolve()
from .common import read,write
def command_log():
 log=root/'actual_commands.json';entries=read(log) if log.exists() else []
 entries.append(shlex.join([sys.executable,'-B','-m','analyse.c_independence_minimal_audit',*sys.argv[1:]]));write(log,entries)
if a.command=='freeze':freeze(root,repo,a.prompt)
elif a.command=='run':
 command_log()
 from .audit import run
 run(root,repo,a.resume,Path(a.acceptance) if a.acceptance else None)
else:
 command_log()
 from .report import report,package
 (report if a.command=='report' else package)(root,repo)
