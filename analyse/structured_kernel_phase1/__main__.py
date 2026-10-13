from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path
from .io import append_jsonl,now,write_json


def main():
    parser=argparse.ArgumentParser(description="Frozen E0-E2 CPU experiment")
    parser.add_argument("command",choices=("preflight","run","diagnose","score","report","package"))
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--pilot-then-all",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    root=args.run_dir.resolve();repo=args.repo.resolve()
    try:
        if args.command=="preflight":
            from .runner import preflight
            preflight(root,repo)
        elif args.command=="run":
            from .runner import run
            run(root,repo,args.pilot_then_all,args.resume,args.workers)
        elif args.command=="diagnose":
            from .legacy import extract_legacy_diagnostics
            extract_legacy_diagnostics(root,repo)
        elif args.command=="score":
            from .score import score_frozen
            score_frozen(root,repo)
        elif args.command=="report":
            from .report import build_report
            build_report(root,repo)
        elif args.command=="package":
            from .report import package_run
            package_run(root,repo)
    except Exception as exc:
        if root.exists():
            write_json(root/f"error_{args.command}.json",{"type":type(exc).__name__,"error":str(exc),
                       "traceback":traceback.format_exc(),"at":now()})
        raise
    finally:
        if root.exists():
            append_jsonl(root/"commands.jsonl",{"at":now(),"executable":sys.executable,
                         "argv":sys.argv,"cwd":str(Path.cwd())})


if __name__=="__main__":
    main()
