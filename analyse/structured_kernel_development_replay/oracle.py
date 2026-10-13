"""Trusted A-only process. No global performance table crosses its RPC boundary."""
import json
import sqlite3
import sys
from pathlib import Path
from analyse.structured_kernel_phase1.io import read_json, write_json, sha256, now
from analyse.structured_kernel_phase1.data import read_A_labels, SOURCE
from .protocol import METHODS, SEEDS


def access_guard(root, repo, role, counts=None):
    root,repo=Path(root).resolve(),Path(repo).resolve()
    allowed={repo/SOURCE/'features.json',repo/SOURCE/'private/evaluation_A.json'} if role in ('oracle','scorer') else set()
    label=repo/SOURCE/'private/evaluation_A.json'
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):
            return
        path=Path(args[0]).resolve()
        if path.is_relative_to(repo/'results') and not path.is_relative_to(root) and path not in allowed:
            raise PermissionError(f'{role}: historical performance/source access refused: {path.name}')
        if role=='learner' and path.is_relative_to(root) and (path.name.startswith('oracle') or
            path.name in ('source_manifest.json','summary.json','paired_comparisons.csv','README_RESULTS.md')):
            raise PermissionError('Learner cannot open oracle/scorer material')
        if path==label and counts is not None:
            counts['physical_A_label_file_opens']+=1
    sys.addaudithook(audit)


class Ledger:
    def __init__(self, root, labels, initial):
        self.root,self.labels,self.initial=Path(root),labels,initial
        self.db=sqlite3.connect(self.root/'oracle.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS observations (trajectory TEXT, slot INTEGER, cid TEXT, payload TEXT, seal TEXT, PRIMARY KEY(trajectory,slot), UNIQUE(trajectory,cid))')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, time TEXT, trajectory TEXT, slot INTEGER, cid TEXT, replay INTEGER)')
        self.db.execute('CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, role TEXT, opens INTEGER, time TEXT)')
        self.db.commit()

    def release(self, request):
        trajectory,slot,cid=request['trajectory'],int(request['slot']),request['candidate_id']
        seed,method=trajectory.split('_',1)
        if int(seed) not in SEEDS or method not in METHODS or cid not in self.labels or not 0<=slot<71:
            raise ValueError('Forbidden candidate/trajectory/budget')
        old=self.db.execute('SELECT cid,payload,seal FROM observations WHERE trajectory=? AND slot=?',(trajectory,slot)).fetchone()
        seal='initial_manifest'
        if slot<50:
            if cid!=self.initial[seed]['ids'][slot]:
                raise ValueError('Initial release differs from frozen manifest')
        else:
            path=self.root/'trajectories'/trajectory/'decisions'/f'{slot-49:02d}.json'
            sealed=read_json(path)
            seal=sha256(path)
            if sealed['candidate_id']!=cid or sealed['observed_n']!=slot or sealed['kind']!=('greedy' if slot==70 else 'qlogei'):
                raise ValueError('No matching immutable decision seal')
            prior=[r[0] for r in self.db.execute('SELECT cid FROM observations WHERE trajectory=? AND slot<? ORDER BY slot',(trajectory,slot))]
            if sealed['observed_ids']!=prior:
                raise ValueError('Decision observation order mismatch')
        if old and (old[0]!=cid or old[2]!=seal):
            raise ValueError('Idempotency token reused for changed decision')
        if not old:
            n=self.db.execute('SELECT COUNT(*) FROM observations WHERE trajectory=?',(trajectory,)).fetchone()[0]
            if n!=slot:
                raise ValueError('Nonsequential oracle request')
        payload=json.loads(old[1]) if old else dict(candidate_id=cid,**self.labels[cid])
        with self.db:
            if not old:
                self.db.execute('INSERT INTO observations VALUES(?,?,?,?,?)',(trajectory,slot,cid,json.dumps(payload),seal))
            self.db.execute('INSERT INTO calls(time,trajectory,slot,cid,replay) VALUES(?,?,?,?,?)',(now(),trajectory,slot,cid,int(old is not None)))
        return payload

    def export(self):
        observations=[dict(trajectory=t,slot=s,candidate_id=c,decision_sha256=h) for t,s,c,h in self.db.execute('SELECT trajectory,slot,cid,seal FROM observations ORDER BY trajectory,slot')]
        calls=[dict(call_id=i,time=t,trajectory=tr,slot=s,candidate_id=c,idempotent_replay=bool(r)) for i,t,tr,s,c,r in self.db.execute('SELECT * FROM calls ORDER BY id')]
        with (self.root/'oracle_access_log.jsonl').open('w') as f:
            for r in calls:
                f.write(json.dumps(r)+'\n')
        sessions=[dict(role=r,physical_A_label_file_opens=o,time=t) for r,o,t in self.db.execute('SELECT role,opens,time FROM sessions ORDER BY id')]
        result=dict(logical_releases=len(observations),unique_candidates=len({o['candidate_id'] for o in observations}),
            oracle_release_calls=len(calls),idempotent_replays=sum(c['idempotent_replay'] for c in calls),
            physical_A_label_file_opens=sum(s['physical_A_label_file_opens'] for s in sessions),
            cache_read_sessions=sessions,B_label_reads=0,B_SE_reads=0,node_test_reads=0,new_GNN_trains=0,KG_runs=0)
        write_json(self.root/'oracle_summary.json',result)
        write_json(self.root/'oracle_release_manifest.json',observations)
        return result


def serve(root,repo):
    counts={'physical_A_label_file_opens':0}
    access_guard(root,repo,'oracle',counts)
    labels=read_A_labels(repo)
    ledger=Ledger(root,labels,read_json(root/'initial_observation_manifest.json'))
    with ledger.db:
        ledger.db.execute('INSERT INTO sessions(role,opens,time) VALUES(?,?,?)',('oracle',counts['physical_A_label_file_opens'],now()))
    print(json.dumps({'ready':True}),flush=True)
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line)
                if request['operation']=='close':
                    print(json.dumps({'ok':True,'summary':ledger.export()}),flush=True)
                    break
                if request['operation']!='release':
                    raise ValueError('Unsupported oracle operation')
                print(json.dumps({'ok':True,'observation':ledger.release(request)}),flush=True)
            except Exception as exc:
                print(json.dumps({'ok':False,'error':repr(exc)}),flush=True)
    finally:
        ledger.export()
        ledger.db.close()
