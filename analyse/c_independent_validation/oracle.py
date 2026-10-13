"""Isolated validation-cache oracle; only sequential, unique query releases."""
import sqlite3,sys,subprocess
from .common import *
from .performance import METHODS,SEEDS

class Ledger:
 def __init__(self,path,labels,initial,random):
  self.labels,self.initial,self.random=labels,initial,random
  self.db=sqlite3.connect(path);self.db.execute('PRAGMA synchronous=FULL')
  self.db.execute('CREATE TABLE IF NOT EXISTS observations(trajectory TEXT,slot INTEGER,cid TEXT,payload TEXT,PRIMARY KEY(trajectory,slot),UNIQUE(trajectory,cid))')
  self.db.execute('CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY,trajectory TEXT,slot INTEGER,cid TEXT,replay INTEGER,time TEXT)');self.db.commit()
 def release(self,tr,slot,cid):
  seed_text,method=tr.split('_',1);seed=int(seed_text)
  if seed not in SEEDS or method not in METHODS or not 0<=slot<71 or cid not in self.labels:raise ValueError('Query outside frozen pool/budget')
  if slot<50 and cid!=self.initial[seed_text]['ids'][slot]:raise ValueError('Changed common initialization')
  if method=='Random':
   order=self.random[seed_text]['initial_ids']+self.random[seed_text]['query_ids']
   if cid!=order[slot]:raise ValueError('Changed Random sequence')
  old=self.db.execute('SELECT cid,payload FROM observations WHERE trajectory=? AND slot=?',(tr,slot)).fetchone()
  if old and old[0]!=cid:raise ValueError('Changed idempotency key')
  if not old:
   if self.db.execute('SELECT COUNT(*) FROM observations WHERE trajectory=?',(tr,)).fetchone()[0]!=slot:raise ValueError('Out-of-order release')
   if self.db.execute('SELECT 1 FROM observations WHERE trajectory=? AND cid=?',(tr,cid)).fetchone():raise ValueError('Repeated query')
  value=read_payload(old[1]) if old else {'candidate_id':cid,'mean':self.labels[cid]['mean'],'se':self.labels[cid]['se']}
  with self.db:
   if not old:self.db.execute('INSERT INTO observations VALUES(?,?,?,?)',(tr,slot,cid,json.dumps(value)))
   self.db.execute('INSERT INTO calls(trajectory,slot,cid,replay,time) VALUES(?,?,?,?,?)',(tr,slot,cid,int(bool(old)),now()))
  return value
 def summary(self):
  n,u=self.db.execute('SELECT COUNT(*),COUNT(DISTINCT cid) FROM observations').fetchone();calls=self.db.execute('SELECT COUNT(*),COALESCE(SUM(replay),0) FROM calls').fetchone()
  return {'logical_releases':n,'unique_configurations':u,'calls':calls[0],'repeated_requests':calls[1]}

def read_payload(s):return json.loads(s)

def serve(root):
 labels=read(root/'private/C_validation_cache.json');initial=read(root/'initial_observation_manifest.json');random=read(root/'random_query_manifest.json')
 if sha(root/'private/C_validation_cache.json')!=read(root/'P4_complete.json')['files']['private/C_validation_cache.json']:raise ValueError('Frozen cache changed')
 ledger=Ledger(root/'C_oracle.sqlite',labels,initial,random)
 print(json.dumps({'ready':True}),flush=True)
 for line in sys.stdin:
  try:
   request=json.loads(line)
   if request.get('operation')=='close':
    print(json.dumps({'ok':True,'summary':ledger.summary()}),flush=True);break
   if set(request)!={'operation','trajectory','slot','candidate_id'} or request['operation']!='release':raise ValueError('Unsupported oracle operation')
   value=ledger.release(request['trajectory'],request['slot'],request['candidate_id']);print(json.dumps({'ok':True,'observation':value}),flush=True)
  except Exception as error:print(json.dumps({'ok':False,'error':type(error).__name__+': '+str(error)}),flush=True)
 write(root/'oracle_summary.json',ledger.summary());ledger.db.close()

class Client:
 def __init__(self,root):
  self.process=subprocess.Popen([sys.executable,'-B','-m','analyse.c_independent_validation','_oracle','--run-dir',str(root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
  line=self.process.stdout.readline()
  if not line or not json.loads(line).get('ready'):raise RuntimeError('Cache oracle failed to initialize')
 def request(self,v):
  self.process.stdin.write(json.dumps(v)+'\n');self.process.stdin.flush();line=self.process.stdout.readline()
  if not line:raise RuntimeError('Oracle disconnected')
  answer=json.loads(line)
  if not answer['ok']:raise ValueError(answer['error'])
  return answer
 def release(self,tr,slot,cid):return self.request({'operation':'release','trajectory':tr,'slot':slot,'candidate_id':cid})['observation']
 def close(self):
  if self.process.poll() is None:self.request({'operation':'close'});self.process.wait(timeout=30)

def learner_guard(root,repo):
 allowed={root/n for n in ['public_C.json','initial_observation_manifest.json','random_query_manifest.json','performance_identity.json','performance_protocol.json']}
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes)):return
  p=Path(args[0]).resolve();mode=args[1];flags=args[2]
  if p.is_relative_to(root/'trajectories') or p in allowed or p in {root/'P5_progress.json',root/'P5_progress.json.tmp'}:return
  if p.is_relative_to(repo/'results') or p.is_relative_to(repo/'legacy_artifacts'):
   raise PermissionError('Learner cannot access full cache, task results, A/B/test scores or other results')
 sys.addaudithook(audit)
