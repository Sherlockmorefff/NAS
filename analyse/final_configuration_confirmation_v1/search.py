"""Original sequential W-I, with an explicit n60 terminal branch."""
import sqlite3
import subprocess
import sys
import time
from .common import *

class QueryLedger:
    def __init__(self, path, labels, initial):
        self.labels, self.initial = labels, initial
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS releases (trajectory TEXT, slot INTEGER, cid TEXT, payload TEXT, PRIMARY KEY(trajectory,slot), UNIQUE(trajectory,cid))')
        self.db.execute('CREATE TABLE IF NOT EXISTS requests (trajectory TEXT, slot INTEGER, cid TEXT, replay INTEGER, time TEXT)')
        self.db.commit()

    def release(self, tr, slot, cid):
        group_name, rule = tr.split('_')
        group = int(group_name[1:])
        limit = 71 if rule == 'U' else 61 if rule == 'SM' else 0
        if not 0 <= group < 20 or not 0 <= slot < limit or cid not in self.labels:
            raise ValueError('Outside frozen query scope')
        if slot < 50 and self.initial[str(group)][slot] != cid:
            raise ValueError('Common initial configuration changed')
        old = self.db.execute('SELECT cid,payload FROM releases WHERE trajectory=? AND slot=?',(tr,slot)).fetchone()
        if old:
            if old[0] != cid:
                raise ValueError('Changed query replay')
            payload = json.loads(old[1])
        else:
            count = self.db.execute('SELECT COUNT(*) FROM releases WHERE trajectory=?',(tr,)).fetchone()[0]
            if count != slot:
                raise ValueError('Sequential query dependency violated')
            if self.db.execute('SELECT 1 FROM releases WHERE trajectory=? AND cid=?',(tr,cid)).fetchone():
                raise ValueError('Repeated candidate query')
            payload = self.labels[cid]
        with self.db:
            if not old:
                self.db.execute('INSERT INTO releases VALUES(?,?,?,?)',(tr,slot,cid,json.dumps(payload)))
            self.db.execute('INSERT INTO requests VALUES(?,?,?,?,?)',(tr,slot,cid,int(bool(old)),now()))
        return payload

    def summary(self):
        releases = self.db.execute('SELECT COUNT(*) FROM releases').fetchone()[0]
        requests,replays = self.db.execute('SELECT COUNT(*),COALESCE(SUM(replay),0) FROM requests').fetchone()
        return {'unique_branch_releases':releases,'requests':requests,'replays':replays,
            'logical_query_train_cost':11580,'new_query_training':0}

def serve(root):
    frozen = read(root / 'FREEZE.json')
    assert sha(root / 'private/QUERY_CACHE.json') == frozen['inputs']['private/QUERY_CACHE.json']
    ledger = QueryLedger(root / 'query_oracle.sqlite',read(root / 'private/QUERY_CACHE.json'),read(root / 'INITIALIZATIONS.json'))
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request['operation'] == 'close':
                print(json.dumps({'ok':True,'summary':ledger.summary()}),flush=True); break
            if set(request) != {'operation','trajectory','slot','candidate_id'} or request['operation'] != 'release':
                raise ValueError('Unsupported oracle request')
            value = ledger.release(request['trajectory'],request['slot'],request['candidate_id'])
            print(json.dumps({'ok':True,'observation':value}),flush=True)
        except Exception as exc:
            print(json.dumps({'ok':False,'error':str(exc)}),flush=True)
    ledger.db.close()

class Client:
    def __init__(self, root):
        self.process = subprocess.Popen([sys.executable,'-B','-m',MODULE,'oracle','--run-dir',str(root)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        if not json.loads(self.process.stdout.readline()).get('ready'):
            raise RuntimeError('Query oracle failed to initialize')

    def request(self, value):
        self.process.stdin.write(json.dumps(value)+'\n'); self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Query oracle disappeared')
        answer = json.loads(line)
        if not answer['ok']:
            raise ValueError(answer['error'])
        return answer

    def release(self, tr, slot, cid):
        return self.request({'operation':'release','trajectory':tr,'slot':slot,'candidate_id':cid})['observation']

    def close(self):
        if self.process.poll() is None:
            answer = self.request({'operation':'close'}); self.process.wait(timeout=30)
            return answer['summary']
        return {'oracle_disappeared':True}

def run(root, repo):
    identity = verify_run(root, repo)
    if (root / 'SEARCH_COMPLETE.json').exists():
        verify(root, read(root / 'SEARCH_COMPLETE.json')['files']); return
    from analyse.structured_kernel_phase1.gp import cpu_setup,restore_model
    from analyse.structured_kernel_development_replay.learner import fit_current
    from analyse.structured_kernel_development_replay.acquisition import FrozenAdapter,enumerate_scores,mc_seed,choose
    import torch
    cpu_setup()
    public = read(root / 'PUBLIC_C.json'); configs = read(root / 'CONFIGURATIONS.json')['by_id']
    initial = read(root / 'INITIALIZATIONS.json'); seeds = read(root / 'SEEDS.json')['groups']
    client = Client(root); started = time.monotonic()
    access_guard(root,repo,'search')
    def decision(folder, name, obs, fit, q, query, seed, ordinary_step=None):
        path = folder / 'decisions' / (name + '.json')
        if path.exists():
            d = read(path)
            assert d['fit_identity'] == fit['identity'] and d['model_sha256'] == fit['selected_model_sha256']
            assert d['observed_ids'] == [o['candidate_id'] for o in obs]
            if d['sampler_sha256']:
                assert sha(path.with_suffix('.sampler.pt')) == d['sampler_sha256']
            return d
        checkpoint = torch.load(folder / 'fits' / f'n{len(obs):02d}' / fit['selected_model'],map_location='cpu',weights_only=False)
        adapter = FrozenAdapter(restore_model(checkpoint),checkpoint['jitter_index'])
        best_f = (max(o['mean'] for o in obs)-fit['scaler']['y_mean'])/fit['scaler']['y_scale']
        begin = time.monotonic(); sampler_hash = None
        if ordinary_step is not None:
            scores,sampler = enumerate_scores(adapter,q,best_f,seed,ordinary_step)
            from analyse.structured_kernel_followup.validation_train import save_torch
            save_torch(path.with_suffix('.sampler.pt'),sampler); sampler_hash = sha(path.with_suffix('.sampler.pt'))
        else:
            with torch.no_grad():
                scores = adapter.posterior(q.unsqueeze(-2)).mean[:,0,0].numpy()*fit['scaler']['y_scale']+fit['scaler']['y_mean']
        d = {'candidate_id':choose(query['ids'],scores),'kind':'qlogei' if ordinary_step is not None else 'terminal_mean',
            'observed_n':len(obs),'observed_ids':[o['candidate_id'] for o in obs], 'candidate_ids':query['ids'],
            'scores':scores.tolist(),'fit_identity':fit['identity'],'model_sha256':fit['selected_model_sha256'],
            'sampler_sha256':sampler_hash,'mc_seed':mc_seed(seed,ordinary_step) if ordinary_step is not None else None,
            'acquisition_seconds':time.monotonic()-begin,'numerical_events':adapter.policy.events,'time':now()}
        seal(path,d); return d
    try:
        for group in range(20):
            folder = root / 'trajectories' / f'g{group:02d}'; folder.mkdir(parents=True,exist_ok=True)
            if (folder / 'complete.json').exists():
                verify(folder,read(folder / 'complete.json')['files']); continue
            op = folder / 'U_observations.json'; obs = read(op) if op.exists() else []
            assert [o['candidate_id'] for o in obs[:50]] == initial[str(group)][:min(50,len(obs))]
            while len(obs) < 50:
                obs.append(client.release(f'g{group:02d}_U',len(obs),initial[str(group)][len(obs)])); write(op,obs)
            seed = seeds[str(group)]['search']
            while len(obs) < 71:
                n = len(obs)
                fit,q,query = fit_current(folder / 'fits' / f'n{n:02d}',public,obs,seed,'W-I',identity)
                if n == 60:
                    smp = folder / 'SM_observations.json'
                    sm = read(smp) if smp.exists() else []
                    assert sm[:min(60,len(sm))] == obs[:min(60,len(sm))]
                    while len(sm) < 60:
                        sm.append(client.release(f'g{group:02d}_SM',len(sm),obs[len(sm)]['candidate_id'])); write(smp,sm)
                    dsm = decision(folder,'SM_terminal_n60',obs,fit,q,query,seed)
                    if len(sm) == 60:
                        sm.append(client.release(f'g{group:02d}_SM',60,dsm['candidate_id'])); write(smp,sm)
                    assert len(sm) == 61 and sm[-1]['candidate_id'] == dsm['candidate_id']
                    top = ordered_observations(sm,configs)[:5]
                    seal(root / 'search_ready' / f'g{group:02d}_SM.json',{'group':group,'observations':sm,
                        'observations_sha256':sha(smp),'top5':[o['candidate_id'] for o in top],
                        'terminal_decision_sha256':sha(folder / 'decisions/SM_terminal_n60.json'),
                        'return_rule_scope':'own61; S/M share exactly this list','run_identity':identity})
                ordinary = n-49 if n < 70 else None
                name = f'U_q{ordinary:02d}' if ordinary is not None else 'U_terminal_n70'
                d = decision(folder,name,obs,fit,q,query,seed,ordinary)
                obs.append(client.release(f'g{group:02d}_U',n,d['candidate_id'])); write(op,obs)
                write(root / 'SEARCH_PROGRESS.json',{'group':group,'observed_n':len(obs),
                    'completed_fits':len(list((root / 'trajectories').glob('*/fits/n*/fit.json'))),
                    'elapsed_seconds':time.monotonic()-started,'time':now()})
                print(json.dumps({'group':group,'U_n':len(obs)}),flush=True)
            winner = ordered_observations(obs,configs)[0]
            seal(root / 'search_ready' / f'g{group:02d}_U.json',{'group':group,'candidate_id':winner['candidate_id'],
                'query_correct_sum':winner['query_correct_sum'],'query_denominator':winner['query_denominator'],
                'observations_sha256':sha(op),'terminal_decision_sha256':sha(folder / 'decisions/U_terminal_n70.json'),
                'run_identity':identity})
            assert (root / 'search_ready' / f'g{group:02d}_SM.json').exists()
            files = {str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name!='complete.json'}
            seal(folder / 'complete.json',{'group':group,'files':files,'U_queries':71,'SM_queries':61,
                'shared_prefix':60,'fits':21,'starts':63})
        summary = client.close()
        completions = list((root / 'trajectories').glob('*/complete.json'))
        assert len(completions) == 20
        assert summary['unique_branch_releases'] == 2640
        seal(root / 'SEARCH_COMPLETE.json',{'files':{str(p.relative_to(root)):sha(p) for p in completions},
            'groups':20,'fits':420,'starts':1260,'query_summary':summary,'elapsed_seconds':time.monotonic()-started,'time':now()})
    finally:
        client.close()
