"""Single writer grants tasks to two independent per-GPU worker processes."""
import sqlite3
import uuid
from .common import *

class Queue:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL, worker INTEGER, gpu TEXT, token TEXT, grants INTEGER DEFAULT 0, reserved INTEGER DEFAULT 0, event TEXT)')
        self.db.commit()

    def add(self, task):
        payload = json.dumps(task,sort_keys=True)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO tasks(id,payload,state) VALUES(?,?,?)',(task['task_id'],payload,'READY'))
            row = self.db.execute('SELECT payload FROM tasks WHERE id=?',(task['task_id'],)).fetchone()
            if row['payload'] != payload:
                raise ValueError('Task identity collision')

    def rows(self, state=None):
        return self.db.execute('SELECT * FROM tasks' + (' WHERE state=?' if state else '') + ' ORDER BY rowid',
            (state,) if state else ()).fetchall()

    def claim(self, worker, gpu, allowed=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM tasks WHERE state=? AND worker=?',('RUNNING',worker)).fetchone():
                self.db.rollback(); return None
            rows = self.db.execute('SELECT * FROM tasks WHERE state=? ORDER BY rowid',('READY',)).fetchall()
            row = next((r for r in rows if (allowed is None or r['id'] in allowed) and (r['gpu'] is None or r['gpu']==gpu['uuid'])),None)
            if row is None:
                self.db.rollback(); return None
            token = uuid.uuid4().hex
            self.db.execute('UPDATE tasks SET state=?,worker=?,gpu=?,token=?,grants=grants+1,reserved=1 WHERE id=?',
                ('RUNNING',worker,gpu['uuid'],token,row['id']))
            self.db.commit()
            return {'task':json.loads(row['payload']),'gpu':gpu,'token':token,'worker':worker,
                'execution_identity':digest({'task':json.loads(row['payload']),'execution':{
                    'GPU_UUID':gpu['uuid'],'physical_index':gpu['index'],'logical_device':'cuda:0',
                    'precision':'float32','strict_determinism':True}})}
        except BaseException:
            self.db.rollback(); raise

    def finish(self, event):
        row = self.db.execute('SELECT * FROM tasks WHERE id=?',(event['task_id'],)).fetchone()
        if row is None or row['token'] != event['token'] or row['worker'] != event['worker']:
            raise ValueError('Worker event does not match exclusive grant')
        if row['state'] in {'COMPLETE','FAILED'}:
            if json.loads(row['event']) != event:
                raise ValueError('Changed terminal event')
            return
        with self.db:
            self.db.execute('UPDATE tasks SET state=?,event=? WHERE id=?',
                ('COMPLETE' if event.get('completed') else 'FAILED',json.dumps(event),event['task_id']))

    def requeue_resumable(self, tid):
        with self.db:
            self.db.execute('UPDATE tasks SET state=? WHERE id=?',('READY',tid))

def pilot_tasks(tasks):
    small = sorted((t for t in tasks if t['role']=='rerank_seed'),key=lambda t:(t['parameter_count'],t['normalized_config_id'],t['seed']))[:3]
    large = sorted((t for t in tasks if t['role']=='rerank_mask'),key=lambda t:(-t['parameter_count'],t['normalized_config_id'],t['split'],t['seed']))[:3]
    if len(small)!=3 or len(large)!=3:
        return []
    return [small[0],large[0],large[1],small[1],small[2],large[2]]
