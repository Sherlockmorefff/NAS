"""Validation-only tasks; reuse the October 7 exact-state trainer unchanged."""
import signal
import subprocess
import time
from .common import *

def load_view(root, task, device):
    import torch
    from types import SimpleNamespace
    saved = torch.load(root / 'development_data.pt',map_location='cpu',weights_only=False)
    assert torch.all(saved['development_y'][640:] == -100)
    split = read(root / 'SPLITS.json')['splits'][task['split']]
    roles = split['indices']
    assert digest(roles) == split['identity']
    def mask(ids):
        result = torch.zeros(2708,dtype=torch.bool); result[ids] = True; return result.to(device)
    view = SimpleNamespace(x=saved['x'].to(device),edge_index=saved['edge_index'].to(device),
        y=saved['development_y'].to(device),train_mask=mask(roles['train']),val_mask=mask(roles['selection']))
    assert set(roles['train']) | set(roles['selection']) == set(range(640))
    assert not set(roles['train']) & set(roles['selection'])
    for attr, role in [('train_mask','train'),('val_mask','selection')]:
        assert digest(getattr(view,attr).nonzero().flatten().cpu().tolist()) == split['index_hashes'][role]
    assert int(view.train_mask.sum()) == 140 and int(view.val_mask.sum()) == 500
    assert torch.all(view.y[~(view.train_mask|view.val_mask)] == -100)
    return view, saved['num_features'], saved['num_classes'], split

def train_task(root, repo, assignment_path, pause=False):
    verify_run(root, repo)
    assignment = read(assignment_path); task = assignment['task']; gpu = assignment['gpu']
    assert read(root/'task_definitions'/(task['task_id']+'.json')) == task
    assert physical_task(root,task['group'],task['role'],task['candidate_id'],task['seed'],task['split']) == task
    folder = root / 'training' / task['task_id']; folder.mkdir(parents=True,exist_ok=True)
    identity = execution_identity(root,task,gpu)
    assert assignment['execution_identity'] == identity
    config = read(root / 'CONFIGURATIONS.json')['by_id'][task['candidate_id']]
    assert config['configuration_hash'] == task['configuration_hash']
    return_seals = {}
    if task['role'] == 'final_validation':
        for rule in assignment['rules']:
            return_seals[rule] = require_seal(root,task['group'],rule,task)
    from deterministic_runtime import prepare_deterministic_environment,configure_torch_determinism
    prepare_deterministic_environment()
    import torch
    configure_torch_determinism(torch); torch.set_num_threads(1); torch.set_default_dtype(torch.float32)
    assert torch.cuda.device_count() == 1
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == gpu['uuid']
    device = torch.device('cuda:0')
    data,in_ch,out_ch,split = load_view(root,task,device)
    from analyse.rank_reversal_diagnostic.train import train,PlannedPause
    access_guard(root,repo,'train')
    def initialize():
        receipt = folder / 'INITIALIZATION.json'
        if receipt.exists():
            raise RuntimeError('Previously initialized task without resumable state; free restart forbidden')
        write(receipt,{'task_id':task['task_id'],'execution_identity':identity,'time':now(),
            'controller_grant':assignment['token'],'seed':task['seed'],'gpu':gpu})
    start = time.time()
    try:
        result = train(config['configuration'],data,in_ch,out_ch,task['seed'],device,folder,identity,
            max_epochs=300,patience=80,pause_after=5 if pause else None,on_initialization=initialize)
    except PlannedPause:
        return 75
    import numpy as np
    with np.load(folder / 'selection.npz') as scores:
        indices = scores['indices'].tolist(); correct = int(scores['correct'].sum()); denominator = len(indices)
    assert indices == split['indices']['selection'] and denominator == 500
    assert correct / denominator == result['selection'] and result['test_forwards'] == 0
    # A completed second call must return saved metadata, preserving every core file.
    before = {n:sha(folder/n) for n in ['latest.pt','best_model.pt','selection.npz','result.json','epoch_events.jsonl','INITIALIZATION.json']}
    repeated = train(config['configuration'],data,in_ch,out_ch,task['seed'],device,folder,identity,
        max_epochs=300,patience=80,on_initialization=initialize)
    assert repeated['identity'] == identity
    verify(folder,before)
    binding = {'task':task,'identity':identity,'gpu':gpu,'logical_device':'cuda:0',
        'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'device_name':torch.cuda.get_device_name(0),
        'configuration_hash':config['configuration_hash'],'masks_hash':split['identity'],
        'actual_train_hash':digest(data.train_mask.nonzero().flatten().cpu().tolist()),
        'actual_validation_hash':digest(indices),'correct':correct,'denominator':denominator,
        'old_test_or_H_labels_in_view':False,'original_data_access_blocked':True,'test_forwards':0,
        'return_seals_before_evaluation':return_seals,'best_restored_verified':True,
        'completed_repeat_new_training':0,'completed_repeat_core_hashes_unchanged':True,
        'result_sha256':sha(folder/'result.json'),'worker_invocation_started':times(start),
        'worker_invocation_completed':times(time.time())}
    write(folder/'BINDING.json',binding)
    print(json.dumps({'task':task['task_id'],'role':task['role'],'epochs':result['stopped_epoch'],
        'validation_correct':correct,'seconds':result['worker_seconds'],'GPU':gpu['index']}),flush=True)
    return 0

def worker(root, repo, number):
    lock = exclusive(root / f'worker-{number}.lock')
    assignment_path = root / 'assignments' / f'worker-{number}.json'
    pulse_path = root / 'workers' / f'worker-{number}.json'
    gpu = read(root/'GPUS.json')['workers'][str(number)]
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    last_token = None
    while True:
        state = read(root/'STATUS.json')['state']
        if state in {'COMPLETE','FAILED','PARTIAL','STOPPED'}:
            write(pulse_path,{'pid':os.getpid(),'worker':number,'state':'EXITED','time':now()}); return
        write(pulse_path,{'pid':os.getpid(),'worker':number,'state':'IDLE','gpu':gpu,'time':now(),'unix':time.time()})
        if not assignment_path.exists():
            time.sleep(2); continue
        assignment = read(assignment_path)
        if assignment['token'] == last_token:
            time.sleep(2); continue
        last_token = assignment['token']; task = assignment['task']; folder = root/'training'/task['task_id']
        folder.mkdir(parents=True,exist_ok=True)
        event_path = root / 'worker_events' / (last_token+'.json')
        if event_path.exists():
            continue
        start = time.time(); strict_resumes = 0; exit_code = None
        cmd = [sys.executable,'-B','-m',MODULE,'train-task','--run-dir',str(root),'--assignment',str(assignment_path)]
        exercise = assignment.get('exercise_interruption',False) and not (folder/'planned_pause.json').exists()
        if exercise:
            cmd += ['--pause']
        while True:
            log(root/f'worker-{number}-commands.jsonl',{'time':now(),'command':cmd,'token':last_token,'task_id':task['task_id']})
            with (folder/'worker.log').open('a') as output:
                proc = subprocess.Popen(cmd,cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                while proc.poll() is None:
                    write(pulse_path,{'pid':os.getpid(),'child_pid':proc.pid,'worker':number,'state':'TRAINING',
                        'gpu':gpu,'task_id':task['task_id'],'token':last_token,'unix':time.time(),'time':now()})
                    if (root/'STOP_REQUEST.json').exists() or read(root/'STATUS.json')['state'] in {'FAILED','PARTIAL','STOPPED'}:
                        os.killpg(proc.pid,signal.SIGTERM); proc.wait(timeout=30)
                        write(event_path,{'token':last_token,'task_id':task['task_id'],'worker':number,'exit_code':proc.returncode,
                            'stopped':True,'started':start,'ended':time.time(),'strict_resumes':strict_resumes})
                        return
                    time.sleep(3)
                exit_code = proc.returncode
            if exit_code == 75 and (folder/'latest.pt').exists() and (folder/'planned_pause.json').exists():
                cmd = [x for x in cmd if x!='--pause']; strict_resumes += 1
                continue
            break
        write(event_path,{'token':last_token,'task_id':task['task_id'],'worker':number,'exit_code':exit_code,
            'started':start,'ended':time.time(),'wall_seconds':time.time()-start,'strict_resumes':strict_resumes,
            'completed':exit_code==0,'gpu':gpu})
        print(json.dumps({'task':task['task_id'],'exit_code':exit_code,'worker':number,'wall_seconds':time.time()-start}),flush=True)
        time.sleep(1)
