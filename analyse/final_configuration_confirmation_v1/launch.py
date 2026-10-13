"""Idempotent tmux lifecycle scoped to this run only."""
import shlex
import subprocess
import time
from .common import *

def process_matches(pid, root, role=None):
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return MODULE in cmd and str(root) in cmd and (role is None or role in cmd)
    except FileNotFoundError:
        return False

def start(root, repo, resume=False):
    verify_run(root,repo)
    state=read(root/'STATUS.json')['state']
    if state=='COMPLETE':
        from .analysis import verify_complete
        verify_complete(root,repo); print('Already COMPLETE; no processes or training started'); return
    session='cfgret_'+root.name.split('_')[-1]
    control=read(root/'CONTROLLER_PROCESS.json') if (root/'CONTROLLER_PROCESS.json').exists() else {}
    if control and process_matches(control['pid'],root,'controller'):
        print(json.dumps({'state':'ALREADY_RUNNING','session':session,'controller_pid':control['pid']})); return
    if state in {'FAILED','STOPPED','PARTIAL'} and not resume:
        raise RuntimeError('Use explicit resume; previous failure/stop is retained')
    if resume and (root/'STOP_REQUEST.json').exists():
        (root/'STOP_REQUEST.json').unlink()
    from .protocol import inventory_gpus
    current=inventory_gpus(); expected=read(root/'GPUS.json')['workers']
    assert {g['uuid'] for g in current['workers'].values()}=={g['uuid'] for g in expected.values()}
    present=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
    python=sys.executable
    def command(role, number=None):
        args=[python,'-u','-B','-m',MODULE,role,'--run-dir',str(root),'--log-file',str(root/(role+('-'+str(number) if number else '')+'.log'))]
        if number:
            args += ['--worker',str(number)]
        if role=='controller' and resume:
            args += ['--resume']
        return 'cd '+shlex.quote(str(repo))+' && exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '+shlex.join(args)
    if present:
        windows=subprocess.check_output(['tmux','list-windows','-t',session,'-F','#{window_name}'],text=True).splitlines()
        if set(windows)!={'controller','worker-1','worker-2'}:
            raise RuntimeError('Existing tmux identity/windows mismatch')
        for name,role,number in [('controller','controller',None),('worker-1','worker',1),('worker-2','worker',2)]:
            subprocess.run(['tmux','respawn-window','-k','-t',session+':'+name,command(role,number)],check=True)
    else:
        subprocess.run(['tmux','new-session','-d','-s',session,'-n','controller',command('controller')],check=True)
        for number in (1,2):
            subprocess.run(['tmux','new-window','-t',session,'-n',f'worker-{number}',command('worker',number)],check=True)
    for name in ['controller','worker-1','worker-2']:
        subprocess.run(['tmux','set-window-option','-t',session+':'+name,'remain-on-exit','on'],check=True)
    write(root/'SESSION.json',{'session':session,'workers':expected,'started':now(),'repo':str(repo),
        'python':python,'root':str(root),'resume':resume})
    base=shlex.join([python,'-B','-m',MODULE])
    (root/'OPERATIONS.md').write_text(f'''# 本次运行操作

会话：{session}
目录：{root}

启动（重复执行幂等）：
```
cd {repo}
{base} start --run-dir {root}
```
查看三个窗口：
```
tmux attach -t {session}
```
状态：
```
cat {root}/STATUS.json
cat {root}/ETA.json
```
停止（仅请求停止本次控制器/worker，保存完整最新状态）：
```
{base} stop --run-dir {root}
```
续跑（只能完整状态续跑；失败无状态不会重新初始化）：
```
{base} start --run-dir {root} --resume
```
完成后核验（新增训练0）：
```
{base} verify --run-dir {root}
```
日志：controller.log、worker-1.log、worker-2.log、SEARCH.log、training/<task_id>/worker.log。
定时验收由controller内的持久服务器循环执行，首次ETA保守完成时点+15分钟，未完成每300秒重查；不依赖聊天在线。
''')
    print(json.dumps({'state':'STARTED','session':session,'run_dir':str(root),'workers':expected}),flush=True)

def stop(root):
    if read(root/'STATUS.json')['state']=='COMPLETE':
        print('Already COMPLETE; nothing stopped'); return
    proc=read(root/'CONTROLLER_PROCESS.json')
    write(root/'STOP_REQUEST.json',{'time':now(),'controller_pid':proc['pid'],'run_dir':str(root)})
    if not process_matches(proc['pid'],root,'controller'):
        import signal
        for number in (1,2):
            pulse=root/'workers'/f'worker-{number}.json'
            if pulse.exists():
                worker=read(pulse); child=worker.get('child_pid')
                if child and process_matches(child,root,'train-task'):
                    os.killpg(child,signal.SIGTERM)
        write(root/'STATUS.json',{'state':'STOPPED','time':now(),'reason':'controller gone; only verified owned training children signalled'})
    print('Stop requested for this run; checkpoint-safe explicit resume only')
