import statistics
import time
from .common import *

def due(eta, timestamp):
    return timestamp >= eta.get('next_check_unix',eta['first_check_unix'])

def recheck(eta, timestamp):
    value = dict(eta); value['next_check_unix'] = timestamp + 300
    value['next_check'] = times(value['next_check_unix'])
    return value

def estimate(root, queue):
    pilot = read(root/'PILOT_MANIFEST.json')['task_ids']
    indexed = {r['id']:r for r in queue.rows()}
    if not all(t in indexed and indexed[t]['state']=='COMPLETE' for t in pilot):
        return None
    measured = []
    for tid in pilot:
        row = indexed[tid]; event = json.loads(row['event']); task = json.loads(row['payload'])
        result = read(root/'training'/tid/'result.json')
        measured.append({'task_id':tid,'worker':row['worker'],'gpu_uuid':row['gpu'],'role':task['role'],
            'parameter_count':task['parameter_count'],'seed':task['seed'],'split':task['split'],
            'started':times(event['started']),'ended':times(event['ended']),
            'wall_seconds':event['wall_seconds'],'worker_seconds':result['worker_seconds'],
            'gpu_event_seconds':result['gpu_event_seconds'],'epochs':result['stopped_epoch'],
            'strict_resumes':event['strict_resumes'],'cache':False,'formal_budget_task':True})
    csvwrite(root/'TIMING_PILOT.csv',measured)
    # With only six samples, extrapolate a range, allowing large future structures and launch/IO variation.
    rates = {}
    for number in (1,2):
        subset = [m for m in measured if m['worker']==number]
        if not subset:
            return None
        rates[number] = len(subset)/sum(m['wall_seconds'] for m in subset)
    pilot_span = max(m['ended']['UTC'] for m in measured)
    pilot_events = [json.loads(indexed[t]['event']) for t in pilot]
    observed_span = max(e['ended'] for e in pilot_events) - min(e['started'] for e in pilot_events)
    observed_throughput = len(pilot) / observed_span
    throughput = min(sum(rates.values()), observed_throughput)
    counts = {s:sum(r['state']==s for r in queue.rows()) for s in ['READY','RUNNING','COMPLETE','FAILED']}
    completed_rerank = sum(r['state']=='COMPLETE' and json.loads(r['payload'])['role']!='final_validation' for r in queue.rows())
    completed_eval = counts['COMPLETE']-completed_rerank
    remaining_low = 1200-completed_rerank + max(0,200-completed_eval)
    remaining_high = 1200-completed_rerank + max(0,600-completed_eval)
    elapsed_gp = read(root/'SEARCH_PROGRESS.json') if (root/'SEARCH_PROGRESS.json').exists() else {}
    fit_n = elapsed_gp.get('completed_fits',0)
    fit_seconds = elapsed_gp.get('elapsed_seconds',0)/fit_n if fit_n else 5
    gp_remaining = max(0,420-fit_n)*fit_seconds
    cpu_allowance = gp_remaining + 300  # explicitly include remaining sequential fits and analysis/packing
    low = remaining_low/throughput*.8 + 120
    high = remaining_high/throughput*1.6 + cpu_allowance
    stamp = time.time(); estimate_high = stamp+high
    value = {'state':'ESTIMATED','pilot_tasks':6,'estimated_at':times(stamp),'estimated_at_unix':stamp,
        'remaining_seconds_range':[low,high],'estimated_completion_range':[times(stamp+low),times(estimate_high)],
        'conservative_completion_unix':estimate_high,'first_check_unix':estimate_high+900,
        'first_check':times(estimate_high+900),'next_check_unix':estimate_high+900,'next_check':times(estimate_high+900),
        'check_buffer_seconds':900,'recheck_seconds':300,'physical_remaining_range':[remaining_low,remaining_high],
        'observed_worker_tasks_per_second':rates,'combined_tasks_per_second':throughput,
        'observed_two_GPU_pilot_wall_seconds':observed_span,'observed_two_GPU_tasks_per_second':observed_throughput,
        'remaining_GP_fit_estimate_seconds':gp_remaining,'CPU_analysis_and_pack_allowance_seconds':300,
        'known_queue_counts':counts,'pilot':measured,
        'limitations':'six formal tasks; stage dependencies and unseen configuration sizes; bound includes unmaterialized evaluation dependencies; cache releases are not charged as training time'}
    if (root/'ETA.json').exists():
        old = read(root/'ETA.json')
        value.update(first_check_unix=old['first_check_unix'],first_check=old['first_check'],
            next_check_unix=old['next_check_unix'],next_check=old['next_check'])
    write(root/'ETA.json',value); log(root/'ETA_HISTORY.jsonl',value)
    return value
