"""Synthetic-only active-window resume check. No formal rules are changed."""
import sys,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from analyse.fuzzy_convergence_core import *
from analyse.fuzzy_loss_interaction_run import oracle_request

def main(o):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);started=time.perf_counter()
    dest=o/'engineering_resume_windows.json';assert not dest.exists()
    fixture=load(o/'synthetic/fixture.pt');data=observed_data(fixture['features'],fixture['scaler'],fixture['labels'],.01)
    initial=JointMetricModel(12,seed=0).state_dict();budget=Budget(o/'synthetic/ledger','synthetic-v1',synthetic=True)
    before=dict(budget.counts);meta=dict(task='synthetic_active_windows',stage='offline',representation_version=0)
    # Only this disposable process uses short toy windows to exercise restore paths within the synthetic cap.
    original=copy.deepcopy(RULES)
    for kind in RULES:RULES[kind].update(prefix=4,interval=2,width=6,cap=14)
    rng=rng_state();d=o/'synthetic/active_windows'
    a=fit_mapping(d/'map_cont',initial,data,.1,budget,meta,adaptive=True,cap_override=14)
    restore_rng(rng);mid=fit_mapping(d/'map_resume',initial,data,.1,budget,meta,adaptive=True,cap_override=14,stop_after=6)
    assert mid['scheduler']['best'] is not None and mid['scheduler']['reference'] and mid['completed_updates']==6
    b=fit_mapping(d/'map_resume',initial,data,.1,budget,meta,adaptive=True,cap_override=14)
    for key in ['model','optimizer','rng','scheduler','history','decisions']:assert tree_hash(a[key])==tree_hash(b[key]),key
    cache=embedding_cache(reload_model(a['model']),data);rng=rng_state()
    ga=fit_gp(d/'gp_cont',cache,data,budget,meta,adaptive=True,cap_override=14)
    restore_rng(rng);gm=fit_gp(d/'gp_resume',cache,data,budget,meta,adaptive=True,cap_override=14,stop_after=6)
    assert gm['scheduler']['best'] is not None
    gb=fit_gp(d/'gp_resume',cache,data,budget,meta,adaptive=True,cap_override=14)
    for key in ['model','optimizer','rng','scheduler','history','decisions']:assert tree_hash(ga[key])==tree_hash(gb[key]),key
    RULES.clear();RULES.update(original)
    assert budget.counts==dict(mapping=before['mapping']+28,gp=before['gp']+28)
    # Exercise the actual persisted-receipt reader twice. It must not request a fresh oracle response.
    ids=fixture['features']['ids'][10:15];selection=dict(selected_ids=ids);bd=d/'receipt_task/batches/b00'
    write(bd/'selection.json',selection)
    receipt=dict(ids=ids,labels={c:{'synthetic':True} for c in ids},query_count_before=0,query_count_after=5,selection_sha=file_hash(bd/'selection.json'))
    write(bd/'reveal.json',receipt)
    for _ in range(2):assert oracle_request(o,d/'receipt_task',dict(name='fake'),{},0,selection)==receipt
    result=dict(passed=True,scope='synthetic only; short prefix4/window6/interval2/cap14 in disposable process',
        production_rules_restored=RULES==original,model_Adam_RNG_scheduler_history_windows_exact=True,
        actual_persisted_receipt_reused_without_oracle_request=True,additional_updates=dict(mapping=28,gp=28),total_synthetic_updates=budget.counts,seconds=time.perf_counter()-started)
    write(dest,result)
    checks=read(o/'engineering_checks.json');write(o/'engineering_checks_initial.json',checks,immutable=True)
    checks['active_window_resume']=result;checks['synthetic'].update(mapping_updates=budget.counts['mapping'],gp_updates=budget.counts['gp']);write(o/'engineering_checks.json',checks)
    print(json.dumps(result))

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
