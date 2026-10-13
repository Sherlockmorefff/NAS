from analyse.structured_kernel_phase1.protocol import frozen_protocol

METHODS = ('W-I', 'RAW16')
SEEDS = tuple(range(2026100200, 2026100210))
PHASE1 = 'results/structured_kernel_phase1_20261002T142841Z'
OLD = 'results/structured_kernel_phase2_20261002T161300Z'


def eligibility(purpose, pool, *, historical_development=True, b_access=False, test_access=False):
    if b_access or test_access:
        raise ValueError('B/test access is prohibited')
    if purpose == 'independent_confirmation' and historical_development:
        return 'BLOCKED'
    if purpose == 'development_replay' and pool == 'A179':
        return 'ELIGIBLE'
    raise ValueError('Unsupported scientific purpose/pool')


def protocol():
    first = frozen_protocol()
    return dict(version=1, purpose='development_replay', independent_confirmation=False,
        pool='A179', B_label_access=False, B_SE_access=False, node_test_access=False,
        methods=list(METHODS), seeds=list(SEEDS), recipe=first['recipe'],
        input=first['input'], wl=first['wl'], noise=first['noise'],
        optimizer=first['optimizer'], kernel=first['kernel'], execution=first['execution'],
        initial={'n':50, 'rule':'sorted(A_ids); default_rng(SeedSequence([seed,0])).permutation; first 50'},
        budget={'trajectories':20, 'fits':420, 'starts':1260, 'initial_per_trajectory':50,
                'qlogei_per_trajectory':20, 'terminal_greedy_per_trajectory':1,
                'fit_n':list(range(50,71)), 'terminal_n':71, 'logical_releases':1420},
        acquisition={'class':'surrogate.accuracy_gp.StableQLogExpectedImprovement',
            'samples':512, 'q':1, 'fat':True, 'tau_max':.01, 'tau_relu':1e-6, 'eta':.001,
            'chunk_size':64, 'mc_step':list(range(1,21)),
            'mc_seed':'int(SeedSequence([search_seed,step,1]).generate_state(1,dtype=uint32)[0])',
            'posterior':'standardized latent function; training known variance + extra noise only',
            'best_f':'(max released mean - current T mean)/current T std(ddof=0)',
            'tie_atol':1e-12, 'tie':'smallest full candidate_id within max-score absolute tolerance',
            'terminal_score':'raw accuracy posterior mean, abs tolerance 1e-12'},
        analysis={'primary':'100*(n71 best W-I - n71 best RAW16)',
            'secondary':['n70 gain','greedy increment','trapezoid regret queries0..20 /20'],
            'bootstrap_replicates':2000, 'bootstrap_seed':2026100232, 'quantiles':[.025,.975],
            'gate':{'complete_trajectories':20, 'mean_gain_pp':.3, 'strictly_positive_seeds':7,'CI_lower_gt':0}},
        forbidden=['B labels/SE/predictions/scoring','node test','new GNN training','KG',
                   'other methods','parameter scan','new seeds','additional queries','WL extension'])
