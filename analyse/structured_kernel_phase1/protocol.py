"""The user-approved scientific and numerical constants."""
METHODS = ("L-A", "L-I", "W-A", "W-I", "RAW16")
DRAWS = (20260922, 20260923)
STARTS = (0.5, 1.0, 2.0)
LBFGS = dict(lr=1., max_iter=100, max_eval=125, history_size=100,
             line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)


def frozen_protocol():
    return {
        "version": 1, "scope": "E0-E2 A-only exploratory evidence; no later experiments",
        "recipe": {"dataset": "Cora", "split": "public", "features": "NormalizeFeatures",
                   "normalization": "none", "hp_mode": "global4", "batch": "full_batch",
                   "max_epochs": 300, "patience": 80, "training_seeds": [0, 1, 2]},
        "input": {"latent_dim": 12, "HP": ["log10(lr)", "dropout", "log2(hidden_dim)", "log10(l2)"],
                  "ddof": 0, "constant_x_scale": 1, "invalid_y_scale": "BLOCKED",
                  "distance_scale": "strictly positive upper-triangle distances within selected T; median; fallback 1"},
        "wl": {"H": 2, "pool": "A179 unlabeled configurations only, including Q structures",
               "directed": True, "separate_round_vocabularies": True,
               "feature": "[phi/l2norm(phi),n_nodes/n_max,n_edges/e_max]", "reduce_dimension": False},
        "kernel": {"base": "isotropic Matern5/2", "eta": .5,
                   "additive": "sf2*(ka+kh)/2", "interaction": "sf2*(.25*(ka+kh)+.5*ka*kh)",
                   "parameter_count": {"L-A":5,"L-I":5,"W-A":5,"W-I":5,"RAW16":4},
                   "priors": [], "extra_losses": []},
        "noise": {"known": "SE^2/y_scale_T^2; SE already estimates three-seed mean",
                  "extra_initial_std_variance": .01, "extra_lower_std_variance": 1e-6,
                  "min_fixed_noise": 0, "jitter_is_not_observation_variance": True},
        "optimizer": {"starts": list(STARTS), "initial_outputscale": 1, "initial_mean": 0,
                      "lower_lengthscale": 1e-4,"lower_outputscale":1e-8,
                      "adam": {"lr": .01,"steps":300,"betas":[.9,.999],"eps":1e-8,"weight_decay":0},
                      "lbfgs": LBFGS, "selection": "final finite training NLL, then start index",
                      "mll_normalized_once_by_library": True,
                      "warning": "Adam final50 improvement>1e-4 or LBFGS max_iter/max_eval reached",
                      "jitter": [1e-8,1e-7,1e-6], "retries": "Cholesky failure only; same parameters, no extra updates"},
        "matrix": {"folds":5,"draws":list(DRAWS),"n":[20,50],"methods":list(METHODS),
                   "contexts":20,"settings":100,"starts":300,"pilot":{"fold":0,"draw":20260922,"n":[20,50]}},
        "execution": {"device":"cpu","dtype":"float64","threads":1,"workers":1},
        "scoring": {"primary":"MAE_pp", "prospective_variance":"latent_var+extra_noise+median_T(known_var), standardized",
                    "retrospective":"latent_var_raw+extra_raw+SE_Q^2", "NLPD_units":"raw accuracy density, nat",
                    "aggregation":"old model seeds then draws then equal folds; selected start only for new GP",
                    "Q_access":"scorer only after state/prediction seal; never used for fit/start selection",
                    "top_ties":"candidate_id ascending", "IS90":"width+20*lower_miss+20*upper_miss"},
        "gate": {"n50_MAE_gain_pp":.3,"positive_folds":4,"n50_NLPD_max_worsening_nat":.1,
                 "n50_IS90_max_worsening_fraction":.05,"n20_MAE_max_worsening_pp":.3,
                 "all_settings_complete":True,"MAE_tie_pp":.05,"NLPD_tie_nat":.01,
                 "tie_preference":["additive","latent"],"no_gain":"NO_STABLE_GAIN"},
        "forbidden": ["B labels/SE", "node test scores", "new GNN training", "VAE training/decoding",
                      "KG", "capacity branch", "CLML", "fuzzy metric", "ARCH_FLOOR", "new loss",
                      "calibration", "prediction clipping", "parameter scan", "cross-dataset", "LayerNorm change"],
    }
