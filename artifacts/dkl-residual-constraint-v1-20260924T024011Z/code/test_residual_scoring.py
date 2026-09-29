"""Synthetic-only statistics checks; zero optimizer updates or model forwards."""
import copy
import itertools
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import score_residual as score

def data():
    train=[f't{i:02d}' for i in range(20)];query=[f'q{i}' for i in range(7)]
    labels={c:{'validation_mean':.5+i*.005,'validation_se':.01} for i,c in enumerate(train)}
    labels.update({c:{'validation_mean':.55+i*.02,'validation_se':.02} for i,c in enumerate(query)})
    nn={'training_ids':train,'query_ids':query,'nearest_train_ids':[train[0]]*7,
        'five_nearest_train_ids':[train[:5] for _ in query],'nearest_correlation':[.999]*7}
    p={'unit_id':'ctx_R_DKL_C','context_id':'ctx','subset_id':'subset','fold':0,'sample_seed':20260922,'n':20,'model_seed':0,'method':'R_DKL_C',
       'candidate_ids':query,'architecture_fingerprints':['a'+c for c in query],
       'mean_raw':[.55+i*.02 for i in range(7)],'mean_standardized':[.25+i*.1 for i in range(7)],
       'latent_variance_standardized':[.25]*7,'latent_variance_raw':[.01]*7,
       'y_center':.5,'y_scale':.2,'extra_noise_variance_standardized':.04,
       'endpoint_state_hash':'a'*64,'training_ids_sha256':score.digest(train),'state_path':'states/ctx.pt','state_file_sha256':'b'*64,
       'cost':{},'geometry_row':nn}
    return p,nn,labels

def science_means():
    common={'gain5_pp':1.,'break_best_rate':.5,'random_regret5_pp':4.,'random_gain5_pp':.8,'random_break_best_probability':.3}
    return {'R_DKL_C':dict(common,regret5_pp=1),'R_REG':dict(common,regret5_pp=2),'RAW16':dict(common,regret5_pp=3)}

class ResidualScoreTests(unittest.TestCase):
    def test_common_gp_payload_uses_old_validation_without_changing_method(self):
        p,nn,labels=data();p['method']='COMMON_GP'
        row=score.prediction_metrics(p,nn,labels)
        self.assertEqual(row['method'],'COMMON_GP')
        self.assertEqual(row['regret5_pp'],0)

    def test_no_gp_geometry_explicit_none_preserves_E1_and_undefined_rho(self):
        p,nn,labels=data();nn['most_correlated_train_ids']=None;nn['most_correlated_rho']=None;nn['nearest_correlation']=None
        row=score.nearest_metrics(nn,labels)
        self.assertIsNone(row['high_correlation_bad_neighbor_fraction'])
        self.assertAlmostEqual(row['E1_pp'],11.)

    def test_random_exact_regret_gain_and_break_probability(self):
        ys=[.1,.3,.4,.4,.5,.7,.9]
        for incumbent in (.0,.65,.95):
            combinations=list(itertools.combinations(ys,5));best=[max(s) for s in combinations]
            actual=score.random5(ys,incumbent)
            self.assertAlmostEqual(actual['random_regret5_pp'],100*(max(ys)-statistics.fmean(best)))
            self.assertAlmostEqual(actual['random_gain5_pp'],100*statistics.fmean(max(0,v-incumbent) for v in best))
            self.assertAlmostEqual(actual['random_break_best_probability'],statistics.fmean(float(v-incumbent>1e-11) for v in best))
        # Positive-part expectation cannot be replaced by positive-part of mean.
        actual=score.random5(ys,.86)['random_gain5_pp']
        wrong=100*max(0,statistics.fmean(max(s) for s in itertools.combinations(ys,5))-.86)
        self.assertGreater(actual,wrong)

    def test_gain_headroom_and_prediction_units(self):
        p,nn,labels=data();r=score.prediction_metrics(p,nn,labels)
        self.assertAlmostEqual(r['gain5_pp'],7.5);self.assertAlmostEqual(r['headroom_pp'],7.5)
        self.assertEqual(r['break_best_rate'],1);self.assertEqual(r['has_headroom_fraction'],1)
        self.assertEqual(r['regret5_pp'],0)

    def test_observed_variance_clips_only_roundoff_and_extra_is_variance(self):
        p,nn,labels=data();p['latent_variance_standardized']=[-1e-9]*7;p['latent_variance_raw']=[-.2**2*1e-9]*7
        r=score.prediction_metrics(p,nn,labels)
        variance=.2**2*.04+.02**2
        self.assertAlmostEqual(r['mean_observed_variance_raw'],variance)
        self.assertEqual(r['negative_latent_roundoff_count'],7)
        self.assertAlmostEqual(r['interval90_width_pp'],200*1.6448536269514722*math.sqrt(variance))
        p['latent_variance_standardized'][0]=-1.01e-8
        with self.assertRaises(ValueError):score.prediction_metrics(p,nn,labels)

    def test_interval_score_penalty_uses_same_observed_variance(self):
        p,nn,labels=data();p['mean_raw']=[x+.3 for x in p['mean_raw']];p['mean_standardized']=[x+1.5 for x in p['mean_standardized']]
        r=score.prediction_metrics(p,nn,labels);h=1.6448536269514722*math.sqrt(.2**2*(.25+.04)+.02**2)
        self.assertAlmostEqual(r['IS90_pp'],100*(2*h+20*(.3-h)))
        self.assertEqual(r['coverage90_fraction'],0)

    def test_nearest_error_uses_selected_training_and_actual_most_correlated(self):
        p,nn,labels=data();r=score.nearest_metrics(nn,labels)
        expected=100*statistics.fmean(abs(labels[q]['validation_mean']-.5) for q in nn['query_ids'])
        self.assertAlmostEqual(r['E1_pp'],expected)
        nn['most_correlated_train_ids']=[nn['training_ids'][-1]]*7;nn['most_correlated_rho']=[.999]*7
        r=score.nearest_metrics(nn,labels)
        self.assertAlmostEqual(r['high_correlation_bad_neighbor_fraction'],statistics.fmean(float(abs(labels[q]['validation_mean']-.595)>.05) for q in nn['query_ids']))
        nn['nearest_train_ids'][0]='unselected'
        with self.assertRaises(ValueError):score.nearest_metrics(nn,labels)

    def test_gate_any_of_four_comparisons_and_four_of_five_required(self):
        deltas={(n,m):dict(enumerate([0.,0.,0.,0.,0.])) for n in (20,50) for m in score.GATE_METHODS}
        deltas[(20,'COMMON_REG')]=dict(enumerate([1.,1.,1.,0.,0.]))
        self.assertEqual(score.gate_decision(deltas,True)['route'],'STOP_DIAGNOSTIC_ONLY')
        deltas[(50,'OLD_DKL_1000')]=dict(enumerate([1.,1.,1.,1.,-.2]))
        self.assertEqual(score.gate_decision(deltas,True)['route'],'RUN_RESIDUAL_ABLATION')
        self.assertEqual(score.gate_decision(deltas,False)['route'],'BLOCKED')

    def test_gate_direction_and_numeric_tolerance_not_rounded(self):
        deltas={(n,m):dict(enumerate([-1.]*5)) for n in (20,50) for m in score.GATE_METHODS}
        deltas[(20,'COMMON_REG')]=dict(enumerate([1e-10]*5))
        self.assertEqual(score.gate_decision(deltas,True)['route'],'STOP_DIAGNOSTIC_ONLY')

    def test_selection_followup_and_mechanism_axes_separate(self):
        seeds={'sample_seeds':{'a':1,'b':1},'model_seeds':{'0':1,'1':1,'2':0}}
        means=science_means();r=score.classify([1,1,1,1,0],[0]*5,means,seeds,True)
        self.assertTrue(r['selection_signal']);self.assertTrue(r['followup_candidate']);self.assertFalse(r['constraint_signal_pass'])
        means['R_DKL_C']['gain5_pp']=.5
        r=score.classify([1,1,1,1,0],[1,1,1,1,0],means,seeds,True)
        self.assertTrue(r['selection_signal']);self.assertFalse(r['followup_candidate']);self.assertTrue(r['constraint_signal_pass'])

    def test_seed_followup_requires_strict_positive_not_effect_threshold(self):
        seeds={'sample_seeds':{'a':1e-12,'b':1e-12},'model_seeds':{'0':1e-12,'1':1e-12,'2':0}}
        self.assertTrue(score.classify([1]*5,[1]*5,science_means(),seeds,True)['followup_candidate'])
        seeds['sample_seeds']['b']=0
        self.assertFalse(score.classify([1]*5,[1]*5,science_means(),seeds,True)['followup_candidate'])
        self.assertIsNone(score.classify([],[],{}, {},False)['followup_candidate'])

    def test_complete_260_hierarchy_raw_reuse_and_pair_directions(self):
        rows=[]
        for fold,sample,n,seed,method in sorted(score.expected(score.METHODS),key=str):
            r={'unit_id':f'{fold}_{sample}_{n}_{seed}_{method}','context_id':f'{fold}_{sample}_{n}_{seed}',
               'subset_id':f'{fold}_{sample}_{n}','fold':fold,'sample_seed':sample,'n':n,'model_seed':seed,'method':method}
            r.update({metric:1. for metric in score.METRICS})
            r.update(regret5_pp={'R_DKL_C':1.,'R_REG':2.,'R_DKL_U':3.,'R_FROZEN':4.,'RAW16':2.5}[method],
                gain5_pp=1.,break_best_rate=.5,random_regret5_pp=3.,random_gain5_pp=.8,random_break_best_probability=.3)
            rows.append(r)
        summary,tables=score.formal_summary(rows,True)
        self.assertEqual(summary['endpoint_count'],260);self.assertEqual(summary['raw_endpoints_counted_once'],20)
        self.assertEqual(summary['per_n']['20']['paired_comparisons']['R_REG']['metrics']['regret5_improvement_pp']['mean'],1.)
        self.assertTrue(summary['per_n']['20']['followup_candidate'])
        pairs=[r for r in tables['paired_unit_metrics.csv'] if r['method']=='RAW16']
        self.assertEqual(len(pairs),60);self.assertEqual(len({r['reference_unit_id'] for r in pairs}),20)

    def test_three_level_equal_fold_and_sample_weights(self):
        rows=[]
        for f,s,values in ((0,20260922,[1,3,5]),(0,20260923,[9]),(1,20260922,[20])):
            rows.extend({'n':20,'method':'R_DKL_C','fold':f,'sample_seed':s,'model_seed':i,'x':v} for i,v in enumerate(values))
        _,fold=score.hierarchy(rows,('x',))
        self.assertEqual(score.base.mean([r['x'] for r in fold]),13)

    def test_new_cost_includes_common_and_diagnostics_once(self):
        units=[]
        for method in score.METHODS:
            p,_,_=data();p['method']=method;p['unit_id']=method
            p['cost']={'common_training_seconds':0 if method=='RAW16' else 1.,'branch_training_seconds':2.,
                'common_model_seconds':0 if method=='RAW16' else 1.5,'branch_model_seconds':2.4,
                'prediction_seconds':0.,'standalone_fit_predict_seconds':2.4 if method=='RAW16' else 3.9}
            if method=='RAW16':p['model_seed']=None
            units.append(p)
        rows,summary=score.residual_costs(units)
        self.assertAlmostEqual(summary['completed_endpoint_shared_model_seconds'],1.5+5*2.4)
        self.assertAlmostEqual(rows[0]['standalone_model_diagnostics_and_other_seconds'],.9)

    def test_unsealed_A_rejects_before_label_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'historical_manifest.json').write_text(json.dumps({'sealed':False}))
            with mock.patch.object(score,'load_labels') as labels:
                with self.assertRaises(ValueError):score.historical_score(root)
                labels.assert_not_called()

    def test_admission_contains_no_truth_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate=score.gate_decision({},False);gate.update(protocol_sha256='a'*64,code_manifest_sha256='b'*64)
            root=Path(tmp);score.write_gate(root,gate)
            admission=json.loads((root/'admission.json').read_text())
            self.assertEqual(set(admission),{'route','diagnostic_gate_sha256','protocol_sha256','code_manifest_sha256','diagnostic_complete'})

if __name__=='__main__':unittest.main(verbosity=2)
