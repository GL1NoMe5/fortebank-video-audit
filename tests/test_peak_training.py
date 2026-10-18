import numpy as np
from forte.analysis.peak_training import PeakColumns, threshold_for, metric

def test_balanced_threshold_does_not_accept_everything_for_f1():
    y=np.array([1,1,1,0,0,0])
    scores=np.array([.3,.7,.8,.4,.5,.6])
    threshold,metrics=threshold_for(y,scores)
    assert .6<threshold<.7
    assert metrics['fp']==0 and metrics['tp']==2

def test_constant_scores_report_chance_not_success():
    _,metrics=threshold_for(np.array([0,0,1,1]),np.ones(4)*.5)
    assert metrics['balanced_accuracy']==.5

def test_bounded_features_do_not_amplify_near_constant_columns():
    x=np.ones((4,4120),dtype=np.float32)
    x[:,0]=[0,1e-9,0,0]
    x[:,12:24]=100
    z=PeakColumns('scores').fit_transform(x)
    assert z.shape==(4,24)
    assert np.max(z[:,0])<1e-8
    assert np.max(z[:,12:])==1

def test_metric_counts_false_negatives():
    m=metric(np.array([1,1,0,0]),np.array([1,0,0,0]))
    assert (m['tp'],m['fn'],m['tn'],m['fp'])==(1,1,2,0)
