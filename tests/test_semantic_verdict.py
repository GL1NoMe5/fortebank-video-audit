import numpy as np
from forte.analysis.semantic_verdict import SemanticClassifier

def test_action_signs_survive_training():
    rng=np.random.default_rng(721)
    x=np.zeros((34,8240));x[:,:12]=rng.random((34,12));x[:,4120:4132]=rng.random((34,12))
    y=np.array([0,1]*17)
    model=SemanticClassifier(strength=10,prior=2).fit(x,y)
    baseline=np.zeros((1,8240))
    cup=baseline.copy();cup[:,3]=1;cup[:,4123]=1
    counted=baseline.copy();counted[:,0]=1;counted[:,4120]=1
    p=model.predict_proba(baseline)[0,1]
    assert model.predict_proba(cup)[0,1]>p
    assert model.predict_proba(counted)[0,1]<p
    assert (model.coefficients_[:-1]>=0).all()
