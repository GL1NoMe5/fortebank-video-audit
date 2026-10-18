"""CPU-only peak classifier selection; test videos never enter calibration."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score

class PeakColumns(TransformerMixin, BaseEstimator):
    def __init__(self, mode='scores'):
        self.mode=mode
    def fit(self,x,y=None):
        # No per-column division by almost-zero variance; probabilities remain bounded.
        return self
    def transform(self,x):
        x=np.asarray(x,dtype=np.float64)
        scores=np.concatenate([x[:,:12],np.clip(x[:,12:24],-12,12)/12],axis=1)
        hidden=x[:,24:]
        hidden=hidden/np.maximum(np.linalg.norm(hidden,axis=1,keepdims=True),1e-12)*10
        if self.mode=='scores':return scores
        if self.mode=='hidden':return hidden
        return np.concatenate([scores,hidden],axis=1)

CANDIDATES=[{'mode':mode,'C':c,'kind':'logistic'} for mode in ('scores','hidden','combined') for c in (.1,1.,10.)]
CANDIDATES += [{'mode':'scores','kind':'trees','leaf':leaf} for leaf in (3,6)]

def make_model(spec):
    if spec['kind']=='trees':
        estimator=ExtraTreesClassifier(n_estimators=100,min_samples_leaf=spec['leaf'],max_features=1.,class_weight='balanced',random_state=928,n_jobs=1)
    else:estimator=LogisticRegression(C=spec['C'],class_weight='balanced',max_iter=1500,solver='liblinear',random_state=928)
    return make_pipeline(PeakColumns(spec['mode']),estimator)

def metric(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    recall=tp/max(1,tp+fn);specificity=tn/max(1,tn+fp)
    precision=tp/max(1,tp+fp)
    return {'balanced_accuracy':float((recall+specificity)/2),'recall':float(recall),'specificity':float(specificity),
            'precision':float(precision),'f1':float(2*tp/max(1,2*tp+fp+fn)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def threshold_for(y,scores):
    unique=np.unique(scores)
    # Midpoints avoid the old zero-threshold / ties behavior.
    thresholds=np.r_[np.nextafter(unique[0],-np.inf),(unique[:-1]+unique[1:])/2,np.nextafter(unique[-1],np.inf)]
    choices=[(float(t),metric(y,scores>=t)) for t in thresholds]
    # Sensitivity and specificity are equally important; recall breaks ties.
    return max(choices,key=lambda v:(v[1]['balanced_accuracy'],v[1]['recall'],v[1]['f1']))

def select(x,y):
    folds=list(RepeatedStratifiedKFold(n_splits=4,n_repeats=2,random_state=928).split(x,y))
    candidates=[]
    for spec in CANDIDATES:
        scores=np.zeros(len(y));counts=np.zeros(len(y))
        for tr,va in folds:
            model=make_model(spec);model.fit(x[tr],y[tr]);scores[va]+=model.predict_proba(x[va])[:,1];counts[va]+=1
        scores/=counts
        threshold,metrics=threshold_for(y,scores)
        candidates.append({'spec':spec,'threshold':threshold,'metrics':metrics,'auc':float(roc_auc_score(y,scores)),'oof_scores':scores.tolist()})
    best=max(candidates,key=lambda r:(r['metrics']['balanced_accuracy'],r['metrics']['recall'],r['auc']))
    return best,candidates

def fit(records,features,stage1_hash,version,settings_hash):
    x=np.asarray(features);y=np.array([int(r['label']) for r in records])
    if len(records)!=34 or len(set(r['id'] for r in records))!=34:raise ValueError('Expected 34 unique training videos')
    nested=np.zeros(len(y),dtype=int);outer_details=[]
    # Outer folds audit the WHOLE model/threshold selection procedure, without test8.
    for i,(tr,va) in enumerate(StratifiedKFold(n_splits=4,shuffle=True,random_state=2718).split(x,y),1):
        best,_=select(x[tr],y[tr]);model=make_model(best['spec']);model.fit(x[tr],y[tr])
        nested[va]=model.predict_proba(x[va])[:,1]>=best['threshold']
        outer_details.append({'fold':i,'selected':best['spec'],'threshold':best['threshold']})
        print(f'Nested training audit {i}/4 complete',flush=True)
    best,candidates=select(x,y)
    model=make_model(best['spec']);model.fit(x,y)
    return {'model':model,'threshold':best['threshold'],'version':version,'settings_sha256':settings_hash,
            'stage1_sha256':stage1_hash,'train_ids':[r['id'] for r in records],
            'oof_scores':best['oof_scores'],'oof_metrics':best['metrics'],'selected':best['spec'],'candidates':candidates,
            'nested_metrics':metric(y,nested),'nested_predictions':nested.tolist(),'nested_folds':outer_details,
            'calibration':'Fixed 11 candidate models; 2x4-fold OOF balanced-accuracy threshold selection on train34 only. Four outer folds audit selection. Stage1 peaks remain in-sample; this is not nested validation of the whole two-stage system.'}
