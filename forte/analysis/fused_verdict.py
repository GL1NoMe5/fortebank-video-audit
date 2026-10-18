"""General-scene plus high-detail peak features, trained without holdout labels."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from .peak_training import metric

KIND='full-and-crop-v1'

class FusedColumns(TransformerMixin,BaseEstimator):
    def __init__(self,mode='combined'):self.mode=mode
    def fit(self,x,y=None):return self
    def transform(self,x):
        x=np.asarray(x);base=x[:,:4120];crop=x[:,4120:]
        def sem(a):return np.concatenate([a[:,:12],np.clip(a[:,12:24],-12,12)/12],axis=1)
        def hidden(a):return a[:,24:]/np.maximum(np.linalg.norm(a[:,24:],axis=1,keepdims=True),1e-12)*10
        scores=np.concatenate([sem(base),sem(crop)],axis=1)
        if self.mode=='scores':return scores
        vectors=np.concatenate([hidden(base),hidden(crop)],axis=1)
        if self.mode=='hidden':return vectors
        return np.concatenate([scores,vectors],axis=1)

SPECS=[{'mode':m,'C':c} for m in ('scores','hidden','combined') for c in (.1,1.,10.)]+[{'mode':'scores','leaf':3},{'mode':'scores','leaf':6}]
def model(spec):
    clf=ExtraTreesClassifier(n_estimators=150,min_samples_leaf=spec['leaf'],max_features=.7,class_weight='balanced',random_state=310,n_jobs=1) if 'leaf' in spec else LogisticRegression(C=spec['C'],class_weight='balanced',solver='liblinear',max_iter=2000)
    return make_pipeline(FusedColumns(spec['mode']),clf)

def threshold(y,scores):
    v=np.unique(scores);ts=np.r_[np.nextafter(v[0],-np.inf),(v[:-1]+v[1:])/2,np.nextafter(v[-1],np.inf)]
    all_options=[(float(t),metric(y,scores>=t)) for t in ts]
    options=[p for p in all_options if p[1]['recall']>=2/3 and p[1]['specificity']>=.25]
    return max(options or all_options,key=lambda p:(p[1]['balanced_accuracy'],p[1]['recall'],p[1]['f1']))

def select(x,y):
    folds=list(RepeatedStratifiedKFold(n_splits=4,n_repeats=2,random_state=310).split(x,y));options=[]
    for spec in SPECS:
        scores=np.zeros(len(y));count=np.zeros(len(y))
        for tr,va in folds:
            m=model(spec);m.fit(x[tr],y[tr]);scores[va]+=m.predict_proba(x[va])[:,1];count[va]+=1
        scores/=count;t,metrics=threshold(y,scores)
        options.append({'spec':spec,'threshold':t,'metrics':metrics,'auc':float(roc_auc_score(y,scores)),'oof_scores':scores.tolist()})
    eligible=[v for v in options if v['metrics']['recall']>=2/3 and v['metrics']['specificity']>=.25]
    return max(eligible or options,key=lambda v:(v['metrics']['balanced_accuracy'],v['metrics']['recall'],v['auc'])),options

def fit(records,x,stage1_hash,version):
    x=np.asarray(x);y=np.array([int(r['label']) for r in records]);pred=np.zeros(len(y),int)
    for i,(tr,va) in enumerate(StratifiedKFold(n_splits=4,shuffle=True,random_state=2718).split(x,y),1):
        b,_=select(x[tr],y[tr]);m=model(b['spec']);m.fit(x[tr],y[tr]);pred[va]=m.predict_proba(x[va])[:,1]>=b['threshold'];print(f'Fused audit {i}/4',flush=True)
    b,options=select(x,y);m=model(b['spec']);m.fit(x,y)
    return {'model':m,'threshold':b['threshold'],'version':version,'settings_sha256':KIND,'feature_kind':KIND,'stage1_sha256':stage1_hash,'train_ids':[r['id'] for r in records],
            'selected':b['spec'],'oof_metrics':b['metrics'],'oof_scores':b['oof_scores'],'candidates':options,'nested_metrics':metric(y,pred),'nested_predictions':pred.tolist(),
            'calibration':'2x4 fold OOF on train34; balanced accuracy with recall>=2/3 and specificity>=1/4 where feasible. Four outer training folds; fixed in-sample Stage1 peaks. No test8 threshold tuning.'}
