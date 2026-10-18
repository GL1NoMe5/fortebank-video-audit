"""Temporal verification using existing frozen features; never relocalizes peaks."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from .peak_training import metric, threshold_for

KIND='temporal-peak-v1'
PEAK_DIM=4120
SEM_DIM=144

def combine(peak_feature,windows,times,peak_time):
    x=np.asarray(windows,dtype=np.float64);t=np.asarray(times)
    if x.ndim!=2 or len(x)!=len(t) or x.shape[1]!=PEAK_DIM:raise ValueError('Invalid cached window features')
    p=int(np.abs(t-peak_time).argmin())
    near=np.abs(t-peak_time)<=8
    before=(t<peak_time)&near;after=(t>peak_time)&near
    initial=t<=min(t[0]+6,peak_time)
    def avg(mask):return x[mask,:12].mean(0) if mask.any() else x[p,:12]
    probs=x[:,:12]
    # Fixed features of action persistence and return to baseline. No duration, IDs or GT.
    blocks=[x[p,:12],avg(before),avg(after),avg(after)-avg(before),
            probs[near].max(0),probs[near].min(0),probs[near].std(0),
            probs.mean(0),np.quantile(probs,.9,axis=0),probs[-min(3,len(x)):].mean(0),
            (probs>.5).mean(0),np.abs(np.diff(probs,axis=0)).mean(0) if len(x)>1 else np.zeros(12)]
    h=x[:,24:];h=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
    residual=h[near].mean(0)-h[initial].mean(0)
    trend=(h[after].mean(0) if after.any() else h[p])-(h[before].mean(0) if before.any() else h[p])
    return np.concatenate([peak_feature,*blocks,residual,trend]).astype(np.float32)

class TemporalColumns(TransformerMixin,BaseEstimator):
    def __init__(self,mode='semantic'):self.mode=mode
    def fit(self,x,y=None):return self
    def transform(self,x):
        x=np.asarray(x,dtype=np.float64)
        peak=np.concatenate([x[:,:12],np.clip(x[:,12:24],-12,12)/12],axis=1)
        sem=x[:,PEAK_DIM:PEAK_DIM+SEM_DIM]
        hidden=x[:,24:PEAK_DIM]
        hidden=hidden/np.maximum(np.linalg.norm(hidden,axis=1,keepdims=True),1e-12)*10
        # Differences in normalized visual embeddings are bounded; no unstable whitening.
        delta=x[:,PEAK_DIM+SEM_DIM:]*20
        if self.mode=='semantic':return np.concatenate([peak,sem],axis=1)
        if self.mode=='dynamics':return delta
        if self.mode=='combined':return np.concatenate([peak,sem,hidden,delta],axis=1)
        return hidden

SPECS=[{'mode':m,'kind':'linear','C':c} for m in ('semantic','dynamics','combined') for c in (.1,1.,10.)]
SPECS += [{'mode':'semantic','kind':'trees','leaf':v} for v in (3,6)]

def make_model(spec):
    if spec['kind']=='trees':
        clf=ExtraTreesClassifier(n_estimators=150,min_samples_leaf=spec['leaf'],max_features=.7,class_weight='balanced',random_state=1801,n_jobs=1)
    else:clf=LogisticRegression(C=spec['C'],class_weight='balanced',solver='liblinear',max_iter=2000,random_state=1801)
    return make_pipeline(TemporalColumns(spec['mode']),clf)

def select(x,y):
    folds=list(RepeatedStratifiedKFold(n_splits=4,n_repeats=3,random_state=1801).split(x,y))
    options=[]
    for spec in SPECS:
        oof=np.zeros(len(y));counts=np.zeros(len(y))
        for tr,va in folds:
            model=make_model(spec);model.fit(x[tr],y[tr]);oof[va]+=model.predict_proba(x[va])[:,1];counts[va]+=1
        oof/=counts;threshold,metrics=threshold_for(y,oof)
        options.append({'spec':spec,'threshold':threshold,'metrics':metrics,'auc':float(roc_auc_score(y,oof)),'oof_scores':oof.tolist()})
    best=max(options,key=lambda r:(r['metrics']['balanced_accuracy'],r['metrics']['recall'],r['auc']))
    return best,options

def fit(records,features,stage1_hash,version,settings_hash):
    x=np.asarray(features);y=np.array([int(r['label']) for r in records])
    assert len(y)==34 and len(set(r['id'] for r in records))==34
    nested=np.zeros(len(y),dtype=int);details=[]
    for i,(tr,va) in enumerate(StratifiedKFold(n_splits=4,shuffle=True,random_state=2718).split(x,y),1):
        best,_=select(x[tr],y[tr]);model=make_model(best['spec']);model.fit(x[tr],y[tr]);nested[va]=model.predict_proba(x[va])[:,1]>=best['threshold']
        details.append({'fold':i,'selected':best['spec'],'threshold':best['threshold']})
        print(f'Temporal nested audit {i}/4 complete',flush=True)
    best,options=select(x,y);model=make_model(best['spec']);model.fit(x,y)
    return {'model':model,'threshold':best['threshold'],'version':version,'settings_sha256':settings_hash,'stage1_sha256':stage1_hash,
            'feature_kind':KIND,'train_ids':[r['id'] for r in records],'selected':best['spec'],'candidates':options,
            'oof_scores':best['oof_scores'],'oof_metrics':best['metrics'],'nested_metrics':metric(y,nested),'nested_predictions':nested.tolist(),'nested_folds':details,
            'calibration':'Temporal frozen features; 3x4-fold OOF balanced-accuracy selection, 4 outer training folds; no test8 used for selection. Stage1 training peaks are in-sample; not full-system nested validation.'}
