"""Window classifier, train-only threshold selection, Python-owned API results."""
from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path
import joblib
import numpy as np
from .config import ROOT, AnalysisConfig
from .video import sample_video, evidence_frames
from .window_features import WindowFeatureExtractor,WindowBuilder,centers,CATEGORIES,VERSION

ARTIFACT=ROOT/'runtime/window_classifier/classifier.joblib'

def training_mask(times,label,gt):
    if not label:return np.ones(len(times),dtype=bool)
    return np.abs(np.asarray(times)-float(gt))<=3.0

def choose_threshold(labels,scores):
    from sklearn.metrics import f1_score,recall_score,precision_score
    choices=np.unique(np.r_[0.,scores,np.nextafter(max(scores),np.inf)])
    best=None
    for threshold in choices:
        pred=np.asarray(scores)>=threshold
        metrics=(f1_score(labels,pred,zero_division=0),recall_score(labels,pred,zero_division=0),precision_score(labels,pred,zero_division=0))
        if best is None or metrics>best[0]:best=(metrics,float(threshold))
    return best[1],{'f1':best[0][0],'recall':best[0][1],'precision':best[0][2]}

def model_factory(c):
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler,Normalizer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    n=len(CATEGORIES)*2
    transform=ColumnTransformer([
        ('scores',StandardScaler(),slice(0,n)),
        ('hidden',make_pipeline(Normalizer(),PCA(n_components=16,whiten=True,random_state=20261018)),slice(n,None)),
    ])
    return make_pipeline(transform,LogisticRegression(C=c,max_iter=2000,solver='lbfgs'))

def fit_classifier(records,features):
    from sklearn.model_selection import StratifiedKFold
    video_labels=np.array([r['label'] for r in records])
    def training_data(indices):
        xx=[];yy=[];weights=[]
        for i in indices:
            r=records[i];f=features[r['id']];mask=training_mask(f['times'],r['label'],r['annotated_sec'])
            if not mask.any():raise ValueError('Positive video has no window within annotation tolerance')
            xx.append(f['features'][mask]);yy.extend([r['label']]*int(mask.sum()))
            weights.extend([1/float(mask.sum())]*int(mask.sum()))
        weight=np.asarray(weights);weight/=weight.mean()
        return np.concatenate(xx),np.asarray(yy),weight
    candidates=[]
    for c in (.03,.3,3.):
        oof=np.full(len(records),np.nan)
        for train_idx,val_idx in StratifiedKFold(n_splits=4,shuffle=True,random_state=20261018).split(records,video_labels):
            xx,yy,weights=training_data(train_idx);model=model_factory(c)
            model.fit(xx,yy,logisticregression__sample_weight=weights)
            for i in val_idx:oof[i]=model.predict_proba(features[records[i]['id']]['features'])[:,1].max()
        assert np.isfinite(oof).all()
        threshold,metrics=choose_threshold(video_labels,oof)
        candidates.append({'C':c,'threshold':threshold,'metrics':metrics,'oof_video_scores':oof.tolist()})
    selected=max(candidates,key=lambda c:(c['metrics']['f1'],c['metrics']['recall'],c['metrics']['precision'],-c['C']))
    xx,yy,weights=training_data(np.arange(len(records)));model=model_factory(selected['C'])
    model.fit(xx,yy,logisticregression__sample_weight=weights)
    return {'model':model,'threshold':selected['threshold'],'version':VERSION,'categories':CATEGORIES,
            'train_ids':[r['id'] for r in records],'selected':selected,'candidates':candidates,
            'training_windows':len(yy),'positive_training_windows':int(yy.sum()),
            'calibration':'4-fold video-disjoint out-of-fold predictions on 34 training videos only; optimize video-level F1 then recall'}

def decision(bundle,features,times):
    probabilities=bundle['model'].predict_proba(features)[:,1]
    peak=int(np.argmax(probabilities))
    return {'violation':bool(probabilities[peak]>=bundle['threshold']),'peak_index':peak,
            'peak_time':float(times[peak]),'peak_score':float(probabilities[peak]),'scores':probabilities}

def make_result(extractor,builder,bundle,features,times,description=True):
    d=decision(bundle,features,times)
    if not d['violation']:return {'verdict':'no_violation','violations':[]},d
    from .final_event import analyze
    result,details=analyze(extractor,builder,d['peak_time'])
    d.update(details)
    return result,d

class VideoAnalyzer:
    def __init__(self,config=None):
        self.config=config or AnalysisConfig.from_env();self.extractor=WindowFeatureExtractor();self.bundle=None;self.last_metrics={}
    def warmup(self):
        self.bundle=joblib.load(ARTIFACT)
        if self.bundle['version']!=VERSION:raise RuntimeError('Window classifier version mismatch')
        current=hashlib.sha256((ROOT/'forte/analysis/window_features.py').read_bytes()).hexdigest()
        if self.bundle.get('feature_settings_sha256')!=current:raise RuntimeError('Classifier requires features from the current extractor; retrain the classifier')
        self.extractor.load()
    def analyze(self,video_path:Path,progress=None):
        if self.bundle is None:self.warmup()
        start=time.monotonic();video=sample_video(video_path);builder=WindowBuilder(video);times=centers(video.duration_sec);features=[]
        for i,t in enumerate(times):
            if time.monotonic()-start>self.config.timeout_sec:raise TimeoutError('Window analysis deadline')
            grid,_=builder.grid(float(t));feature,_=self.extractor.extract(grid);features.append(feature)
            if progress:progress('window_scoring',.05+.85*(i+1)/len(times))
        result,d=make_result(self.extractor,builder,self.bundle,np.asarray(features),times)
        self.last_metrics={'seconds':round(time.monotonic()-start,3),'windows':len(times),'peak_score':d['peak_score'],
                           'threshold':self.bundle['threshold'],'qlora_enabled':False,'json_generation':False,
                           'final_category':d.get('category'),'event_time':d.get('event_time'),'peak_time':d['peak_time']}
        return result
