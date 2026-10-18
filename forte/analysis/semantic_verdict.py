"""Sign-constrained action classifier with a prior for sparsely represented actions."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
from .peak_training import metric
from .fused_verdict import threshold

class SemanticClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,strength=1.,prior=1.):self.strength=strength;self.prior=prior
    def design(self,x):
        x=np.asarray(x)
        # Pool both views; same sign for each action regardless of camera context.
        p=(x[:,:12]+x[:,4120:4132])/2
        signs=np.array([-1,-1,-1,1,1,1,1,-1,1,1,1,-1])
        return p*signs
    def fit(self,x,y):
        a=self.design(x);y=np.asarray(y);weight=np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
        prior=np.ones(12)*self.prior
        def loss(z):
            logit=a@z[:-1]+z[-1];error=(expit(logit)-y)*weight/len(y)
            value=np.mean(weight*(np.logaddexp(0,logit)-y*logit))+self.strength*np.sum((z[:-1]-prior)**2)/(2*len(y))
            grad=np.r_[a.T@error+self.strength*(z[:-1]-prior)/len(y),error.sum()]
            return value,grad
        result=minimize(loss,np.r_[prior,0.],jac=True,bounds=[(0,None)]*12+[(None,None)],method='L-BFGS-B',options={'maxiter':500})
        if not result.success:raise RuntimeError(result.message)
        self.coefficients_=result.x;self.classes_=np.array([0,1]);return self
    def predict_proba(self,x):
        p=expit(self.design(x)@self.coefficients_[:-1]+self.coefficients_[-1]);return np.c_[1-p,p]

SPECS=[{'strength':s,'prior':p} for s in (.1,1.,10.) for p in (.5,2.)]
def select(x,y):
    choices=[];folds=list(RepeatedStratifiedKFold(n_splits=4,n_repeats=2,random_state=733).split(x,y))
    for spec in SPECS:
        scores=np.zeros(len(y));count=np.zeros(len(y))
        for tr,va in folds:
            m=SemanticClassifier(**spec).fit(x[tr],y[tr]);scores[va]+=m.predict_proba(x[va])[:,1];count[va]+=1
        scores/=count;t,metrics=threshold(y,scores);choices.append({'spec':spec,'threshold':t,'metrics':metrics,'oof_scores':scores.tolist()})
    eligible=[c for c in choices if c['metrics']['recall']>=2/3 and c['metrics']['specificity']>=.25]
    return max(eligible or choices,key=lambda c:(c['metrics']['balanced_accuracy'],c['metrics']['recall'])),choices

def fit(records,x,digest,version):
    x=np.asarray(x);y=np.array([int(r['label']) for r in records]);pred=np.zeros(len(y),int)
    for tr,va in StratifiedKFold(n_splits=4,shuffle=True,random_state=2718).split(x,y):
        b,_=select(x[tr],y[tr]);m=SemanticClassifier(**b['spec']).fit(x[tr],y[tr]);pred[va]=m.predict_proba(x[va])[:,1]>=b['threshold']
    b,choices=select(x,y);m=SemanticClassifier(**b['spec']).fit(x,y)
    return {'model':m,'threshold':b['threshold'],'version':version,'settings_sha256':'full-and-crop-v1','feature_kind':'full-and-crop-v1',
            'stage1_sha256':digest,'train_ids':[r['id'] for r in records],'selected':b['spec'],'oof_metrics':b['metrics'],'candidates':choices,
            'nested_metrics':metric(y,pred),'calibration':'Sign constraints encode visible-action priors. Train34 only, 2x4 OOF threshold selection with recall/specificity constraints; four outer folds. Repeated benchmark is not independent.'}
