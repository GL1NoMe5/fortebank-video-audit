import json
import numpy as np
import pytest
from forte.analysis.window_features import centers,nearest_indices,OFFSETS
from forte.analysis.window_classifier import training_mask,choose_threshold,decision
from forte.analysis.video import VideoFrame,SampledVideo
from PIL import Image

def test_window_stride_context_and_no_zero_center():
    ts=centers(11)
    assert np.array_equal(ts,[2,4,6,8,10])
    assert np.all(np.diff(ts)==2) and min(ts)>0
    frames=[VideoFrame(i/2,Image.new('RGB',(10,10))) for i in range(23)]
    video=SampledVideo(frames,11,10,10)
    assert [frames[i].timestamp_sec for i in nearest_indices(video,10)]==[2,8.5,10,11]
    assert [frames[i].timestamp_sec for i in nearest_indices(video,2)]==[0,.5,2,3.5]
    assert centers(1)[0]==.5

def test_only_annotated_positive_neighborhood_and_all_clean_windows():
    times=np.array([2,4,6,8,10,12])
    assert training_mask(times,1,7).tolist()==[False,True,True,True,True,False]
    assert training_mask(times,0,None).all()

def test_threshold_and_peak_are_numeric_not_generated():
    threshold,m=choose_threshold([0,0,1,1],np.array([.1,.2,.7,.9]))
    assert threshold==.7 and m['f1']==1
    class Model:
        def predict_proba(self,features):return np.c_[1-features[:,0],features[:,0]]
    d=decision({'model':Model(),'threshold':.7},np.array([[.1],[.8],[.3]]),np.array([2,4,6]))
    assert d['violation'] and d['peak_time']==4
    assert json.loads(json.dumps({'timecode_sec':d['peak_time']}))['timecode_sec']==4

def test_active_pipeline_uses_no_qlora_or_json_generator():
    from forte.analysis.pipeline import VideoAnalyzer
    from forte.analysis.window_classifier import VideoAnalyzer as Expected
    from forte.analysis.vlm import LocalVLM
    from forte.analysis.config import AnalysisConfig
    assert VideoAnalyzer is Expected
    with pytest.raises(RuntimeError,match='JSON generation is disabled'):
        LocalVLM(AnalysisConfig.from_env()).generate([], '',max_pixels=1,max_tokens=1,deadline=0)

def test_classifier_fits_only_supplied_training_video_groups():
    from forte.analysis.window_classifier import fit_classifier
    rng=np.random.default_rng(42);records=[];features={}
    for i in range(34):
        label=int(i<18);key=f'synthetic-{i}'
        records.append({'id':key,'label':label,'annotated_sec':6 if label else None})
        values=rng.normal(size=(6,44)).astype(np.float32)
        values[:,0]+=label*4
        features[key]={'times':np.arange(2,14,2),'features':values}
    artifact=fit_classifier(records,features)
    assert artifact['train_ids']==[r['id'] for r in records]
    assert artifact['positive_training_windows']==54
    assert artifact['training_windows']==54+16*6
    assert all(len(c['oof_video_scores'])==34 for c in artifact['candidates'])
    assert np.isfinite(artifact['threshold'])
