from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from forte.analysis import window_classifier as window
from forte.analysis import verdict_classifier as verdict
from forte.analysis import peak_crop
from forte.analysis.temporal_verdict import combine,SEM_DIM,PEAK_DIM

def test_temporal_features_handle_short_clip_without_nan():
    feature=np.zeros(PEAK_DIM)
    result=combine(feature,feature.reshape(1,-1),np.array([.5]),.5)
    assert result.shape==(PEAK_DIM+SEM_DIM+2*4096,)
    assert np.isfinite(result).all()
