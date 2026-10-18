"""High-detail crop verification shared by training and inference."""
from types import SimpleNamespace
import numpy as np
from .verdict_classifier import PeakFeatureExtractor

KIND="crop-peak-400k-v1"

def extract(base,builder,peak):
    _,metadata=builder.grid(peak)
    roi=metadata['roi_normalized']
    # Same ROI at all times, retaining a 5% border for people entering the workspace.
    roi=[max(0,roi[0]-.05),max(0,roi[1]-.05),min(1,roi[2]+.05),min(1,roi[3]+.05)]
    cropped=[]
    stamps=np.array([f.timestamp_sec for f in builder.video.frames])
    indices=sorted(set(int(np.abs(stamps-np.clip(peak+dt,stamps[0],stamps[-1])).argmin()) for dt in (-6,-2,0,2,6)))
    for i in indices:
        f=builder.video.frames[i];w,h=f.image.size
        cropped.append(SimpleNamespace(timestamp_sec=f.timestamp_sec,image=f.image.crop((int(roi[0]*w),int(roi[1]*h),int(roi[2]*w),int(roi[3]*h)))))
    proxy=SimpleNamespace(video=SimpleNamespace(frames=cropped),font=builder.font)
    x,meta=PeakFeatureExtractor(base,max_pixels=401408).extract(proxy,peak)
    return x,{**meta,'roi':roi,'kind':KIND}

