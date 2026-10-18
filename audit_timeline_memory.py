"""Validate coverage and actual visual vectors for every reference entry."""
import json,math
from pathlib import Path
import numpy as np
from forte.analysis.timeline_memory import index,MEMORY,atomic_json

def main():
    profiles=index();rows=[]
    for r in profiles:
        folder=Path(r['profile_dir']);doc=json.loads((folder/'chronology.json').read_text(encoding='utf-8'))
        assert doc['complete'] and len(doc['intervals'])==math.ceil(doc['duration']/3)
        assert doc['intervals'][0]['start']==0 and doc['intervals'][-1]['end']==doc['duration']
        for a,b in zip(doc['intervals'],doc['intervals'][1:]):assert a['end']==b['start']
        with np.load(folder/'visual.npz') as v:
            assert len(v['embeddings'])==4*len(doc['intervals'])
            assert v['embeddings'].ndim==2 and np.isfinite(v['embeddings']).all()
            assert np.linalg.norm(v['embeddings'],axis=1).min()>0
            assert len(v['frame_times'])==len(v['embeddings'])
            dimension=v['embeddings'].shape[1]
        rows.append({'id':r['id'],'intervals':len(doc['intervals']),'dimension':dimension})
    result={'reference_entries':len(rows),'unique_video_files':len({r['sha256'] for r in profiles}),'all_complete':True,'rows':rows}
    atomic_json(MEMORY/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
