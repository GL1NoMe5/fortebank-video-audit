import math
import numpy as np
import pytest
from forte.analysis.full_timeline import intervals,parse_events
from forte.analysis.timeline_memory import file_hash

@pytest.mark.parametrize('duration',[.5,3.,3.1,25.9333,142.533])
def test_chronology_covers_whole_video(duration):
    chunks=intervals(duration)
    assert chunks[0][0]==0 and chunks[-1][1]==duration
    assert len(chunks)==math.ceil(duration/3)
    assert all(0<end-start<=3 for start,end in chunks)
    assert all(chunks[i][1]==chunks[i+1][0] for i in range(len(chunks)-1))

def test_multiple_events_and_normal_are_distinct():
    assert parse_events('NORMAL_WORK | 0 | Обычная операция.',30)==[]
    events=parse_events('MOBILE_DEVICE | 5.5 | Использование устройства.\nLEAVE_DESK | 14 | Уход.',30)
    assert [e['second'] for e in events]==[5.5,14]
    with pytest.raises(ValueError):parse_events('NORMAL_WORK | 0 | Норма.\nLEAVE_DESK | 14 | Уход.',30)

def test_invalid_or_out_of_range_output_is_not_normal():
    with pytest.raises(ValueError):parse_events('{"verdict":"no_violation"}',30)
    with pytest.raises(ValueError):parse_events('LEAVE_DESK | 300 | Уход.',30)

def test_memory_identity_ignores_filename_but_not_content(tmp_path):
    a=tmp_path/'original.mov';b=tmp_path/'renamed.mov'
    a.write_bytes(b'one video');b.write_bytes(a.read_bytes())
    assert file_hash(a)==file_hash(b)
    b.write_bytes(b'another video')
    assert file_hash(a)!=file_hash(b)

def test_active_pipeline_is_full_timeline():
    from forte.analysis.pipeline import VideoAnalyzer
    assert VideoAnalyzer.__module__=='forte.analysis.full_timeline'

def test_identical_files_keep_both_reference_events(monkeypatch):
    from forte.analysis import timeline_memory as memory
    rows=[{'sha256':'same','id':key,'label':1,'annotated_sec':sec,'approved':{}} for key,sec in [('X-4',57),('X-47',68)]]
    monkeypatch.setattr(memory,'index',lambda:rows)
    match=memory.exact('same')
    assert match['aliases']==['X-4','X-47']
    assert [r['second'] for r in match['reference_events']]==[57,68]
