import numpy as np
import pytest
from PIL import Image,ImageFont
from types import SimpleNamespace
from forte.analysis.final_event import parse_answer,prepare
from forte.analysis import window_classifier as window,final_event

def test_normal_is_not_overridden_by_mentions():
    assert parse_answer('NORMAL_WORK\nСотрудник работает. Телефон не используется.')[0]=='NORMAL_WORK'

def test_action_keyword_corrects_category():
    assert parse_answer('CASH_UNATTENDED_OR_NO_RECOUNT\nСотрудник выходит через дверь. Стол остаётся пустым.')[0]=='LEAVE_DESK'

def test_invalid_category_never_defaults_to_cash():
    with pytest.raises(ValueError):parse_answer('UNKNOWN\nСотрудник работает.')

def test_evidence_distinct_and_event_bounded():
    frames=[SimpleNamespace(timestamp_sec=i*.5,image=Image.new('RGB',(100,100),'gray')) for i in range(21)]
    builder=SimpleNamespace(video=SimpleNamespace(frames=frames,duration_sec=10),font=ImageFont.load_default(),grid=lambda t:(None,{'roi_normalized':[0,0,1,1]}))
    event,_,_,times=prepare(builder,6.)
    assert 1<=event<=6 and len(set(times))==3 and times==sorted(times)

def test_normal_final_overrides_positive_stage1(monkeypatch):
    bundle={'model':SimpleNamespace(predict_proba=lambda x:np.array([[.01,.99]])),'threshold':.1}
    monkeypatch.setattr(final_event,'analyze',lambda *a:({'verdict':'no_violation','violations':[]},{'category':'NORMAL_WORK','event_time':2.}))
    result,d=window.make_result(None,None,bundle,np.zeros((1,4120)),np.array([6.]))
    assert result=={'verdict':'no_violation','violations':[]}
    assert d['category']=='NORMAL_WORK'
