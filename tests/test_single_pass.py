import pytest
from forte.analysis.single_pass import parse_events
from scripts.inspect_dataset import parse_timestamp

def test_minute_timestamp():
    assert parse_timestamp('1минута 8 секунда') == 68

def test_single_pass_no_confirmed_gate_and_multiple_events():
    event={'rule_id':'r','type':'event','timecode_sec':2.,'description':'A visible action at the desk','confirmed':False}
    assert len(parse_events({'events':[event,dict(event,timecode_sec=4.)]},[1.,2.,4.],{'r':{'title':'event'}}))==2
    assert parse_events({'events':[]},[1.,2.],{})==[]
    with pytest.raises(ValueError):
        parse_events({'events':[dict(event,timecode_sec=80)]},[1.,2.],{'r':{'title':'event'}})
