import csv
import json
from types import SimpleNamespace
import pytest
import build_database as build
import train_qlora as train

def test_argon_service_error_recovers_only_complete_structured_draft():
    payload={'type':'category','description':'synthetic visible action','rule_id':'rule','uncertainty':''}
    event={'event':'result','result':{'status':'ERROR','error':'UNAVAILABLE code 503','structured_output':payload}}
    result=build.parse_argon_output(json.dumps(event),[{'id':'rule'}],recovery_only=True)
    assert result['description']==payload['description']
    assert result['uncertainty']
    del payload['description']
    with pytest.raises(ValueError): build.parse_argon_output(json.dumps(event),[{'id':'rule'}])
    event['result']['error']='permission denied'
    with pytest.raises(ValueError): build.parse_argon_output(json.dumps(event),[{'id':'rule'}])

def test_chronology_rejects_one_entry_and_duplicate_indices():
    frames=[SimpleNamespace(timestamp_sec=float(i)) for i in range(16)]
    entries=[{'frame_index':i,'second':float(i),'action':'synthetic action'} for i in range(16)]
    assert len(build.validate_timeline({'timeline':entries},frames,list(range(16))))==16
    with pytest.raises(ValueError): build.validate_timeline({'timeline':entries[:1]},frames,list(range(16)))
    entries[-1]=entries[0]
    with pytest.raises(ValueError): build.validate_timeline({'timeline':entries},frames,list(range(16)))

def test_all_42_supervised_targets_and_disjoint_split(tmp_path,monkeypatch):
    records=[{'id':f'X-{n}','path':f'{n}.mov'} for n in sorted(set(build.TIMES)|build.NEGATIVES)]
    monkeypatch.setattr(build,'manifest',lambda:records)
    monkeypatch.setattr(train,'ROOT',tmp_path);monkeypatch.setattr(train,'OUT',tmp_path/'out')
    with (tmp_path/'violations_database.txt').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,delimiter='|')
        for r in records:
            n=build.number(r);pos=n in build.TIMES
            writer.writerow([r['id'],build.TIMES[n] if pos else '-', 'category' if pos else 'без нарушения','visible synthetic action','reference' if pos else '-'])
            p=tmp_path/'runtime/deep_database'/str(n);p.mkdir(parents=True)
            (p/'complete_v3.json').write_text(json.dumps({'frames':[f'{n}/{i}.jpg' for i in range(16)],'timestamps':list(range(16)), 'observations':{'timeline':[{} for _ in range(16)]}}))
    training,testing=train.read_database()
    assert len(training)==34 and len(testing)==8
    assert not ({r['id'] for r in training}&{r['id'] for r in testing})
    assert sum(not r['positive'] for r in training)==16
    assert sum(not r['positive'] for r in testing)==4
    for row in training+testing:
        assert row['target']['description']=='visible synthetic action'
        if not row['positive']:
            assert row['target']['verdict']=='no_violation' and row['target']['violations']==[]
    assert not (tmp_path/'out/review_approval.json').exists()
