import json
from types import SimpleNamespace
from fastapi.testclient import TestClient
from forte.api import create_app
from forte.settings import Settings
from forte.analysis import timeline_memory

def test_timeline_requires_key_and_exposes_provenance(tmp_path,monkeypatch):
    settings=Settings(root_dir=tmp_path,runtime_dir=tmp_path/'runtime',api_key='local-test-secret')
    app=create_app(settings)
    video=tmp_path/'upload.mov';video.write_bytes(b'video')
    app.state.job_manager=SimpleNamespace(get_job_record=lambda job:SimpleNamespace(status='done',video_path=video))
    memory=tmp_path/'memory';folder=memory/timeline_memory.file_hash(video);folder.mkdir(parents=True)
    (folder/'chronology.json').write_text(json.dumps({'intervals':[{'start':0,'end':3,'observations':'Synthetic visual observation'}]}),encoding='utf-8')
    monkeypatch.setattr(timeline_memory,'MEMORY',memory)
    metadata=timeline_memory.request_metadata_path(video,timeline_memory.file_hash(video))
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({'metrics':{'known_label_used':True,'reference_id':'synthetic'}}),encoding='utf-8')
    client=TestClient(app)
    try:
        assert client.get('/api/v1/jobs/test/timeline').status_code==401
        response=client.get('/api/v1/jobs/test/timeline',headers={'X-API-Key':'local-test-secret'})
        assert response.status_code==200
        assert response.json()['metrics']['known_label_used'] is True
        assert response.json()['intervals'][0]['end']==3
    finally:client.close()
