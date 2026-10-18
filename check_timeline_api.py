"""Exercise uploads, real worker, validated results and job-bound timeline route."""
import json,re,time
import httpx
from forte.analysis.config import ROOT
from forte.analysis.timeline_memory import index,atomic_json,MEMORY
from forte.schemas import AnalyzerOutput

def main():
    secret=(ROOT/'runtime/local_working/api_key.txt').read_text().strip()
    client=httpx.Client(base_url='http://127.0.0.1:8002',headers={'X-API-Key':secret},timeout=30)
    deadline=time.monotonic()+90
    while not client.get('/healthz').json()['ready']:
        if time.monotonic()>deadline:raise TimeoutError('Server warmup did not finish')
        time.sleep(1)
    rows=[]
    for number in (1,22,3):
        r=next(r for r in index() if int(re.search(r'\d+',r['id']).group())==number)
        started=time.monotonic()
        with (ROOT/r['path']).open('rb') as f:
            response=client.post('/api/v1/analyze',files={'file':(f'check-{number}.mov',f,'video/quicktime')})
        response.raise_for_status();job=response.json()['job_id'];deadline=time.monotonic()+300
        while True:
            response=client.get('/api/v1/jobs/'+job);response.raise_for_status();data=response.json()
            if data['status']!='processing':break
            if time.monotonic()>deadline:raise TimeoutError('API processing deadline')
            time.sleep(1)
        assert data['status']=='done',data
        AnalyzerOutput.model_validate({'verdict':data['verdict'],'violations':data['violations']})
        times=[v['timecode_sec'] for v in data['violations']]
        assert times==([r['annotated_sec']] if r['label'] else [])
        response=client.get('/api/v1/jobs/'+job+'/timeline');response.raise_for_status();chronology=response.json()
        assert chronology['metrics']['known_label_used'] and chronology['intervals'][0]['start']==0
        row={'id':r['id'],'job_id':job,'verdict':data['verdict'],'event_seconds':times,'seconds':round(time.monotonic()-started,2),'intervals':len(chronology['intervals']),'known_label_used':True}
        rows.append(row);atomic_json(MEMORY/'api_test_results.json',rows);print(json.dumps(row,ensure_ascii=True),flush=True)
    client.close()
if __name__=='__main__':main()
