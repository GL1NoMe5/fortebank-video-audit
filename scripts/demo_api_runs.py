"""Upload three real local files to the local API; keep result contents private."""
import html
import json
import re
import time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]

def main():
    key=(ROOT/'runtime/local_demo/api_key.txt').read_text().strip()
    session=requests.Session()
    session.trust_env=False
    session.headers['X-API-Key']=key
    base='http://127.0.0.1:8002'
    deadline=time.monotonic()+300
    while True:
        try:
            if session.get(base+'/healthz',timeout=5).json().get('ready'): break
        except requests.RequestException: pass
        if time.monotonic()>deadline: raise RuntimeError('Local API not ready')
        time.sleep(2)
    records=json.loads((ROOT/'data/dataset_manifest.json').read_text(encoding='utf-8-sig'))['records']
    result_rows=[]
    for number in (1,2,3):
        record=next(r for r in records if int(re.search(r'\d+',r['id']).group())==number)
        with (ROOT/record['path']).open('rb') as handle:
            response=session.post(base+'/api/v1/analyze',files={'file':(Path(record['path']).name,handle,'video/quicktime')},timeout=60)
        response.raise_for_status()
        job=response.json()['job_id']
        start=time.monotonic()
        while True:
            response=session.get(base+'/api/v1/jobs/'+job,timeout=15)
            response.raise_for_status()
            result=response.json()
            if result['status'] in ('done','failed'):break
            if time.monotonic()-start>310:raise TimeoutError('Demo job deadline')
            time.sleep(2)
        (ROOT/'runtime/local_demo'/f'result_{number}.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        row={'video_id':record['id'],'expected':record['label'],'job_id':job,'status':result['status'],
             'verdict':result.get('verdict'),'events':len(result.get('violations',[])),'seconds':round(time.monotonic()-start,1)}
        result_rows.append(row)
        (ROOT/'runtime/local_demo/demo_summary.json').write_text(json.dumps(result_rows,ensure_ascii=False,indent=2),encoding='utf-8')
        cards=[]
        for r in result_rows:
            url=base+'/#key='+key+'&job='+r['job_id']
            cards.append('<article><h2>'+html.escape(r['video_id'])+'</h2><p>Метка таблицы: '+str(r['expected'])+' · Результат API: '+html.escape(str(r['verdict'] or r['status']))+' · событий: '+str(r['events'])+' · '+str(r['seconds'])+' с</p><a target="_blank" href="'+html.escape(url)+'">Открыть результат в основном интерфейсе</a></article>')
        page='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Локальные результаты</title><style>body{font:20px system-ui;background:#f7f2f4;padding:28px}article{padding:24px;background:white;margin:20px 0;border-radius:16px}a{color:#951b4c}</style><h1>Реальные прогоны через локальный API</h1><p>Это демонстрация на знакомых примерах, не независимый тест качества. Положительная метка при загрузке в API не передаётся.</p><p><a href="visual_report.html">Отчёт авторазметки 22 роликов</a></p>'+''.join(cards)+'</html>'
        (ROOT/'runtime/local_visual/demo_results.html').write_text(page,encoding='utf-8')
        print(json.dumps(row,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
