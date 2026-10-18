"""Local report explicitly separates visual observations from known annotations."""
import html,json
from pathlib import Path
from forte.analysis.config import ROOT
from forte.analysis.timeline_memory import index,MEMORY

def main():
    cards=[];count=0
    for r in index():
        folder=Path(r['profile_dir']);cp=folder/'chronology.json'
        if not cp.exists():continue
        c=json.loads(cp.read_text(encoding='utf-8'))
        if not c.get('complete'):continue
        count+=1
        text=''.join('<tr><td>'+f"{v['start']:.1f}–{v['end']:.1f}"+'</td><td>'+html.escape(v['observations'])+'</td></tr>' for v in c['intervals'])
        ap=folder/'last_analysis.json';result=''
        if ap.exists():
            a=json.loads(ap.read_text(encoding='utf-8'));m=a['metrics'];p=a['result']
            result='<p>Результат: '+html.escape(p['verdict'])+'; эталон: '+html.escape(str(m['reference_id']))+'; время '+str(m['seconds'])+' с; хроника из кэша: '+str(m['chronology_cached'])+'</p>'
            if m.get('reference_model_disagreement'):
                result+='<p class="annotation">Визуальная оценка модели расходится с известной разметкой. Итог откалиброван по эталону; проверьте кадры и хронику.</p>'
            for v in p['violations']:
                result+='<h3>'+html.escape(v['type'])+' — '+str(v['timecode_sec'])+' с</h3><p>'+html.escape(v['description'])+'</p>'+''.join('<img src="data:image/jpeg;base64,'+e['image_base64']+'">' for e in v['evidence_frames'])
        manual='<p class="annotation">Уточнение пользователя: '+html.escape(r['user_chronology'])+'</p>' if r.get('user_chronology') else ''
        cards.append('<details><summary>'+html.escape(r['id'])+' — '+str(len(c['intervals']))+' интервалов, '+f"{c['duration']:.1f}"+' с</summary>'+result+manual+'<h3>Визуальные наблюдения модели</h3><table><tr><th>Секунды</th><th>Факты интервала</th></tr>'+text+'</table></details>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Полная хронология операций</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto;padding:12px;background:#f5f7fb;color:#172339}details{background:white;padding:20px;margin:18px 0;border:1px solid #ddd;border-radius:10px}summary{cursor:pointer;font-size:22px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}img{max-width:32%;margin:.5%}.annotation{background:#fff4ce;padding:16px}</style><h1>Полный просмотр операций</h1><p>Готовые визуальные профили: '''+str(count)+''' / 42. Хроника строится каждые 3 секунды от начала до конца.</p><p class="annotation">Все 42 ролика включены в эталонную память. Для совпавшего видео итог калибруется известной разметкой; результат не является независимой оценкой точности. Наблюдения модели и уточнения пользователя показаны отдельно.</p>'''+''.join(cards)+'</html>'
    (ROOT/'visual_report.html').write_text(page,encoding='utf-8');(MEMORY/'report.html').write_text(page,encoding='utf-8')
    print('Chronology report profiles:',count)
if __name__=='__main__':main()
