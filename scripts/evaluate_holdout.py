"""Audit saved predictions locally; output only numerical/label summaries."""
import html
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'runtime/qlora'

def main():
    rows=json.loads((FOLDER/'holdout_predictions.json').read_text(encoding='utf-8'))
    approval=json.loads((FOLDER/'review_approval.json').read_text(encoding='utf-8'))
    assert len(rows)==8 and {r['id'] for r in rows}==set(approval['test_ids'])
    assert not set(approval['test_ids']) & set(approval['train_ids'])
    results=[];cards=[]
    for row in rows:
        raw=row['prediction'];expected=row['reference'];parsed=None
        try:
            start=raw.index('{');parsed=json.JSONDecoder().raw_decode(raw[start:])[0]
            if not isinstance(parsed,dict): parsed=None
        except (ValueError,TypeError): pass
        match=re.search(r'"verdict"\s*:\s*"(no_violation|violation)"',raw)
        verdict=parsed.get('verdict') if parsed else match.group(1) if match else None
        events=parsed.get('violations',[]) if parsed else []
        valid=bool(parsed is not None and verdict in ('violation','no_violation') and isinstance(events,list))
        reference_times=[e['timecode_sec'] for e in expected.get('violations',[])]
        predicted_times=[e.get('timecode_sec') for e in events if isinstance(e,dict)] if isinstance(events,list) else []
        nearest=[min((abs(p-t) for p in predicted_times if isinstance(p,(float,int))),default=None) for t in reference_times]
        summary={'id':row['id'],'reference':expected['verdict'],'prediction':verdict,'valid_json':valid,
                 'binary_correct':verdict==expected['verdict'],'usable_binary_correct':valid and verdict==expected['verdict'],
                 'reference_times':reference_times,'predicted_times':predicted_times,'timestamp_errors':nearest}
        results.append(summary)
        status='Верно по наличию нарушения' if summary['binary_correct'] else 'Ошибка наличия нарушения'
        if not valid: status+='; JSON незавершён — ответ непригоден для API'
        if nearest: status+='; ошибка таймкода: '+str(nearest)
        cards.append('<article><h2>'+html.escape(row['id'])+'</h2><p>'+html.escape(status)+'</p><p>Эталон: '+html.escape(expected['verdict'])+'; ответ: '+html.escape(str(verdict))+'</p><p>Секунды эталона: '+str(reference_times)+'; секунды модели: '+str(predicted_times)+'</p><div class="cols"><section><h3>Утверждённая разметка</h3><pre>'+html.escape(json.dumps(expected,ensure_ascii=False,indent=2))+'</pre></section><section><h3>Фактический ответ модели</h3><pre>'+html.escape(raw)+'</pre></section></div></article>')
    totals={'videos':8,'valid_json':sum(r['valid_json'] for r in results),
            'binary_correct_including_incomplete':sum(r['binary_correct'] for r in results),
            'usable_binary_correct':sum(r['usable_binary_correct'] for r in results),
            'positive_timestamp_within_2s':sum(any(e is not None and e<=2 for e in r['timestamp_errors']) for r in results)}
    (FOLDER/'holdout_metrics.json').write_text(json.dumps({'totals':totals,'videos':results},ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка восьми тестовых роликов</title><style>body{font:18px system-ui;background:#f2f3f5;margin:28px;color:#17232d}article{background:white;padding:24px;margin:20px 0;border-radius:12px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px system-ui}</style><h1>Qwen3-VL-8B + QLoRA: 8 тестовых видео</h1><p>Проверены сохранённые ответы финального адаптера, не новый прогон. Все 8 роликов исключены из обучения. Разметка подготовлена автоматически и затем утверждена пользователем.</p><p>Бинарный вердикт верен в 4/8 ответов, включая один незавершённый. Полный JSON с верным бинарным вердиктом: 3/8. На положительных роликах нет полного ответа с таймкодом в пределах ±2 секунд.</p>'+''.join(cards)+'</html>'
    (FOLDER/'holdout_report.html').write_text(page,encoding='utf-8')
    print(json.dumps(totals))

if __name__=='__main__': main()
