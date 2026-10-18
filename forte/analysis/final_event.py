"""One final visible-action decision, plain text only; Python owns the JSON."""
import re
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from .video import evidence_frames

CATEGORIES={
'LEAVE_DESK':('Оставление кассиром рабочего места и кассового помещения без присмотра','гл. 2, п. 14–15'),
'CASH_UNATTENDED_OR_NO_RECOUNT':('Нарушение порядка пересчёта, приёма/выдачи и хранения наличных денег','гл. 2, п. 14–16; гл. 3, п. 19–20; гл. 4, п. 31–34'),
'BAG_OR_PACKAGE_HANDLING':('Нарушение регламента приёма и вскрытия сумки / упаковки с ценностями','гл. 3, п. 19–20; гл. 5, п. 48–52'),
'FOREIGN_OBJECTS_DRINK_CLOTHES':('Нарушение кассовой дисциплины: посторонние предметы (напитки, личные вещи, верхняя одежда) на рабочем месте','гл. 2, п. 12–13'),
'MOBILE_DEVICE':('Использование личного мобильного устройства в помещении кассового узла','гл. 2, п. 12'),
'UNAUTHORIZED_PERSON_IN_ZONE':('Нарушение режима доступа: нахождение постороннего лица в рабочей зоне кассира','гл. 2, п. 9–11'),
'NORMAL_WORK':('Обычная работа',''),
}
KEYWORDS=[('LEAVE_DESK',('выходит','уходит','покидает','пустым')),('FOREIGN_OBJECTS_DRINK_CLOTHES',('стакан','пьёт','пьет','пальто','одежд')),('MOBILE_DEVICE',('телефон','смартфон',' к лицу')),('BAG_OR_PACKAGE_HANDLING',('сумк','пакет')),('UNAUTHORIZED_PERSON_IN_ZONE',('посторонн','второй муж','рядом с кассиром'))]

def parse_answer(text):
    first,*rest=text.strip().splitlines()
    found=re.findall(r'\b(?:'+ '|'.join(CATEGORIES)+r')\b',first)
    if len(found)!=1:raise ValueError('Final VLM response must contain exactly one category code on its first line')
    code=found[0];description=' '.join(rest).strip()
    if not description:raise ValueError('Final VLM response has no description')
    # NORMAL_WORK must always remain normal; negated mentions are not actions.
    if code!='NORMAL_WORK':
        clauses=re.split(r'[.!?;]',description.lower())
        for mapped,words in KEYWORDS:
            if any(any(w in clause for w in words) and not re.search(r'\b(не|нет|без|отсутств\w*)\b',clause) for clause in clauses):
                code=mapped;break
    return code,description

def prepare(builder,peak):
    frames=builder.video.frames;stamps=np.array([f.timestamp_sec for f in frames])
    if len(frames)<3:raise ValueError('At least three distinct video frames are required')
    _,meta=builder.grid(peak);roi=meta['roi_normalized']
    def crop(frame):
        w,h=frame.image.size
        return frame.image.crop((int(roi[0]*w),int(roi[1]*h),int(roi[2]*w),int(roi[3]*h)))
    candidates=np.flatnonzero((stamps>=max(1.,peak-6))&(stamps<=peak))
    scores=[]
    for i in candidates:
        if i==0:scores.append(0.);continue
        a=np.asarray(crop(frames[i-1]).convert('L').resize((192,192)),dtype=float)
        b=np.asarray(crop(frames[i]).convert('L').resize((192,192)),dtype=float)
        scores.append(float(np.mean(np.abs((b-np.median(b))-(a-np.median(a))))))
    event=float(stamps[candidates[int(np.argmax(scores))]]) if len(candidates) and max(scores)>.01 else float(stamps[np.abs(stamps-max(1.,peak-4)).argmin()])
    requested=np.clip(np.array([event-4,event-1.5,event,event+2]),stamps[0],stamps[-1])
    indices=[int(np.abs(stamps-t).argmin()) for t in requested]
    collage=Image.new('RGB',(1024,1024),'black')
    for k,i in enumerate(indices):
        tile=ImageOps.contain(frames[i].image,(512,480));x=(k%2)*512;y=(k//2)*512
        collage.paste(tile,(x+(512-tile.width)//2,y+32+(480-tile.height)//2))
        ImageDraw.Draw(collage).text((x+6,y+3),f'{stamps[i]:.2f} s',fill='white',font=builder.font)
    ei=int(np.abs(stamps-event).argmin());detail=ImageOps.contain(crop(frames[ei]),(896,896))
    targets=[max(.5,event-2.5),event,min(builder.video.duration_sec-.5,event+2.5)]
    chosen=[]
    for target in targets:
        chosen.append(next(int(i) for i in np.argsort(abs(stamps-target)) if int(i) not in chosen))
    chosen.sort()
    return event,collage,detail,[float(stamps[i]) for i in chosen]

def analyze(extractor,builder,peak):
    import torch
    event,collage,crop,stamps=prepare(builder,peak)
    extractor.load();processor=extractor.processor
    prompt='''Первое изображение: хронологический коллаж кассовой операции 2x2 (слева направо, сверху вниз). Второе: крупный план рабочей зоны сотрудника в момент события. Помещение — касса банка, человек за рабочим столом — кассир.
Выбери ровно один код по ВИДИМЫМ действиям:
LEAVE_DESK — кассир встаёт, оставляет рабочее место/кассу, уходит через дверь; оцени сейф и оставленные ценности.
CASH_UNATTENDED_OR_NO_RECOUNT — видимая передача наличных без наблюдаемого пересчёта или наличные оставлены открыто без присмотра. Не считай отсутствие машинки в кадре доказательством нарушения.
BAG_OR_PACKAGE_HANDLING — приём, открытие, извлечение предметов из сумки/баула/пакета без наблюдаемой сверки. Само наличие упаковки не доказывает нарушение.
FOREIGN_OBJECTS_DRINK_CLOTHES — стакан/напиток, питьё, еда, пальто/куртка или личная сумка на кассовом столе.
MOBILE_DEVICE — кассир использует смартфон, держит его в руках или подносит к лицу/экрану.
UNAUTHORIZED_PERSON_IN_ZONE — второй человек заходит за спину кассиру в рабочую зону к оборудованию/сейфу либо вмешивается в обслуживание; клиент перед столом сам по себе не нарушение.
NORMAL_WORK — штатная работа за ноутбуком, с документами или кассовая операция без видимых перечисленных нарушений.
Формат: первая строка содержит ТОЛЬКО один код. Затем два предложения по-русски: конкретное действие и предмет, затем развитие действия по кадрам. Если видна обычная работа, выбери NORMAL_WORK. Не выдумывай невидимые действия. Не пиши JSON, юридические ссылки, список альтернатив или рассуждения.'''
    content=[{'type':'image'},{'type':'image'},{'type':'text','text':prompt}]
    text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    old=dict(processor.image_processor.size);processor.image_processor.size={'shortest_edge':1024,'longest_edge':802816}
    try:inputs=processor(text=[text],images=[collage,crop],return_tensors='pt').to('cuda')
    finally:processor.image_processor.size=old
    with torch.no_grad():output=extractor.model.generate(**inputs,max_new_tokens=240,repetition_penalty=1.2,do_sample=False)
    answer=processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
    code,description=parse_answer(answer)
    diagnostic={'event_time':event,'category':code,'evidence_times':stamps}
    if code=='NORMAL_WORK':return {'verdict':'no_violation','violations':[]},diagnostic
    title,rule=CATEGORIES[code]
    description+=' Сопоставление по заданному справочнику: Правила № 231 НБРК, '+rule+'.'
    return {'verdict':'violation','violations':[{'type':title,'timecode_sec':event,'description':description,'evidence_frames':evidence_frames(builder.video,stamps,3)}]},diagnostic
