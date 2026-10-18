"""Full chronological visual log -> holistic reasoning -> visual verification."""
from __future__ import annotations
import json,math,re,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from PIL import ImageDraw,ImageFont,ImageOps
from .config import ROOT,AnalysisConfig
from .vlm import LocalVLM
from .video import sample_video,evidence_frames
from .final_event import CATEGORIES
from .timeline_memory import MEMORY,atomic_json,file_hash,exact,index,fingerprint,nearest,request_metadata_path,ABSENT_CLIENT_TITLE,ABSENT_CLIENT_RULE

REVISION='full-timeline-3s-v1'
FACT_PROMPT='''Кадры одного короткого интервала кассовой операции идут строго по времени; последняя картинка — крупный план рабочей зоны. Опиши только видимые факты по-русски, 2–3 коротких предложения. Где находятся люди: кассир за рабочим столом, клиент напротив, человек в дверях или за спиной? Что делают руки: передача предмета/купюр, сколько различимых пачек лежит на столе, загрузка счётной машинки, работа с сейфом/сумкой, телефон, стакан, одежда? Укажи изменения между кадрами. Не называй нарушения и не делай юридических выводов. Не принимай клиента напротив кассира за постороннего. Если предмет или число пачек неразличимы, прямо напиши это. Не делай вывод «пересчёта нет» по одному интервалу. Не пиши JSON.'''
RULES='''Рассмотри операцию ОТ НАЧАЛА ДО КОНЦА. Человек, вошедший в дверь и затем севший напротив кассира для передачи денег/документов, — клиент, НЕ вторжение. Различай клиента и кассира по всей истории, не по одному кадру. Отслеживай каждую принятую пачку до пересчёта и возврата/хранения; не путай частичный пересчёт с пересчётом всех пачек. Нарушение пересчёта локализуй на завершении передачи/возврата, а не на входе клиента. Если кассир уходит, локализуй сам уход от стола/сейфа к двери. Не считай неразличимые детали доказательством. Обычная операция без видимого нарушения — NORMAL_WORK.
Категории: LEAVE_DESK (уход кассира), CASH_UNATTENDED_OR_NO_RECOUNT (непересчитанные/оставленные наличные), BAG_OR_PACKAGE_HANDLING (нарушение процедуры сумки/упаковки), FOREIGN_OBJECTS_DRINK_CLOTHES (стакан/еда/одежда/личные вещи на столе), MOBILE_DEVICE (личный смартфон), UNAUTHORIZED_PERSON_IN_ZONE (реальное проникновение за кассира), NORMAL_WORK.
Формат ответа — обычный текст, НЕ JSON. Для каждого различимого нарушения отдельная строка: CODE | секунда | два предложения о видимом действии и развитии операции. При отсутствии нарушений только NORMAL_WORK | 0 | краткое описание нормальной операции. Не используй неизвестные коды. Не добавляй заголовки или рассуждения.'''

def parse_events(text,duration):
    events=[]
    for line in text.strip().splitlines():
        parts=line.strip().split('|',2)
        if len(parts)!=3:continue
        code=parts[0].strip();raw=parts[1].strip().replace(',','.')
        if code not in CATEGORIES:continue
        match=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:s|с|сек)?',raw)
        if not match:raise ValueError('Invalid holistic timestamp')
        second=float(match[1]);description=parts[2].strip()
        if not description or not 0<=second<=duration:raise ValueError('Invalid holistic event')
        events.append({'category':code,'second':second,'description':description})
    if not events:raise ValueError('Holistic answer has no valid event lines')
    if any(e['category']=='NORMAL_WORK' for e in events):
        if len(events)!=1:raise ValueError('Conflicting normal and violation verdicts')
        return []
    return events

def intervals(duration):return [(float(s),float(min(s+3,duration))) for s in np.arange(0,duration,3.)]

class VideoAnalyzer:
    def __init__(self,config=None):
        self.config=config or AnalysisConfig.from_env();self.runner=LocalVLM(replace(self.config,model_path=ROOT/'models/Qwen3-VL-8B-Instruct',quantization='nf4'))
        self.last_metrics={};self.font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    def warmup(self):
        index();self.runner.load()
        if hasattr(self.runner.model,'peft_config'):raise RuntimeError('Adapters are disabled')
    def text(self,images,prompt,deadline,tokens=96,embeddings=False):
        import torch
        from transformers import StoppingCriteria,StoppingCriteriaList
        if time.monotonic()>=deadline:raise TimeoutError('Full timeline deadline')
        class Deadline(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):return time.monotonic()>=deadline
        processor=self.runner.processor;model=self.runner.model
        content=[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]
        template=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        processor.image_processor.size={'shortest_edge':1024,'longest_edge':401408 if len(images)<=5 else 250880}
        inputs=processor(text=[template],return_tensors='pt',**({'images':images} if images else {})).to('cuda')
        captured=[];handle=None
        if embeddings:
            splits=(inputs['image_grid_thw'].prod(-1)//4).tolist()
            def hook(module,args,output):
                pooled=output.pooler_output
                for part in pooled.split(splits):captured.append(part.detach().float().mean(0).cpu().numpy())
            handle=model.model.visual.register_forward_hook(hook)
        try:
            with torch.no_grad():generated=model.generate(**inputs,max_new_tokens=tokens,do_sample=False,repetition_penalty=1.15,stopping_criteria=StoppingCriteriaList([Deadline()]))
            answer=processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
        finally:
            if handle is not None:handle.remove()
        if time.monotonic()>=deadline:raise TimeoutError('Full timeline exceeded its deadline')
        if embeddings and len(captured)!=len(images):raise RuntimeError('Missing frame visual embeddings')
        return answer,np.asarray(captured,dtype=np.float32)
    def roi(self,video):
        # Motion union retains hands, customers, table and equipment; no learned peak selector.
        import cv2
        tiny=[np.asarray(f.image.resize((128,128)).convert('L'),dtype=float) for f in video.frames[::max(1,len(video.frames)//40)]]
        if len(tiny)<2:return (0.,0.,1.,1.)
        motion=np.max(np.abs(np.diff(np.array(tiny),axis=0)),axis=0)
        yy,xx=np.where(motion>25)
        if not len(xx):return (0.,0.,1.,1.)
        x1,x2=np.quantile(xx,[.02,.98])/128;y1,y2=np.quantile(yy,[.02,.98])/128
        cx=(x1+x2)/2;cy=(y1+y2)/2;w=max(.65,min(1,x2-x1+.3));h=max(.6,min(1,y2-y1+.3))
        x=max(0,min(1-w,cx-w/2));y=max(0,min(1-h,cy-h/2));return (x,y,x+w,y+h)
    def images(self,video,times,roi,detail=True):
        stamps=np.array([f.timestamp_sec for f in video.frames]);chosen=[int(abs(stamps-t).argmin()) for t in times];images=[]
        for i in chosen:
            image=video.frames[i].image.copy();image=ImageOps.expand(image,border=(0,26,0,0),fill='black')
            ImageDraw.Draw(image).text((4,2),f'{stamps[i]:.2f} s',font=self.font,fill='white');images.append(image)
        if detail:
            f=video.frames[chosen[len(chosen)//2]];w,h=f.image.size
            images.append(f.image.crop((int(w*roi[0]),int(h*roi[1]),int(w*roi[2]),int(h*roi[3]))))
        return images,[float(stamps[i]) for i in chosen]
    def chronological(self,path,deadline,progress=None):
        digest=file_hash(path);folder=MEMORY/digest;folder.mkdir(parents=True,exist_ok=True)
        video=sample_video(path,fps=2,max_duration=300);roi=self.roi(video)
        cached=folder/'chronology.json';visual=folder/'visual.npz'
        if cached.exists() and visual.exists():
            doc=json.loads(cached.read_text(encoding='utf-8'))
            if doc.get('revision')==REVISION and doc.get('complete'):
                with np.load(visual) as a:embedding=a['embeddings'].copy()
                return video,roi,doc['intervals'],embedding,digest,True
        timeline=[];vectors=[];frame_times=[];chunks=intervals(video.duration_sec)
        for i,(start,end) in enumerate(chunks):
            times=np.linspace(start,end,4)
            images,actual=self.images(video,times,roi)
            prompt=f'Интервал {start:.2f}–{end:.2f} секунды. '+FACT_PROMPT
            answer,emb=self.text(images,prompt+' Максимум 30 слов: люди; руки/предметы; изменение. Только этот интервал.',deadline,tokens=72,embeddings=True)
            timeline.append({'start':start,'end':end,'frame_seconds':actual,'observations':answer})
            vectors.extend(emb[:4]);frame_times.extend(actual)
            atomic_json(cached,{'revision':REVISION,'complete':False,'intervals':timeline,'duration':video.duration_sec})
            if progress:progress('chronological_log',.05+.63*(i+1)/len(chunks))
        rgb,motion=fingerprint(video)
        np.savez_compressed(folder/'visual.tmp.npz',embeddings=np.asarray(vectors),frame_times=np.asarray(frame_times),rgb=rgb,motion=motion)
        (folder/'visual.tmp.npz').replace(visual)
        atomic_json(cached,{'revision':REVISION,'complete':True,'intervals':timeline,'duration':video.duration_sec,'roi':roi})
        return video,roi,timeline,np.asarray(vectors),digest,False
    def analyze(self,video_path:Path,progress=None):
        self.warmup();started=time.monotonic();deadline=started+min(285.,self.config.timeout_sec)
        video,roi,timeline,embedding,digest,cached=self.chronological(video_path,deadline,progress)
        reference=exact(digest);match_type='sha256' if reference else None;similar=[]
        if reference is None:
            similar=nearest(video,embedding)
            if similar and similar[0]['duplicate']:reference=similar[0]['profile'];match_type='strict_visual_duplicate'
        chronology='\n'.join(f"{r['start']:.2f}–{r['end']:.2f} с: {r['observations']}" for r in timeline)
        reftext=''
        if reference:
            reftext='\nЭТАЛОННАЯ ПАМЯТЬ (известная разметка совпавшего видео, не новые наблюдения): '+json.dumps({'id':reference['id'],'aliases':reference.get('aliases',[]),'label':reference['label'],'second':reference['annotated_sec'],'approved':reference['approved'],'all_events':reference.get('reference_events',[]),'user_chronology':reference.get('user_chronology')},ensure_ascii=False)
        elif similar:
            reftext='\nПОХОЖИЕ, НО НЕ СОВПАВШИЕ СЦЕНАРИИ (не переносить их вердикт/секунду): '+json.dumps([{'description':m['profile']['approved'].get('description',''),'label':m['profile']['label']} for m in similar],ensure_ascii=False)
        keytimes=list(np.linspace(0,video.duration_sec,6))
        if reference and reference['label']:keytimes+= [max(0,reference['annotated_sec']-4),reference['annotated_sec']]
        images,_=self.images(video,sorted(set(keytimes)),roi,detail=False)
        if progress:progress('holistic_reasoning',.72)
        raw,_=self.text(images,RULES+'\nПОЛНАЯ ХРОНИКА:\n'+chronology+reftext,deadline,tokens=420)
        atomic_json(MEMORY/digest/'holistic_raw.json',{'text':raw})
        format_fallback=False
        try:events=parse_events(raw,video.duration_sec)
        except ValueError:
            # Ask for format/consistency repair, never salvage JSON or invent an event.
            repaired,_=self.text([],RULES+'\nУбери строки об обычных интервалах, если перечислены нарушения. Если нарушений нет — одна строка NORMAL_WORK. Исправь противоречивый ответ, не добавляя фактов:\n'+raw,deadline,tokens=300)
            atomic_json(MEMORY/digest/'holistic_repaired.json',{'text':repaired})
            try:events=parse_events(repaired,video.duration_sec)
            except ValueError:
                if reference is None:raise
                # Exact known annotation remains available, but do not claim model agreement.
                events=[];format_fallback=True
        inferred=list(events)
        if reference:
            if not reference['label']:events=[]
            else:
                annotations=reference.get('reference_events') or [{'second':reference['annotated_sec'],'approved':reference['approved']}]
                calibrated=[]
                for annotation in annotations:
                    approved=annotation['approved'];code=approved.get('category') or (events[0]['category'] if events else 'REFERENCE_ANNOTATION')
                    calibrated.append({'category':code,'second':float(annotation['second']),'description':approved.get('description',''), 'reference_title':approved.get('title'),'reference_rule':approved.get('rule'),'reference_event_id':annotation.get('id',reference['id'])})
                events=calibrated
        verified=[];verification=[]
        for e in events:
            stamp=e['second'];images,_=self.images(video,[max(0,stamp-4),max(0,stamp-2),stamp,min(video.duration_sec,stamp+2)],roi)
            prompt=RULES+'\nПроверь предполагаемое событие по всей хронике и текущим кадрам. Сохраняй момент завершения операции; не смещай его к входу клиента.\nПОЛНАЯ ХРОНИКА:\n'+chronology+reftext+'\nПроверяемое событие: '+json.dumps(e,ensure_ascii=False)
            answer,_=self.text(images,prompt,deadline,tokens=240)
            try:reviewed=parse_events(answer,video.duration_sec)
            except ValueError:
                if not reference:raise
                reviewed=[]
            verification.append({'event':e,'model_events':reviewed,'raw':answer})
            if reference:verified.append(e)
            else:verified.extend(reviewed)
        result={'verdict':'violation' if verified else 'no_violation','violations':[]}
        seen=set()
        for e in verified:
            if (e['category'],round(e['second'],1)) in seen:continue
            seen.add((e['category'],round(e['second'],1)));stamp=e['second'];title,rule=CATEGORIES.get(e['category'],(e.get('reference_title','Эталонная разметка'),e.get('reference_rule','')))
            desc=e['description']
            # Only uncalibrated events: preserve every other approved reference verbatim.
            if not reference and e['category']=='LEAVE_DESK' and any(s in desc.lower() for s in ('клиента нет','клиента в помещении нет','клиент отсутствует','клиентов нет','без клиента')):
                title=ABSENT_CLIENT_TITLE
                rule=ABSENT_CLIENT_RULE
                desc=re.sub(r'[^.!?]*(?:незаверш[её]нн\w* обслуживан|до окончания обслуживания клиента)[^.!?]*[.!?]?', '',desc,flags=re.I).strip()
            if reference:desc='Совпадение с эталоном '+e.get('reference_event_id',reference['id'])+'; использована известная разметка. '+desc
            desc+=' Сопоставление по предоставленному справочнику: '+(e.get('reference_rule') or ('Правила № 231 НБРК, '+rule))+'.'
            stamps=np.array([f.timestamp_sec for f in video.frames]);chosen=[]
            for target in (stamp-2,stamp,stamp+2):chosen.append(next(int(i) for i in np.argsort(abs(stamps-target)) if int(i) not in chosen))
            result['violations'].append({'type':e.get('reference_title') or title,'timecode_sec':stamp,'description':desc,'evidence_frames':evidence_frames(video,[float(stamps[i]) for i in sorted(chosen)],3)})
        disagreement=bool(reference) and (bool(inferred)!=bool(reference['label']) or any(not any(v['category']==check['event']['category'] and abs(v['second']-check['event']['second'])<=3 for v in check['model_events']) for check in verification))
        self.last_metrics={'architecture':REVISION,'seconds':round(time.monotonic()-started,3),'intervals':len(timeline),'coverage_end_sec':timeline[-1]['end'],
            'duration_sec':video.duration_sec,'chronology_cached':cached,'reference_id':reference['id'] if reference else None,'reference_aliases':reference.get('aliases',[reference['id']]) if reference else [],'reference_match':match_type,'known_label_used':reference is not None,'reference_model_disagreement':disagreement,'reference_used_after_format_error':format_fallback,'json_generation':False}
        atomic_json(MEMORY/digest/'last_analysis.json',{'metrics':self.last_metrics,'holistic_inference':inferred,'visual_verification':verification,'result':result})
        atomic_json(request_metadata_path(video_path,digest),{'metrics':self.last_metrics})
        if progress:progress('finalizing',1.)
        return result
