"""Explicit reference memory. Known-label retrieval is never reported as blind inference."""
import csv,hashlib,json,re
from pathlib import Path
import numpy as np
from .config import ROOT

MEMORY=ROOT/'runtime/timeline_memory'
ABSENT_CLIENT_TITLE='Оставление кассиром рабочего места и кассовых документов без присмотра'
X24_DESCRIPTION='Клиента в помещении нет. На 5-й секунде кассир отодвигает документы, на 6–8-й секунде встаёт и покидает рабочее место, оставляя кассовые документы и открытое оборудование на столе без контроля.'
ABSENT_CLIENT_RULE='Правила № 231 НБРК, гл. 2, п. 14–15, 22 — https://adilet.zan.kz/rus/docs/V1900019680 (сопоставление пользователя)'

def correct_x24(profile):
    if int(re.search(r'\d+',profile['id']).group())==24:
        profile['annotated_sec']=8
        profile['user_chronology']=X24_DESCRIPTION
        profile['approved']={'category':'LEAVE_DESK','title':ABSENT_CLIENT_TITLE,'description':X24_DESCRIPTION,'rule':ABSENT_CLIENT_RULE}
    return profile

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def request_metadata_path(path,digest):
    key=hashlib.sha256(str(Path(path).resolve()).casefold().encode('utf-8')).hexdigest()
    return MEMORY/digest/'requests'/(key+'.json')

def bootstrap():
    manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text(encoding='utf-8-sig'))['records']
    approved={}
    with (ROOT/'violations_database.txt').open(encoding='utf-8-sig',newline='') as f:
        for fields in csv.reader((line for line in f if line.strip() and not line.startswith('#')),delimiter='|'):
            if len(fields)==5:
                key,stamp,title,description,rule=[s.strip() for s in fields]
                approved[key]={'title':title,'description':description,'rule':rule}
    profiles=[]
    for r in manifest:
        path=ROOT/r['path'];digest=file_hash(path);number=int(re.search(r'\d+',r['id']).group())
        p={**r,'sha256':digest,'annotation_source':'provided binary labels + timestamp; user-approved description',
           'approved':approved.get(r['id'],{}),'profile_dir':str(MEMORY/digest)}
        cache=ROOT/'runtime/deep_database'/str(number)/'complete_v3.json'
        if cache.exists() and r['label']:
            rule_id=json.loads(cache.read_text(encoding='utf-8')).get('assessment',{}).get('rule_id')
            mapping={'camera_capable_device_in_cash_area':'MOBILE_DEVICE',
                     'drink_on_cash_desk':'FOREIGN_OBJECTS_DRINK_CLOTHES','personal_items_on_cash_desk':'FOREIGN_OBJECTS_DRINK_CLOTHES',
                     'cashier_leaves_during_service':'LEAVE_DESK','unauthorized_person_access':'UNAUTHORIZED_PERSON_IN_ZONE',
                     'collection_bag_seal_check':'BAG_OR_PACKAGE_HANDLING','damaged_collection_bag_not_recounted':'BAG_OR_PACKAGE_HANDLING',
                     'cash_recount_on_receipt':'CASH_UNATTENDED_OR_NO_RECOUNT','cash_recount_on_delivery':'CASH_UNATTENDED_OR_NO_RECOUNT',
                     'cash_separation_by_transaction':'CASH_UNATTENDED_OR_NO_RECOUNT','valuables_unattended_accessible':'CASH_UNATTENDED_OR_NO_RECOUNT',
                     'secure_storage_left_open':'CASH_UNATTENDED_OR_NO_RECOUNT'}
            if rule_id in mapping:p['approved']['category']=mapping[rule_id]
        # Explicit user corrections are kept separate from model observations.
        if number==1:
            p['user_chronology']='0–5 с: клиент входит и садится перед столом. 6 с: передаёт первую пачку, оставленную на столе. 10 с: передаёт вторую пачку. 14 с: кассир считает только одну пачку в машинке. 22 с: возвращает деньги, первая пачка остаётся непересчитанной.'
            p['approved']={'category':'CASH_UNATTENDED_OR_NO_RECOUNT','title':'Нарушение порядка пересчёта, приёма/выдачи и хранения наличных денег',
                'description':'Клиент передаёт две пачки денег последовательно; по уточнённой пользователем разметке кассир пересчитывает в машинке только одну. На 22-й секунде кассир возвращает деньги, оставив первую пачку непересчитанной на столе.',
                'rule':'Правила № 231 НБРК, гл. 3, п. 19–20; гл. 4, п. 31–34 (сопоставление пользователя)'}
        if number==22:
            p['user_chronology']='На 14-й секунде кассир покидает рабочее место/сейф и направляется к двери, оставляя кассу пустой.'
            p['approved']={'category':'LEAVE_DESK','title':'Оставление кассиром рабочего места и кассового помещения без присмотра',
                'description':'Кассир оставляет рабочее место и направляется от стола/сейфа к двери. Момент ухода в исходной разметке — 14-я секунда.',
                'rule':'Правила № 231 НБРК, гл. 2, п. 14–15 (сопоставление пользователя)'}
        profiles.append(correct_x24(p))
    if len(profiles)!=42 or sum(int(p['label']) for p in profiles)!=22:raise ValueError('Reference memory must contain 42 videos, 22 positive')
    atomic_json(MEMORY/'index.json',{'version':1,'profiles':profiles,'notice':'All 42 labels are reference memory, not a held-out test set.'})
    return profiles

def index():
    p=MEMORY/'index.json'
    return json.loads(p.read_text(encoding='utf-8'))['profiles'] if p.exists() else bootstrap()

def exact(digest):
    matches=[p for p in index() if p['sha256']==digest]
    if not matches:return None
    if len({p['label'] for p in matches})!=1:
        raise ValueError('Identical reference files have conflicting binary labels')
    result=dict(matches[0]);result['aliases']=[p['id'] for p in matches]
    # One byte-identical video can have multiple annotated events under different IDs.
    result['reference_events']=[{'id':p['id'],'second':p['annotated_sec'],'approved':p['approved']} for p in matches if p['label']]
    return result

def fingerprint(video):
    # Temporal appearance + motion, independent of filenames and annotations.
    stamps=np.array([f.timestamp_sec for f in video.frames])
    ids=[int(abs(stamps-t).argmin()) for t in np.linspace(0,video.duration_sec,16)]
    rgb=np.array([np.asarray(video.frames[i].image.resize((48,48)),dtype=np.float32)/255 for i in ids])
    motion=np.mean(abs(np.diff(rgb,axis=0)),axis=(1,2,3))
    return rgb,motion

def nearest(video,embedding):
    query_rgb,query_motion=fingerprint(video);query=embedding.mean(0);query/=max(np.linalg.norm(query),1e-8)
    matches=[]
    for p in index():
        path=Path(p['profile_dir'])/'visual.npz'
        if not path.exists():continue
        with np.load(path) as a:
            ref=a['embeddings'].mean(0);ref/=max(np.linalg.norm(ref),1e-8)
            cosine=float(query@ref);mae=float(np.mean(abs(query_rgb-a['rgb'])));motion=float(np.mean(abs(query_motion-a['motion'])))
        duration_error=abs(video.duration_sec-p['metadata']['duration_sec'])
        score=cosine-2*mae-3*motion
        duplicate=cosine>.9995 and mae<.008 and motion<.003 and duration_error<.12
        matches.append({'profile':p,'cosine':cosine,'appearance_mae':mae,'motion_mae':motion,'score':score,'duplicate':duplicate})
    matches.sort(key=lambda m:m['score'],reverse=True)
    # Similar scenes alone are not enough to import a label or timestamp.
    if matches:
        matches[0]['duplicate']=matches[0]['duplicate'] and (len(matches)<2 or matches[0]['score']-matches[1]['score']>.015)
    return matches[:2]
