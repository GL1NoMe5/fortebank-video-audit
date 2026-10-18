"""Frozen base Qwen window features. Scoring never calls generate()."""
from __future__ import annotations
import math
import time
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .config import ROOT, AnalysisConfig
from .video import SampledVideo, VideoFrame
from .vlm import LocalVLM

VERSION='window-base-logits-v1'
STEP=2.0
OFFSETS=(-8.0,-1.5,0.0,1.5)
GRID_PIXELS=786432
# Observable task categories, not automatic legal conclusions.
CATEGORIES=[
 ('camera_capable_device_in_cash_area','A mobile phone or handheld camera is present or used in the cashier work area.'),
 ('drink_on_cash_desk','A drinking cup, glass, bottle, or drink is on the cash desk.'),
 ('personal_items_on_cash_desk','A coat, personal handbag, or personal clothing is placed on the cash desk.'),
 ('cashier_leaves_during_service','The cashier leaves the desk during the shown operation, leaving cash or an unfinished transaction.'),
 ('unauthorized_person_access','A second person enters or stands inside the cashier workspace, beside or behind the cashier.'),
 ('cash_recount_on_receipt','Received banknotes are set aside or put away without a visible count during the shown sequence.'),
 ('cash_recount_on_delivery','Banknotes are handed out without a visible count during the shown sequence.'),
 ('cash_separation_by_transaction','Banknotes from different transactions are mixed or moved aside separately from the current transaction.'),
 ('collection_bag_seal_check','A cash bag or package is handled, opened, or accepted without a visible seal inspection.'),
 ('damaged_collection_bag_not_recounted','An opened or damaged cash package is handled without a visible recount of its contents.'),
 ('valuables_unattended_accessible','Cash or a cash bag remains exposed and reachable while the cashier is away.'),
 ('secure_storage_left_open','A cash drawer, cash box, or safe is visibly left open while unattended.'),
]

def centers(duration):
    if not math.isfinite(duration) or duration<=0: raise ValueError('Invalid duration')
    # Nominal windows [t-2,t+2], exact two-second stride, no zero timestamp.
    if duration<2:return np.array([duration/2],dtype=np.float32)
    return np.arange(2,duration+1e-5,STEP,dtype=np.float32)

def nearest_indices(video,t):
    stamps=np.asarray([f.timestamp_sec for f in video.frames])
    requested=np.clip(np.asarray(OFFSETS)+float(t),0,stamps[-1])
    return [int(np.abs(stamps-s).argmin()) for s in requested]

class WindowBuilder:
    def __init__(self,video:SampledVideo):
        self.video=video
        self.gray={};self.people={}
        self.hog=cv2.HOGDescriptor();self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)

    def small(self,i):
        if i not in self.gray:
            self.gray[i]=cv2.cvtColor(np.asarray(self.video.frames[i].image.resize((192,320))),cv2.COLOR_RGB2GRAY)
        return self.gray[i]

    def person_boxes(self,i):
        if i not in self.people:
            # Classical local HOG person detector; no downloaded model/adapters.
            img=np.asarray(self.video.frames[i].image.resize((256,448)))
            boxes,weights=self.hog.detectMultiScale(img,winStride=(8,8),padding=(8,8),scale=1.08)
            self.people[i]=[(x/256,y/448,(x+w)/256,(y+h)/448) for (x,y,w,h),score in zip(boxes,weights) if float(score)>.4]
        return self.people[i]

    def grid(self,t):
        indices=nearest_indices(self.video,t);frames=[self.video.frames[i] for i in indices]
        boxes=[]
        for i in set(indices): boxes.extend(self.person_boxes(i))
        for i,j in zip(indices,indices[1:]):
            a=self.small(i).astype(np.float32);b=self.small(j).astype(np.float32)
            delta=np.abs((b-np.median(b))-(a-np.median(a)))
            mask=(delta>18).astype(np.uint8)*255
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(contour)<28:continue
                x,y,w,h=cv2.boundingRect(contour);boxes.append((x/192,y/320,(x+w)/192,(y+h)/320))
        if boxes:
            x1=min(b[0] for b in boxes);y1=min(b[1] for b in boxes)
            x2=max(b[2] for b in boxes);y2=max(b[3] for b in boxes)
            # Generous table/person context and minimum retained area.
            cx=(x1+x2)/2;cy=(y1+y2)/2
            w=min(1,max(.65,(x2-x1)*1.45));h=min(1,max(.60,(y2-y1)*1.45))
            x1=max(0,min(1-w,cx-w/2));y1=max(0,min(1-h,cy-h/2));roi=(x1,y1,x1+w,y1+h)
        else:roi=(0.,0.,1.,1.)
        grid=Image.new('RGB',(768,1024),(20,20,20))
        for k,f in enumerate(frames):
            iw,ih=f.image.size;box=(int(roi[0]*iw),int(roi[1]*ih),int(roi[2]*iw),int(roi[3]*ih))
            tile=ImageOps.contain(f.image.crop(box),(384,482))
            px=(k%2)*384+(384-tile.width)//2;py=(k//2)*512+30+(482-tile.height)//2
            grid.paste(tile,(px,py));ImageDraw.Draw(grid).text(((k%2)*384+8,(k//2)*512+3),f'{f.timestamp_sec:.2f} s',font=self.font,fill='white')
        return grid,{'center_sec':float(t),'frame_seconds':[f.timestamp_sec for f in frames],
                     'roi_normalized':list(roi),'person_boxes':len(self.person_boxes(indices[2]))}

class WindowFeatureExtractor:
    def __init__(self):
        self.runner=LocalVLM(replace(AnalysisConfig.from_env(),model_path=ROOT/'models/Qwen3-VL-8B-Instruct',quantization='nf4'))
        self.ready=False

    def load(self):
        if self.ready:return
        self.runner.load();self.model=self.runner.model;self.processor=self.runner.processor
        if hasattr(self.model,'peft_config'):raise RuntimeError('QLoRA adapters are forbidden')
        self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.processor.image_processor.size={'shortest_edge':1024,'longest_edge':GRID_PIXELS}
        tok=self.processor.tokenizer
        yes=tok.encode('YES',add_special_tokens=False);no=tok.encode('NO',add_special_tokens=False)
        if len(yes)!=1 or len(no)!=1:raise ValueError('YES/NO must each be exactly one tokenizer token')
        self.yes,self.no=yes[0],no[0]
        self.ready=True

    def extract(self,grid,verify_cache=False):
        import torch
        self.load();start=time.monotonic()
        marker='QUESTION_MARKER_82716'
        common=('The image is a chronological 2x2 contact sheet of a bank cashier operation. '
                'Read top-left, top-right, bottom-left, bottom-right; timestamps are printed on the tiles. '
                'The person behind the desk is the cashier. Inspect hands, cash, bags, objects, and people. '
                'Answer the specific visible-action question with exactly YES or NO. '+marker)
        template=self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':common}]}],tokenize=False,add_generation_prompt=True)
        prefix,suffix=template.split(marker)
        inputs=self.processor(text=[prefix],images=[grid],return_tensors='pt').to('cuda')
        probs=[];margins=[];mass=[];cache_check=None
        with torch.no_grad():
            base=self.model.model(**inputs,use_cache=True,return_dict=True)
            hidden=base.last_hidden_state[0,-1].float().cpu().numpy()
            cache=base.past_key_values;prefix_length=cache.get_seq_length();del base
            rope=self.model.model.rope_deltas.clone()
            for index,(_,question) in enumerate(CATEGORIES):
                tail=self.processor.tokenizer(question+suffix,add_special_tokens=False,return_tensors='pt')['input_ids'].to('cuda')
                count=tail.shape[1]
                self.model.model.rope_deltas=rope.clone()
                # Explicit positions avoid full attention-mask length being mistaken for tail length.
                position=torch.arange(prefix_length,prefix_length+count,device='cuda').view(1,1,-1).expand(3,1,-1)+rope.to('cuda').view(1,1,1)
                output=self.model(input_ids=tail,past_key_values=cache,position_ids=position,use_cache=True,logits_to_keep=1,return_dict=True)
                logits=output.logits[0,-1].float();pair=logits[[self.yes,self.no]]
                probs.append(float(torch.softmax(pair,dim=-1)[0]));margins.append(float(pair[0]-pair[1]))
                mass.append(float(torch.exp(torch.logsumexp(pair,dim=0)-torch.logsumexp(logits,dim=0))))
                cache=output.past_key_values;cache.crop(-count)
                assert cache.get_seq_length()==prefix_length
                del output
                if verify_cache and index==0:
                    full=self.processor(text=[prefix+question+suffix],images=[grid],return_tensors='pt').to('cuda')
                    check=self.model(**full,use_cache=False,logits_to_keep=1,return_dict=True)
                    direct=torch.softmax(check.logits[0,-1,[self.yes,self.no]].float(),dim=-1)[0].item()
                    cache_check=abs(direct-probs[0])
                    if cache_check>.015:raise RuntimeError(f'Cached category scoring differs from direct forward: {cache_check}')
                    del full,check
            del inputs,cache
        feature=np.concatenate([np.asarray(probs),np.asarray(margins),hidden]).astype(np.float32)
        if not np.isfinite(feature).all():raise ValueError('Non-finite window features')
        return feature,{'seconds':round(time.monotonic()-start,3),'yes_probabilities':probs,
                        'yes_no_token_mass':mass,'cached_vs_direct_error':cache_check,'hidden_dimensions':len(hidden)}

    def describe(self,grid):
        import torch
        self.load()
        prompt='Опиши одним-двумя короткими предложениями конкретное видимое действие в трёх последних кадрах кассовой операции с учётом первого кадра. Не пиши JSON, таймкоды, ссылки или юридические выводы. Не придумывай невидимые действия.'
        text=self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=[grid],return_tensors='pt').to('cuda')
        with torch.no_grad():output=self.model.generate(**inputs,max_new_tokens=80,repetition_penalty=1.2,do_sample=False)
        result=self.processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
        return result or 'В выбранном окне классификатор обнаружил признаки возможного нарушения.'
