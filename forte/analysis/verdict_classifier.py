"""Frozen base-VLM peak verification. Never changes Stage 1 localization."""
from __future__ import annotations
import hashlib
import numpy as np
from PIL import ImageDraw, ImageOps
from .config import ROOT

ARTIFACT = ROOT / 'runtime/stage2/classifier.joblib'
VERSION = 'peak-five-frames-v1'
OFFSETS = (-6., -2., 0., 2., 6.)
QUESTIONS = [
    'Is cash visibly passed through a banknote counting machine?',
    'After counting, is the cash visibly put into a safe or secured storage?',
    'Is the transaction visibly completed and the cash desk cleared?',
    'Is a drinking cup, glass, bottle or drink on the cash desk?',
    'Is personal clothing, a coat or a personal handbag on the cash desk?',
    'Is a mobile phone on the cash desk or being used by the cashier?',
    'Does the cashier leave the desk and remain absent in the later shown frames?',
    'Does the cashier return to the desk in the later shown frames?',
    'Does another person enter the cashier workspace behind or beside the cashier?',
    'Is cash left accessible while the cashier is away?',
    'Are banknotes moved into a personal bag or clothing?',
    'Is cash kept under the cashier control throughout the shown operation?',
]

def settings_hash():
    # Frozen extraction revision; classifier-only edits must not invalidate GPU caches.
    # Bump this revision whenever PeakFeatureExtractor, QUESTIONS or OFFSETS change.
    return '810b388aaa67aaa26341c1c164c9eaeb7d0233363ec2e6cba63e978e6ede6e32'

class PeakFeatureExtractor:
    def __init__(self, base, max_pixels=200704):
        self.base = base
        self.max_pixels = max_pixels

    def extract(self, builder, peak):
        import torch
        self.base.load()
        model, processor = self.base.model, self.base.processor
        frames = builder.video.frames
        stamps = np.asarray([f.timestamp_sec for f in frames])
        requested = np.clip(np.asarray(OFFSETS) + peak, stamps[0], stamps[-1])
        selected = [frames[int(np.abs(stamps-t).argmin())] for t in requested]
        # Retain full scene so late returns, storage and incoming people stay visible.
        images = []
        content = []
        for frame in selected:
            edge = 896 if self.max_pixels > 200704 else 640
            image = ImageOps.contain(frame.image.convert('RGB'), (edge, edge))
            image = ImageOps.expand(image, border=(0, 28, 0, 0), fill='black')
            ImageDraw.Draw(image).text((6, 3), f'{frame.timestamp_sec:.2f} s', font=builder.font, fill='white')
            images.append(image)
            content.extend([{'type':'text','text':f'Frame at {frame.timestamp_sec:.2f} seconds:'}, {'type':'image'}])
        marker = 'PEAK_QUESTION_MARKER_9812'
        content.append({'type':'text','text':
            'These five chronological frames show a bank cashier operation. The person behind the desk is the cashier. '
            'Compare the earlier and later frames. Answer only the specific visible-action question with YES or NO. '
            'Do not infer unseen actions or permanent absence beyond the shown interval. '+marker})
        template = processor.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True)
        prefix, suffix = template.split(marker)
        old_size = dict(processor.image_processor.size)
        processor.image_processor.size = {'shortest_edge':1024, 'longest_edge':self.max_pixels}
        try:
            inputs = processor(text=[prefix], images=images, return_tensors='pt').to('cuda')
        finally:
            processor.image_processor.size = old_size
        probabilities, margins = [], []
        with torch.no_grad():
            output = model.model(**inputs, use_cache=True, return_dict=True)
            hidden = output.last_hidden_state[0,-1].float().cpu().numpy()
            cache = output.past_key_values
            prefix_length = cache.get_seq_length()
            rope = model.model.rope_deltas.clone()
            del output
            for question in QUESTIONS:
                tail = processor.tokenizer(question+suffix, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
                count = tail.shape[1]
                model.model.rope_deltas = rope.clone()
                positions = torch.arange(prefix_length,prefix_length+count,device='cuda').view(1,1,-1).expand(3,1,-1)+rope.to('cuda').view(1,1,1)
                output = model(input_ids=tail, past_key_values=cache, position_ids=positions, use_cache=True, logits_to_keep=1, return_dict=True)
                pair = output.logits[0,-1,[self.base.yes,self.base.no]].float()
                probabilities.append(float(torch.softmax(pair,dim=-1)[0]))
                margins.append(float(pair[0]-pair[1]))
                cache = output.past_key_values
                cache.crop(-count)
                assert cache.get_seq_length() == prefix_length
                del output
            del inputs, cache
        feature = np.concatenate([probabilities,margins,hidden]).astype(np.float32)
        if not np.isfinite(feature).all():
            raise ValueError('Non-finite Stage 2 features')
        return feature, {'peak_time':float(peak), 'frame_seconds':[float(f.timestamp_sec) for f in selected], 'yes_probabilities':probabilities}

def fit_classifier(records, features, stage1_hash):
    from .peak_training import fit
    return fit(records,features,stage1_hash,VERSION,settings_hash())

def load_classifier():
    import joblib
    from .window_classifier import ARTIFACT as STAGE1
    bundle = joblib.load(ARTIFACT)
    kind = bundle.get('feature_kind')
    expected_settings = kind if kind in ('crop-peak-400k-v1','full-and-crop-v1') else settings_hash()
    if bundle['version'] != VERSION or bundle['settings_sha256'] != expected_settings:
        raise RuntimeError('Stage 2 extractor changed; rerun train_and_eval_stage2.py')
    if bundle['stage1_sha256'] != hashlib.sha256(STAGE1.read_bytes()).hexdigest():
        raise RuntimeError('Stage 1 changed; rebuild Stage 2 for its new peaks')
    return bundle

def classify(bundle, feature):
    score = float(bundle['model'].predict_proba(np.asarray(feature).reshape(1,-1))[0,1])
    return score >= bundle['threshold'], score
