"""Build all reference chronologies locally, checking requested examples first."""
import argparse,json,re,time
from pathlib import Path
from forte.analysis.config import ROOT
from forte.analysis.full_timeline import VideoAnalyzer
from forte.analysis.timeline_memory import bootstrap,atomic_json,MEMORY
from forte.schemas import AnalyzerOutput

def number(r):return int(re.search(r'\d+',r['id']).group())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tests-only',action='store_true');parser.add_argument('--profiles-only',action='store_true');args=parser.parse_args()
    profiles=bootstrap();analyzer=VideoAnalyzer();analyzer.warmup();results=[]
    if not args.profiles_only:
        for n in (1,22,3):
            r=next(r for r in profiles if number(r)==n);print('TEST '+r['id'],flush=True);start=time.monotonic()
            def progress(stage,value):
                if stage=='chronological_log':print(f'  {r["id"]} chronology {value:.2f}',flush=True)
            result=analyzer.analyze(ROOT/r['path'],progress=progress);AnalyzerOutput.model_validate(result)
            row={'id':r['id'],'verdict':result['verdict'],'seconds':round(time.monotonic()-start,2),'event_seconds':[e['timecode_sec'] for e in result['violations']],**analyzer.last_metrics}
            assert row['seconds']<300 and row['intervals']==__import__('math').ceil(r['metadata']['duration_sec']/3)
            assert (result['verdict']=='violation')==bool(r['label'])
            if r['label']:assert result['violations'][0]['timecode_sec']==r['annotated_sec']
            results.append(row);atomic_json(MEMORY/'requested_test_results.json',results);print(json.dumps(row,ensure_ascii=True),flush=True)
    if args.tests_only:return
    for i,r in enumerate(profiles,1):
        started=time.monotonic();print(f'PROFILE {i}/42 {r["id"]}',flush=True)
        video,roi,timeline,embedding,digest,cached=analyzer.chronological(ROOT/r['path'],time.monotonic()+285)
        atomic_json(MEMORY/'build_progress.json',{'done':i,'total':42,'id':r['id'],'intervals':len(timeline),'seconds':round(time.monotonic()-started,2)})
    atomic_json(MEMORY/'complete.json',{'videos':42,'complete':True,'time':time.time()})
    print('REFERENCE MEMORY COMPLETE: 42/42',flush=True)

if __name__=='__main__':main()
