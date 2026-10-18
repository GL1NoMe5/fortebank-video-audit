"""Keep the jury HTTPS tunnel and GitHub Pages endpoint alive locally."""
import ctypes,json,re,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runtime/public_gateway'
RUN.mkdir(parents=True,exist_ok=True)
URL=re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')
def publish(url):
    config=ROOT/'site/config.js'
    config.parent.mkdir(exist_ok=True)
    value='window.FORTE_API_BASE = '+json.dumps(url)+';\n'
    if config.exists() and config.read_text()==value:return
    config.write_text(value,encoding='utf-8')
    (RUN/'endpoint.json').write_text(json.dumps({'url':url}),encoding='utf-8')
    if (RUN/'publish_ready').exists():
        subprocess.run(['git','add','site/config.js'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Update public API tunnel endpoint'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    while True:
        with (RUN/'tunnel.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([str(ROOT/'tools/cloudflared/cloudflared.exe'),'tunnel','--url','http://127.0.0.1:8002','--protocol','http2','--no-autoupdate'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            (RUN/'tunnel.pid').write_text(str(process.pid))
            published=False
            while process.poll() is None:
                if not published:
                    content=(RUN/'tunnel.log').read_text(encoding='utf-8',errors='replace')
                    match=URL.search(content)
                    if match:
                        try:publish(match.group());published=True
                        except Exception as e:print(type(e).__name__,str(e),flush=True)
                time.sleep(3)
        time.sleep(3)
if __name__=='__main__':main()
