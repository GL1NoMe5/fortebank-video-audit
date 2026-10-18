"""Isolated browser QA server. Accepts only its generated synthetic fixture.

Never imported by production. No model is loaded and no real video is analyzed.
"""

import base64
import hashlib
import io
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FIXTURE=ROOT/".orchestration/ui_test/synthetic.mov"


class SyntheticAnalyzer:
    def warmup(self):
        pass

    def analyze(self,path,progress=None):
        if hashlib.sha256(Path(path).read_bytes()).digest()!=hashlib.sha256(FIXTURE.read_bytes()).digest():
            raise ValueError("UI test server accepts only its synthetic fixture")
        from PIL import Image,ImageDraw
        if progress:
            progress("overview",.3)
        time.sleep(2)
        image=Image.new("RGB",(320,240),"#f5edf1")
        draw=ImageDraw.Draw(image)
        draw.rectangle((60,60,160,160),fill="#951b4c")
        buff=io.BytesIO()
        image.save(buff,format="JPEG")
        frame=base64.b64encode(buff.getvalue()).decode("ascii")
        if progress:
            progress("verification",.8)
        time.sleep(1)
        return {"verdict":"violation","violations":[{"type":"Синтетическое событие для проверки интерфейса","timecode_sec":.5,"description":"Проверка отображения карточки: искусственно созданная геометрическая фигура. Это тест интерфейса. Нормативный источник для проверки ссылки: Правила №231 НБРК, п.20. https://adilet.zan.kz/rus/docs/V1900019680","evidence_frames":[{"timestamp_sec":.5,"image_base64":frame}]}]}


def generate_fixture():
    import av
    from PIL import Image,ImageDraw
    FIXTURE.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(FIXTURE),"w") as output:
        stream=output.add_stream("libx264",rate=10)
        stream.width=320
        stream.height=240
        stream.pix_fmt="yuv420p"
        for i in range(20):
            image=Image.new("RGB",(320,240),"white")
            ImageDraw.Draw(image).rectangle((20+i*4,60,100+i*4,140),fill="#951b4c")
            frame=av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


if __name__=="__main__":
    import uvicorn
    from forte.api import create_app
    from forte.settings import Settings
    generate_fixture()
    settings=Settings(runtime_dir=ROOT/".orchestration/ui_test/runtime",api_key="synthetic-ui-test-key")
    app=create_app(settings,analyzer_factory=SyntheticAnalyzer)
    uvicorn.run(app,host="127.0.0.1",port=8001,access_log=False)
