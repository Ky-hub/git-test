import asyncio
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from bithuman.runtime_async import AsyncBithuman
import threading

# pip install fastapi uvicorn
# uvicorn main:app --host 0.0.0.0 --port 5000


API_SECRET = "boj65FL6Ga7K6Gvw01dm1cV2kW9LVpymSxEjW5AiW9WVI2EmvQtW9uUNu6Q3CVku7"
MODEL_PATH = "./art_teacher.imx"

app = FastAPI()
frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)  # 只保留最新一帧

# -----------------------------
# 异步运行 BitHuman 并填充队列
# -----------------------------
def start_bithuman_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_bithuman():
        runtime = await AsyncBithuman.create(model_path=MODEL_PATH, api_secret=API_SECRET)
        async for frame in runtime.run():
            if frame.has_image:
                if frame_queue.full():
                    await frame_queue.get()  # 丢掉旧帧
                await frame_queue.put(frame.bgr_image)

    loop.run_until_complete(run_bithuman())

# 后台线程启动 BitHuman
threading.Thread(target=start_bithuman_loop, daemon=True).start()

# -----------------------------
# MJPEG 视频流生成器
# -----------------------------
async def mjpeg_generator():
    while True:
        frame = await frame_queue.get()
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

# -----------------------------
# FastAPI 路由
# -----------------------------
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    html = """
    <html>
        <head><title>BitHuman Stream</title></head>
        <body>
            <h1>BitHuman MJPEG Stream</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    """
    return HTMLResponse(content=html)

# -----------------------------
# 启动 FastAPI: uvicorn main:app --host 0.0.0.0 --port 5000
# -----------------------------
