import asyncio
import cv2
import time
from bithuman.runtime_async import AsyncBithuman

API_SECRET = "boj65FL6Ga7K6Gvw01dm1cV2kW9LVpymSxEjW5AiW9WVI2EmvQtW9uUNu6Q3CVku7"
MODEL_PATH = "./art_teacher.imx"

async def main():
    runtime = await AsyncBithuman.create(
        model_path=MODEL_PATH,
        api_secret=API_SECRET
    )

    print("BitHuman runtime started. Press 'q' to quit.")

    prev_time = time.time()
    frame_count = 0
    fps = 0

    async for frame in runtime.run():
        if frame.has_image:
            frame_count += 1

            # 每秒计算一次 FPS
            current_time = time.time()
            elapsed = current_time - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = current_time

            # 在图像上显示 FPS
            display_frame = frame.bgr_image.copy()
            cv2.putText(
                display_frame,
                f"FPS: {fps:.2f}",
                (10, 30),                # 左上角位置
                cv2.FONT_HERSHEY_SIMPLEX,
                1,                        # 字体大小
                (0, 255, 0),              # 绿色
                2,                        # 字体粗细
                cv2.LINE_AA
            )

            cv2.imshow("BitHuman Frame", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
