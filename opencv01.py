import cv2
import depthai as dai
from pathlib import Path
from datetime import datetime

WIDTH, HEIGHT, FPS = 640, 480, 30

with dai.Pipeline() as pipeline:
    # DepthAI 3.x: OAK-D Lite의 CAM_A(RGB) 사용
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = cam.requestOutput(
        size=(WIDTH, HEIGHT),
        type=dai.ImgFrame.Type.BGR888p,
        fps=FPS
    )
    queue = output.createOutputQueue()
    pipeline.start()

    try:
        while pipeline.isRunning():
            msg = queue.get()
            if msg is None:
                continue
            frame = msg.getCvFrame()  # OpenCV BGR 배열
            cv2.imshow("OAK-D Lite", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("OAK-D Lite", cv2.WND_PROP_VISIBLE) < 1:
                
                break
    finally:
        cv2.destroyAllWindows()