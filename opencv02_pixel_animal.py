import urllib.request
from pathlib import Path

import cv2
import numpy as np
import depthai as dai

WIDTH, HEIGHT, FPS = 640, 480, 30
WINDOW = "Pixel Animal Filter"

# ---------------------------------------------------------------------------
# 픽셀 아트 스프라이트 (문자 1개 = 픽셀 1개)
#   .  투명      K  외곽선    O  몸 색     S  줄무늬/진한 몸 색
#   P  귀 안쪽   E  눈        B  볼터치    M  주둥이    W  흰자   L  배
# ---------------------------------------------------------------------------
SPRITES = {
    "cat": [
        "..K..........K..",
        ".KPK........KPK.",
        ".KPPK......KPPK.",
        "KOOOOKKKKKKOOOOK",
        "KOOOOOSOOSOOOOOK",
        "KOOOOOOOOOOOOOOK",
        "KOOEEOOOOOOEEOOK",
        "KOOEEOOOOOOEEOOK",
        "KOBBOOOPPOOOBBOK",
        "KOOOOOOKKOOOOOOK",
        "KOOOOOKOOKOOOOOK",
        ".KOOOOOOOOOOOOK.",
        "..KKOOOOOOOOKK..",
        "....KKKKKKKK....",
    ],
    "rabbit": [
        "...KK......KK...",
        "..KOOK....KOOK..",
        "..KOPK....KPOK..",
        "..KOPK....KPOK..",
        "..KOPK....KPOK..",
        "..KOOK....KOOK..",
        "..KOOKKKKKKOOK..",
        ".KOOOOOOOOOOOOK.",
        "KOOOOOOOOOOOOOOK",
        "KOOEEOOOOOOEEOOK",
        "KOOEEOOOOOOEEOOK",
        "KOBBOOOPPOOOBBOK",
        "KOOOOOOKKOOOOOOK",
        "KOOOOOKOOKOOOOOK",
        ".KOOOOOOOOOOOOK.",
        "..KKOOOOOOOOKK..",
        "....KKKKKKKK....",
    ],
    "bear": [
        ".KKKK......KKKK.",
        "KOOOOK....KOOOOK",
        "KOPPOKKKKKKOPPOK",
        "KOOOOOOOOOOOOOOK",
        "KOOOOOOOOOOOOOOK",
        "KOOEEOOOOOOEEOOK",
        "KOOEEOOOOOOEEOOK",
        "KOOOOOMMMMOOOOOK",
        "KOBBOMMKKMMOBBOK",
        "KOOOOMMMMMMOOOOK",
        "KOOOOOMMMMOOOOOK",
        ".KOOOOOOOOOOOOK.",
        "..KKOOOOOOOOKK..",
        "....KKKKKKKK....",
    ],
    "frog": [
        "..KKKK....KKKK..",
        ".KOOOOK..KOOOOK.",
        ".KOWEOKKKKOWEOK.",
        "KOOOOOOOOOOOOOOK",
        "KOOOOOOOOOOOOOOK",
        "KOBBOOOOOOOOBBOK",
        "KOOOKOOOOOOKOOOK",
        "KOOOOKKKKKKOOOOK",
        "KOOOOOOOOOOOOOOK",
        ".KLLLLLLLLLLLLK.",
        "..KKLLLLLLLLKK..",
        "....KKKKKKKK....",
    ],
}

# 색상 팔레트 (BGR)
COMMON = {"K": (60, 60, 70), "E": (40, 40, 50), "W": (255, 255, 255)}
PALETTES = {
    "cat":    {"O": (80, 150, 240), "S": (40, 110, 210), "P": (170, 160, 245), "B": (120, 120, 245)},
    "rabbit": {"O": (245, 245, 250), "P": (190, 180, 250), "B": (170, 160, 250)},
    "bear":   {"O": (70, 120, 180), "P": (140, 180, 220), "M": (150, 200, 235), "B": (110, 110, 220)},
    "frog":   {"O": (90, 220, 160), "L": (160, 240, 210), "B": (130, 140, 240)},
}
ANIMALS = list(SPRITES.keys())


def build_sprite(name, blink=False):
    """문자 그리드 -> BGRA 이미지 (작은 원본 크기)"""
    rows = SPRITES[name]
    w = max(len(r) for r in rows)
    colors = {**COMMON, **PALETTES[name]}
    img = np.zeros((len(rows), w, 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row.ljust(w, ".")):
            if ch == ".":
                continue
            if blink and ch in "EW":
                ch = "K" if ch == "E" else "O"  # 눈 감기: 눈동자 -> 선, 흰자 -> 몸 색
            img[y, x, :3] = colors.get(ch, colors["O"])
            img[y, x, 3] = 255
    return img


SPRITE_CACHE = {(n, b): build_sprite(n, b) for n in ANIMALS for b in (False, True)}


def overlay(frame, sprite, x, y, tw, angle=0.0, pivot=None):
    """스프라이트를 폭 tw로 확대(최근접 보간 -> 픽셀 느낌 유지)해서 (x, y)에 합성.
    angle(도)만큼 pivot을 중심으로 회전 (고개 기울기 반영)"""
    sh, sw = sprite.shape[:2]
    th = int(tw * sh / sw)
    if tw <= 0 or th <= 0:
        return
    big = cv2.resize(sprite, (tw, th), interpolation=cv2.INTER_NEAREST)

    H, W = frame.shape[:2]
    if pivot is None:
        pivot = (x + tw / 2, y + th / 2)
    T = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)
    R = np.vstack([cv2.getRotationMatrix2D(pivot, -angle, 1.0), [0, 0, 1]])
    layer = cv2.warpAffine(big, (R @ T)[:2], (W, H), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0))

    ys, xs = np.nonzero(layer[:, :, 3])
    if len(xs) == 0:
        return
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = layer[y0:y1, x0:x1]
    alpha = crop[:, :, 3:4] / 255.0
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = (alpha * crop[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)


def pixelate(frame, block=8):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // block, h // block), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


MODEL_PATH = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def create_detector():
    """YuNet DNN 얼굴 검출기 (Haar보다 옆얼굴/안경/어두운 조명에 강함)"""
    if not MODEL_PATH.exists():
        print("YuNet 모델 다운로드 중...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (WIDTH, HEIGHT),
                                     score_threshold=0.6, nms_threshold=0.3, top_k=50)


def detect_faces(detector, frame):
    """[(x, y, w, h, 기울기각도), ...] 반환"""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    out = []
    for f in faces:
        x, y, fw, fh = f[:4]
        rx, ry, lx, ly = f[4:8]  # 오른쪽 눈, 왼쪽 눈 (화면상 왼쪽, 오른쪽)
        angle = np.degrees(np.arctan2(ly - ry, lx - rx))
        out.append((x, y, fw, fh, angle))
    return out


class FaceTracker:
    """박스 떨림을 줄이는 지수 이동 평균 + 몇 프레임 놓쳐도 유지"""

    def __init__(self, alpha=0.4, hold=8):
        self.alpha = alpha
        self.hold = hold
        self.tracks = []  # [{"box": array(x, y, w, h, angle), "miss": int}]

    def update(self, faces):
        unmatched = list(range(len(self.tracks)))
        for f in faces:
            f = np.array(f, np.float32)
            c = f[:2] + f[2:4] / 2
            best, best_d = None, f[2]  # 얼굴 폭 이내만 같은 얼굴로 간주
            for i in unmatched:
                p = self.tracks[i]["box"]
                d = np.linalg.norm(c - (p[:2] + p[2:4] / 2))
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                self.tracks.append({"box": f, "miss": 0})
            else:
                t = self.tracks[best]
                t["box"] = self.alpha * f + (1 - self.alpha) * t["box"]
                t["miss"] = 0
                unmatched.remove(best)
        for i in unmatched:
            self.tracks[i]["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= self.hold]
        return [t["box"] for t in self.tracks]


def main():
    detector = create_detector()
    tracker = FaceTracker()
    animal_idx = 0
    pixel_bg = True
    frame_no = 0

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        output = cam.requestOutput(size=(WIDTH, HEIGHT), type=dai.ImgFrame.Type.BGR888p, fps=FPS)
        queue = output.createOutputQueue()
        pipeline.start()

        try:
            while pipeline.isRunning():
                msg = queue.get()
                if msg is None:
                    continue
                frame = msg.getCvFrame()
                frame_no += 1

                faces = tracker.update(detect_faces(detector, frame))

                out = pixelate(frame) if pixel_bg else frame.copy()
                blink = (frame_no % 90) < 5  # 약 3초마다 깜빡

                for i, (x, y, w, h, angle) in enumerate(faces):
                    # 얼굴마다 다른 동물 (첫 얼굴은 선택한 동물)
                    name = ANIMALS[(animal_idx + i) % len(ANIMALS)]
                    sprite = SPRITE_CACHE[(name, blink)]
                    tw = int(w * 1.6)
                    th = int(tw * sprite.shape[0] / sprite.shape[1])
                    # 스프라이트 아래쪽을 턱에 맞추고, 귀는 위로 튀어나오게
                    sx = int(x + w / 2 - tw / 2)
                    sy = int(y + h * 1.15) - th
                    overlay(out, sprite, sx, sy, tw, angle, pivot=(x + w / 2, y + h / 2))

                cv2.putText(out, f"[{ANIMALS[animal_idx]}]  1-4/space: animal  p: pixel bg  q: quit",
                            (10, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, out)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    animal_idx = (animal_idx + 1) % len(ANIMALS)
                elif ord("1") <= key < ord("1") + len(ANIMALS):
                    animal_idx = key - ord("1")
                elif key == ord("p"):
                    pixel_bg = not pixel_bg
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
