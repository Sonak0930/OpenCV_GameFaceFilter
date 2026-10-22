import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import depthai as dai
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from opencv02_pixel_animal import create_detector

WIDTH, HEIGHT, FPS = 640, 480, 30
WINDOW = "Sunglasses Filter"
HERE = Path(__file__).parent
ASSETS = HERE / "assets"
LANDMARK_MODEL = HERE / "face_landmarker.task"
LANDMARK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task")

# MediaPipe Face Mesh 랜드마크 번호
R_EYE = (33, 133)    # 화면 왼쪽 눈 (바깥, 안쪽 눈꼬리)
L_EYE = (362, 263)   # 화면 오른쪽 눈
FOREHEAD, CHIN = 10, 152
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


# ---------------------------------------------------------------------------
# 공통 그리기 도구 (float BGRA 캔버스, 0~1)
# ---------------------------------------------------------------------------
def over(canvas, color, alpha):
    """canvas 위에 color(튜플 or 이미지, 0~255)를 alpha(HxW, 0~1)로 덮어 그림"""
    a = alpha[..., None]
    ca = canvas[..., 3:4]
    out_a = a + ca * (1 - a)
    col = np.broadcast_to(np.asarray(color, np.float32) / 255, canvas[..., :3].shape)
    canvas[..., :3] = (col * a + canvas[..., :3] * ca * (1 - a)) / np.maximum(out_a, 1e-6)
    canvas[..., 3:4] = out_a


def poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255


def chaikin(pts, n=3):
    """다각형 모서리를 둥글게"""
    pts = np.asarray(pts, np.float32)
    for _ in range(n):
        nxt = np.roll(pts, -1, axis=0)
        pts = np.stack([0.75 * pts + 0.25 * nxt, 0.25 * pts + 0.75 * nxt], 1).reshape(-1, 2)
    return pts


def vgrad(shape, y0, y1, stops):
    """세로 그라데이션 이미지. stops = [(t, (b, g, r)), ...]"""
    h, w = shape[:2]
    t = np.clip((np.arange(h) - y0) / (y1 - y0), 0, 1)
    ts = [s[0] for s in stops]
    cols = np.stack([np.interp(t, ts, [s[1][c] for s in stops]) for c in range(3)], -1)
    return np.repeat(cols[:, None, :], w, axis=1).astype(np.float32)


def add_shadow(rgba, off=(12, 18), sigma=12, strength=0.35, pad=40):
    """부드러운 그림자를 깔고, 사방으로 pad만큼 키운 BGRA uint8 반환"""
    h, w = rgba.shape[:2]
    a = rgba[..., 3].astype(np.float32) / 255
    sh = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    sh[pad + off[1]:pad + off[1] + h, pad + off[0]:pad + off[0] + w] = a
    sh = cv2.GaussianBlur(sh, (0, 0), sigma) * strength
    out = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
    out[..., 3] = sh
    top = np.zeros_like(sh)
    top[pad:pad + h, pad:pad + w] = a
    col = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.float32)
    col[pad:pad + h, pad:pad + w] = rgba[..., :3]
    over(out, col, top)
    return (out * 255).clip(0, 255).astype(np.uint8)


def to_u8(canvas):
    return (canvas * 255).clip(0, 255).astype(np.uint8)


def paste_warp(frame, rgba, M, vis=None):
    """rgba를 변환 행렬 M(2x3 아핀 또는 3x3 원근)으로 frame에 합성. vis(HxW, 0~1)=가려짐 마스크"""
    H, W = frame.shape[:2]
    h, w = rgba.shape[:2]
    M3 = M if M.shape == (3, 3) else np.vstack([M, [0, 0, 1]])
    corners = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [0, h], [w, h]]]), M3)[0]
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x0 >= x1 or y0 >= y1:
        return
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    layer = cv2.warpPerspective(rgba, T @ M3, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderValue=(0, 0, 0, 0))
    a = layer[..., 3:4].astype(np.float32) / 255
    if vis is not None:
        a *= vis[y0:y1, x0:x1, None]
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = (a * layer[..., :3] + (1 - a) * roi).astype(np.uint8)


# ---------------------------------------------------------------------------
# 선글라스 (600x300 캔버스, 눈 중심이 (200,150) / (400,150)에 오도록 디자인)
# ---------------------------------------------------------------------------
GW, GH = 600, 300
G_PAD = 30
EYE_L, EYE_R = (200 + G_PAD, 150 + G_PAD), (400 + G_PAD, 150 + G_PAD)


def mirror_x(pts):
    pts = np.asarray(pts, np.float32).copy()
    pts[:, 0] = GW - pts[:, 0]
    return pts


def lens_shine(canvas, lens_alpha, strength=0.18):
    """렌즈 위 대각선 반사광"""
    for x in (150, 350):
        band = poly_mask(canvas.shape, [(x, 60), (x + 55, 60), (x - 25, 240), (x - 80, 240)])
        over(canvas, (255, 255, 255), band * lens_alpha * strength)


def dots(canvas, pts, r, color):
    m = np.zeros(canvas.shape[:2], np.float32)
    for p in pts:
        cv2.circle(m, (int(p[0]), int(p[1])), r, 1, -1, cv2.LINE_AA)
    over(canvas, color, m)


def glasses_gm_visor():
    """젠틀몬스터 실버 미러 바이저 (일체형 쉴드 렌즈)"""
    c = np.zeros((GH, GW, 4), np.float32)
    xs = np.linspace(60, 540, 40)
    top = np.stack([xs, 102 - 14 * (1 - ((xs - 300) / 240) ** 2)], 1)
    bottom = [(540, 168), (470, 195), (380, 210), (330, 200), (300, 176),
              (270, 200), (220, 210), (130, 195), (60, 168)]
    lens = poly_mask(c.shape, chaikin(np.vstack([top, bottom]), 2))

    mirror = vgrad(c.shape, 85, 215, [(0, (250, 245, 240)), (0.3, (215, 205, 200)), (0.48, (120, 112, 105)),
                                      (0.62, (175, 168, 160)), (1, (80, 75, 70))])
    xfade = 1 - 0.4 * ((np.arange(GW) - 300) / 240) ** 2
    mirror *= np.clip(xfade, 0.5, 1)[None, :, None]
    over(c, mirror, lens)
    hl = poly_mask(c.shape, chaikin([(90, 118), (510, 118), (500, 126), (100, 126)], 2))
    over(c, (255, 255, 255), hl * lens * 0.5)

    cv2.polylines(c, [np.round(top).astype(np.int32)], False, (0.08, 0.08, 0.08, 1), 9, cv2.LINE_AA)
    for side in (1, -1):
        pts = [(48, 98), (78, 92), (80, 172), (52, 160)]
        over(c, (25, 25, 28), poly_mask(c.shape, chaikin(pts if side == 1 else mirror_x(pts), 2)))
        x = 64 if side == 1 else GW - 64
        dots(c, [(x, 108 + i * 15) for i in range(4)], 2, (220, 220, 225))
    return c


def glasses_gm_square():
    """젠틀몬스터 오버사이즈 스퀘어 (블랙 + 실버 3도트)"""
    c = np.zeros((GH, GW, 4), np.float32)
    left = chaikin([(78, 82), (288, 86), (286, 200), (262, 228), (120, 226), (86, 196)], 2)
    lens = np.maximum(poly_mask(c.shape, left), poly_mask(c.shape, mirror_x(left)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (24, 24))
    frame = np.maximum(cv2.dilate(lens, k), np.roll(cv2.dilate(lens, k), -6, axis=0))
    over(c, vgrad(c.shape, 60, 245, [(0, (60, 60, 62)), (0.15, (20, 20, 22)), (1, (8, 8, 10))]), frame)
    over(c, vgrad(c.shape, 90, 230, [(0, (85, 85, 85)), (1, (45, 45, 47))]), lens * 0.88)
    lens_shine(c, lens, 0.1)
    for x in (88, GW - 88):  # 실버 3도트
        dots(c, [(x, 80), (x + (8 if x < 300 else -8), 80), (x + (4 if x < 300 else -4), 87)], 3, (225, 225, 230))
    return c


def _wayfarer(frame_grad, lens_grad, lens_alpha):
    c = np.zeros((GH, GW, 4), np.float32)
    left = chaikin([(90, 96), (292, 102), (284, 196), (262, 216), (142, 212), (116, 194)], 2)
    lens = np.maximum(poly_mask(c.shape, left), poly_mask(c.shape, mirror_x(left)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (34, 34))
    frame = cv2.dilate(lens, k)
    frame = np.maximum(frame, np.roll(frame, -9, axis=0))  # 윗테를 더 두껍게
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    over(c, vgrad(c.shape, 70, 235, frame_grad), frame)
    over(c, vgrad(c.shape, 100, 215, lens_grad), lens * lens_alpha)
    lens_shine(c, lens, 0.12)
    return c


def glasses_rayban():
    """레이반 웨이페어러 (블랙 유광 + G-15 그린 렌즈)"""
    c = _wayfarer([(0, (70, 70, 72)), (0.2, (25, 25, 27)), (1, (10, 10, 12))],
                  [(0, (55, 85, 55)), (1, (30, 52, 34))], 0.93)
    for x in (108, GW - 108):
        dots(c, [(x - 5, 92), (x + 5, 92)], 3, (215, 215, 220))
    t = np.zeros((GH, GW), np.float32)
    cv2.putText(t, "Ray-Ban", (112, 128), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, 1, 1, cv2.LINE_AA)
    over(c, (235, 235, 235), t * 0.8)
    return c


def glasses_rayban_meta():
    """레이반 x 메타 (무광 블랙 + 카메라/LED)"""
    c = _wayfarer([(0, (40, 40, 42)), (1, (18, 18, 20))], [(0, (75, 80, 72)), (1, (40, 44, 40))], 0.9)
    dots(c, [(100, 88)], 9, (5, 5, 5))
    dots(c, [(100, 88)], 5, (90, 60, 50))      # 카메라 렌즈
    dots(c, [(97, 85)], 2, (255, 255, 255))
    dots(c, [(GW - 100, 88)], 3, (250, 250, 255))  # LED
    return c


def glasses_from_photo(path):
    """흰 배경 정면 제품 사진 -> 배경 제거 후 두 렌즈 중심을 캔버스 눈 위치에 맞춤"""
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    alpha = np.clip((238 - gray) / 45, 0, 1)
    # 렌즈 = 프레임(아주 검정)보다 살짝 밝은 회색 영역 -> 가장 큰 두 덩어리의 중심
    lens = ((gray > 35) & (gray < 120)).astype(np.uint8)
    lens = cv2.morphologyEx(lens, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(lens)
    big = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]
    (ax, ay), (bx, by) = sorted([cent[i] for i in big], key=lambda p: p[0])
    rgba = np.dstack([img, (alpha * 255).astype(np.uint8)])
    src = np.float32([(ax, ay), (bx, by), (ax - (by - ay), ay + (bx - ax))])
    dst = np.float32([(200, 150), (400, 150), (200, 350)])
    canvas = cv2.warpAffine(rgba, cv2.getAffineTransform(src, dst), (GW, GH),
                            flags=cv2.INTER_AREA, borderValue=(255, 255, 255, 0))
    return canvas.astype(np.float32) / 255


GLASSES = [
    ("GENTLE MONSTER - Silver Visor", glasses_gm_visor, (40, 40, 44)),
    ("GENTLE MONSTER - Cat-eye (silver)", lambda: glasses_from_photo(ASSETS / "gm_cateye_silver.png"), (15, 15, 17)),
    ("GENTLE MONSTER - Cat-eye (gold)", lambda: glasses_from_photo(ASSETS / "gm_cateye_gold.png"), (15, 15, 17)),
    ("GENTLE MONSTER - Oversize Square", glasses_gm_square, (15, 15, 17)),
    ("Ray-Ban - Wayfarer G-15", glasses_rayban, (15, 15, 17)),
    ("Ray-Ban x Meta - Wayfarer", glasses_rayban_meta, (25, 25, 27)),
]


class Eyewear:
    def __init__(self, name, fn, arm_color):
        self.name = name
        self.arm_color = arm_color
        self.img = add_shadow(to_u8(fn()), off=(0, 12), sigma=6, strength=0.3, pad=G_PAD)
        ys, xs = np.nonzero(self.img[..., 3] > 200)
        # 다리(템플)가 붙는 힌지 = 프레임 좌우 끝, 위쪽 1/3 높이
        hy = ys.min() + 0.3 * (ys.max() - ys.min())
        self.hinges = [(xs.min() + 6, hy), (xs.max() - 6, hy)]


EYEWEAR = [Eyewear(*g) for g in GLASSES]


class FacePose:
    """랜드마크로 얼굴 3D 좌표계(원점=두 눈 사이, x=오른쪽, y=아래, z=머리 안쪽) 계산"""

    def __init__(self, P):
        er = P[list(R_EYE)].mean(0)  # 눈동자가 아닌 눈꼬리 평균 -> 시선이 움직여도 안경은 고정
        el = P[list(L_EYE)].mean(0)
        self.o = (er + el) / 2
        x = el - er
        self.d = np.linalg.norm(x)  # 두 눈 사이 거리 (스케일 기준)
        self.x = x / self.d
        y = P[CHIN] - P[FOREHEAD]
        y -= y.dot(self.x) * self.x
        self.y = y / np.linalg.norm(y)
        self.z = np.cross(self.x, self.y)

    def project(self, local):
        """얼굴 좌표(눈 사이 거리 단위) -> 화면 픽셀. 약한 원근 효과 포함"""
        local = np.atleast_2d(local)
        world = local[:, :1] * self.x + local[:, 1:2] * self.y + local[:, 2:3] * self.z
        world *= self.d
        f = 10 * self.d  # 카메라까지 거리 ~ 눈 사이 거리의 10배로 가정
        k = f / (f + world[:, 2:3])
        return self.o[:2] + world[:, :2] * k


def draw_eyewear(frame, ew, pose, scale=1.08, front=0.22):
    """안경 캔버스를 얼굴 앞 평면(front만큼 앞)에 놓고 원근 변환. 고개를 돌리면 다리도 보임"""
    cx, cy = (EYE_L[0] + EYE_R[0]) / 2, EYE_L[1]
    unit = (EYE_R[0] - EYE_L[0]) / scale

    def to_local(u, v, z=-front):
        return [(u - cx) / unit, (v - cy) / unit, z]

    # 다리: 고개를 돌려서 카메라 쪽으로 향한 쪽만 (반대쪽은 머리에 가려짐)
    turn = pose.x[2]
    for side, (hx, hy) in zip((-1, 1), ew.hinges):
        if side * turn < -0.12:
            p = pose.project([to_local(hx, hy), to_local(hx, hy + 6, 1.3), to_local(hx, hy + 30, 1.3),
                              to_local(hx, hy + 34)])
            cv2.fillConvexPoly(frame, np.round(p * 16).astype(np.int32), ew.arm_color, cv2.LINE_AA, shift=4)

    h, w = ew.img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = pose.project([to_local(*p) for p in src]).astype(np.float32)
    paste_warp(frame, ew.img, cv2.getPerspectiveTransform(src, dst))


# ---------------------------------------------------------------------------
# 얼굴 랜드마크 (MediaPipe) + 떨림 보정 (One Euro Filter)
# ---------------------------------------------------------------------------
class OneEuro:
    """느릴 땐 강하게, 빠를 땐 약하게 스무딩 -> 떨림은 없애고 반응은 빠르게"""

    def __init__(self, min_cutoff=1.2, beta=0.04, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._a(self.dc, dt) * (dx - self.dx)
        a = self._a(self.mc + self.beta * np.abs(self.dx), dt)
        self.x = self.x + a * (x - self.x)
        self.t = t
        return self.x


def ensure_landmark_model():
    if not LANDMARK_MODEL.exists():
        print("MediaPipe 얼굴 랜드마크 모델 다운로드 중...")
        urllib.request.urlretrieve(LANDMARK_URL, LANDMARK_MODEL)


class FaceLandmarks:
    """YuNet으로 얼굴을 찾고(작은/먼 얼굴에 강함) -> 얼굴을 확대한 크롭에서 MediaPipe 478점 랜드마크"""
    CROP = 256

    def __init__(self, hold=4):
        ensure_landmark_model()
        self.yunet = create_detector()
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(LANDMARK_MODEL)),
                                            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.hold = hold
        self.tracks = []  # {"c": 중심, "f": OneEuro, "P": 마지막 결과, "miss": 놓친 프레임 수}

    def _landmarks(self, frame, box):
        x, y, w, h = box
        side = 2.0 * max(w, h)
        cx, cy = x + w / 2, y + h / 2
        s = self.CROP / side
        M = np.float32([[s, 0, -(cx - side / 2) * s], [0, s, -(cy - side / 2) * s]])
        crop = cv2.warpAffine(frame, M, (self.CROP, self.CROP), borderMode=cv2.BORDER_REPLICATE)
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            return None
        P = np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], np.float64) * self.CROP / s
        P[:, 0] += cx - side / 2
        P[:, 1] += cy - side / 2
        return P

    def detect(self, frame, t):
        """[(478, 3) 픽셀 좌표 배열, ...] 반환 (z도 픽셀 단위, 작을수록 카메라에 가까움)"""
        self.yunet.setInputSize((frame.shape[1], frame.shape[0]))
        _, dets = self.yunet.detect(frame)
        faces = [P for P in (self._landmarks(frame, d[:4]) for d in ([] if dets is None else dets)) if P is not None]

        free = list(self.tracks)
        new = []
        for P in faces:  # 이전 프레임의 가장 가까운 얼굴과 매칭 -> 같은 필터로 스무딩
            c = P[:, :2].mean(0)
            near = min(free, key=lambda tr: np.linalg.norm(tr["c"] - c), default=None)
            if near is not None and np.linalg.norm(near["c"] - c) < 0.5 * np.ptp(P[:, 0]):
                free = [tr for tr in free if tr is not near]  # remove()는 dict 안의 numpy 배열을 == 비교해서 에러
                f = near["f"]
            else:
                f = OneEuro()
            new.append({"c": c, "f": f, "P": f(P, t), "miss": 0})
        for tr in free:  # 잠깐 놓친 얼굴은 몇 프레임 유지 (깜빡임 방지)
            tr["miss"] += 1
            if tr["miss"] <= self.hold:
                new.append(tr)
        self.tracks = new
        return [tr["P"] for tr in self.tracks]


def detect_still(path):
    """정지 이미지에서 한 번만 랜드마크 검출"""
    ensure_landmark_model()
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(LANDMARK_MODEL)))
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        img = cv2.imread(str(path))
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    h, w = img.shape[:2]
    if not res.face_landmarks:
        return img, None
    return img, np.array([(p.x * w, p.y * h, p.z * w) for p in res.face_landmarks[0]])


# ---------------------------------------------------------------------------
# 젠틀몬스터 스타일 거대 얼굴 조각상 (실제 얼굴 사진으로 만듦)
# ---------------------------------------------------------------------------
def sculpture_from_face(img, P, pale=0.35, tint=(1.0, 1.0, 1.0)):
    """얼굴 윤곽(랜드마크) 대로 오려내고 피부를 매끈하고 창백하게 -> 실리콘 마스크 느낌"""
    oval = P[FACE_OVAL, :2]
    c = oval.mean(0)
    oval = c + (oval - c) * np.array([1.04, 1.10])  # 이마/턱까지 살짝 넉넉하게
    x0, y0 = np.floor(oval.min(0)).astype(int) - 4
    x1, y1 = np.ceil(oval.max(0)).astype(int) + 4
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None
    crop = img[y0:y1, x0:x1]
    s = 420 / (y1 - y0)  # 높이 420px로 통일
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    oval = (oval - (x0, y0)) * s

    smooth = cv2.edgePreservingFilter(crop, flags=cv2.RECURS_FILTER, sigma_s=25, sigma_r=0.12)
    smooth = cv2.bilateralFilter(smooth, 9, 35, 7)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1 - pale                       # 채도 낮추기
    hsv[..., 2] = hsv[..., 2] * (1 - pale * 0.5) + 255 * pale * 0.5  # 밝게
    col = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    col *= np.array(tint, np.float32)

    mask = np.zeros(crop.shape[:2], np.float32)
    cv2.fillPoly(mask, [np.round(oval).astype(np.int32)], 1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((7, 7), np.uint8)), (0, 0), 4)
    # 가장자리를 살짝 어둡게 -> 두께감 있는 조각 느낌
    edge = cv2.GaussianBlur(mask, (0, 0), 18)
    col *= (0.72 + 0.28 * np.clip(edge * 1.6, 0, 1))[..., None]
    rgba = np.dstack([col.clip(0, 255), mask * 255]).astype(np.uint8)
    return add_shadow(rgba, off=(16, 24), sigma=16, strength=0.4)


class FloatingHeads:
    def __init__(self):
        img, P = detect_still(ASSETS / "gm_face.png")
        base = sculpture_from_face(img, P, pale=0.15) if P is not None else None
        self.rng = np.random.default_rng(7)
        self.heads = []
        if base is not None:
            variants = [base, cv2.flip(base, 1),
                        sculpture_from_face(img, P, 0.3, (1.0, 1.0, 0.96)),
                        cv2.flip(sculpture_from_face(img, P, 0.25, (1.04, 1.0, 0.98)), 1)]
            # (가로 위치, 세로 위치, 화면 높이 대비 크기) - 크게, 화면 밖으로 살짝 잘려도 OK
            spots = [(0.10, 0.40, 0.85), (0.92, 0.35, 0.95), (0.55, 0.05, 0.55), (0.30, 0.02, 0.45)]
            for v, s in zip(variants, spots):
                self._add(v, *s)

    def _add(self, img, fx, fy, size):
        self.heads.append(dict(img=img, fx=fx, fy=fy, size=size, ph=self.rng.uniform(0, 6.28),
                               sp=self.rng.uniform(0.4, 0.7)))

    def add_captured(self, img):
        self._add(img, self.rng.uniform(0.1, 0.9), self.rng.uniform(0.1, 0.35), self.rng.uniform(0.6, 0.85))
        if len(self.heads) > 7:
            self.heads.pop(0)

    def draw(self, frame, t, vis):
        H, W = frame.shape[:2]
        for hd in sorted(self.heads, key=lambda d: d["size"]):  # 작은(먼) 것부터
            img = hd["img"]
            ph, sp = hd["ph"], hd["sp"]
            cx = hd["fx"] * W + 25 * np.sin(t * sp * 0.6 + ph)
            cy = hd["fy"] * H + 18 * np.sin(t * sp + ph)  # 위아래로 둥둥
            scale = hd["size"] * H / img.shape[0]
            angle = 5 * np.sin(t * sp * 0.4 + ph * 2)
            yaw = 0.88 + 0.12 * np.cos(t * sp * 0.5 + ph)  # 천천히 좌우로 도는 느낌
            # 중심 이동 -> 가로 압축(yaw) + 확대 -> 회전 -> 화면 위치로
            a = np.radians(angle)
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            S = np.diag([scale * yaw, scale, 1])
            T0 = np.array([[1, 0, -img.shape[1] / 2], [0, 1, -img.shape[0] / 2], [0, 0, 1]])
            T1 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
            paste_warp(frame, img, T1 @ R @ S @ T0, vis)


def people_mask(shape, faces):
    """사람(머리+상체) 영역은 0 -> 조각상이 사람 뒤에 있는 것처럼 보이게"""
    H, W = shape[:2]
    vis = np.ones((H, W), np.float32)
    for P in faces:
        oval = P[FACE_OVAL, :2]
        c = oval.mean(0)
        head = c + (oval - c) * np.array([1.35, 1.45]) - (0, 0.12 * np.ptp(oval[:, 1]))  # 머리카락까지
        cv2.fillPoly(vis, [np.round(head).astype(np.int32)], 0)
        w = np.ptp(oval[:, 0])
        y = oval[:, 1].max()
        body = np.int32([(c[0] - 0.45 * w, y - 0.2 * w), (c[0] + 0.45 * w, y - 0.2 * w),
                         (c[0] + 1.5 * w, y + 0.8 * w), (c[0] + 1.7 * w, H), (c[0] - 1.7 * w, H),
                         (c[0] - 1.5 * w, y + 0.8 * w)])
        cv2.fillPoly(vis, [body], 0)
    return cv2.GaussianBlur(vis, (0, 0), 8)


def main():
    landmarks = FaceLandmarks()
    heads = FloatingHeads()
    style = 0
    show_heads = True
    t0 = time.time()

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
                t = time.time() - t0
                faces = landmarks.detect(frame, t)

                out = frame.copy()
                if show_heads:
                    heads.draw(out, t, people_mask(out.shape, faces))
                for P in faces:
                    draw_eyewear(out, EYEWEAR[style], FacePose(P))

                name = EYEWEAR[style].name
                cv2.putText(out, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(out, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(out, f"1-{len(EYEWEAR)}/space: glasses  h: heads  c: capture face  q: quit",
                            (10, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, out)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    style = (style + 1) % len(EYEWEAR)
                elif ord("1") <= key < ord("1") + len(EYEWEAR):
                    style = key - ord("1")
                elif key == ord("h"):
                    show_heads = not show_heads
                elif key == ord("c") and faces:  # 내 얼굴을 조각상으로 만들어 띄우기
                    img = sculpture_from_face(frame, faces[0])
                    if img is not None:
                        heads.add_captured(img)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
