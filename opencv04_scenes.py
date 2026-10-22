import datetime
from pathlib import Path

import cv2
import numpy as np
import depthai as dai

from opencv02_pixel_animal import create_detector, FaceTracker

WIDTH, HEIGHT, FPS = 640, 480, 30  # 1280x720 BGR은 USB 대역폭 초과로 장치가 크래시남
WINDOW = "Scene Filter"
SHOW_W = 1600  # 화면에 띄울 가로 크기
ASSETS = Path(__file__).with_name("assets")



# ---------------------------------------------------------------------------
# 공통: 얼굴 잘라내기 / 붙이기
# ---------------------------------------------------------------------------
def eye_angle(face):
    (rx, ry), (lx, ly) = face[4:6], face[6:8]
    return np.degrees(np.arctan2(ly - ry, lx - rx))


def face_patch(frame, face, size=128, zoom=1.3):
    """얼굴을 똑바로 세워(눈 기울기 보정) size x size 정사각형으로 잘라냄"""
    x, y, w, h = face[:4]
    c = (float(x + w / 2), float(y + h / 2))
    M = cv2.getRotationMatrix2D(c, eye_angle(face), size / (max(w, h) * zoom))
    M[:, 2] += (size / 2 - c[0], size / 2 - c[1])
    return cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REPLICATE)


def ellipse_alpha(size, rx=0.40, ry=0.47, blur=0.0):
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, size // 2), (int(size * rx), int(size * ry)), 0, 0, 360, 1, -1, cv2.LINE_AA)
    if blur:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return m


def paste_affine(frame, rgba, M):
    """BGRA 이미지를 아핀 행렬 M으로 변환해 frame에 알파 합성"""
    H, W = frame.shape[:2]
    h, w = rgba.shape[:2]
    corners = M @ np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    x0, y0 = np.floor(corners.min(1)).astype(int)
    x1, y1 = np.ceil(corners.max(1)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x0 >= x1 or y0 >= y1:
        return
    M2 = M.copy()
    M2[:, 2] -= (x0, y0)
    layer = cv2.warpAffine(rgba, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = layer[..., 3:4].astype(np.float32) / 255
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = (a * layer[..., :3] + (1 - a) * roi).astype(np.uint8)


def place(dst, rgba, center, width, angle=0.0):
    """rgba를 center에 가로 width 크기, angle 기울기로 합성"""
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, width / w)
    M[:, 2] += (center[0] - w / 2, center[1] - h / 2)
    paste_affine(dst, rgba, M)


def label(img, text, org, scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 1. PNN 뉴스: 벽 액자 속 화면에 내가 생중계로 등장
# ---------------------------------------------------------------------------
NEWS_BG = cv2.imread(str(ASSETS / "news.png"))
NEWS_FRAME = (391, 68, 554, 187)  # 액자 안쪽 (x0, y0, x1, y1)


def cartoon(img):
    """일러스트 톤: 매끈하게 + 색 단순화 + 외곽선 + 빈티지 웜톤"""
    s = img
    for _ in range(2):
        s = cv2.bilateralFilter(s, 7, 60, 7)
    s = np.minimum(s.astype(np.int32) // 40 * 40 + 20, 255).astype(np.uint8)  # uint8 넘침 방지
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    s = cv2.bitwise_and(s, s, mask=edges).astype(np.float32)
    s = s * (0.80, 0.90, 1.0) + (30, 35, 45)  # 배경 그림처럼 누렇게 바랜 색감
    return s.clip(0, 255).astype(np.uint8)


def scene_news(frame, faces, t):
    out = NEWS_BG.copy()
    x0, y0, x1, y1 = NEWS_FRAME
    fw, fh = x1 - x0, y1 - y0
    H, W = frame.shape[:2]

    # 얼굴을 중심으로 액자 비율에 맞게 잘라내기 (얼굴 없으면 전체 화면)
    if faces:
        f = faces[0]
        cx, cy, cw = f[0] + f[2] / 2, f[1] + f[3] * 0.65, f[2] * 3.0
    else:
        cx, cy, cw = W / 2, H / 2, W
    cw = min(cw, W, H * fw / fh)
    ch = cw * fh / fw
    sx = int(np.clip(cx - cw / 2, 0, W - cw))
    sy = int(np.clip(cy - ch / 2, 0, H - ch))
    crop = cv2.resize(frame[sy:sy + int(ch), sx:sx + int(cw)], (fw, fh), interpolation=cv2.INTER_AREA)
    crop = cartoon(crop)
    crop[::3] = (crop[::3] * 0.8).astype(np.uint8)  # 브라운관 스캔라인
    out[y0:y1, x0:x1] = crop

    if int(t * 2) % 2 == 0:  # 깜빡이는 LIVE 배지
        cv2.rectangle(out, (x0 + 5, y0 + 5), (x0 + 37, y0 + 18), (40, 40, 200), -1)
        cv2.putText(out, "LIVE", (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# 2. 킹 오브 파이터즈: 쿄 vs 이오리 얼굴을 내 얼굴(도트)로
# ---------------------------------------------------------------------------
KOF_BG = cv2.imread(str(ASSETS / "kof.png"))
KOF_SPRITES = [((183, 186), 34, -15), ((563, 187), 32, 8)]  # 쿄, 이오리 (얼굴 중심, 폭, 기울기)
KOF_PORTRAITS = [((36, 70), 66), ((704, 80), 66)]          # 양옆 초상화 (얼굴 중심, 폭)


def pixel_face(patch, px=12, size=96):
    """도트 스프라이트 얼굴: 저해상도 + 색 단순화 + 각진 알파"""
    small = cv2.resize(patch, (px, px), interpolation=cv2.INTER_AREA)
    small = cv2.convertScaleAbs(small, alpha=1.3, beta=-25)
    small = np.minimum(small.astype(np.int32) // 48 * 48 + 24, 255).astype(np.uint8)
    a = (cv2.resize(ellipse_alpha(px * 4, 0.42, 0.48), (px, px), interpolation=cv2.INTER_AREA) > 0.5)
    rgba = np.dstack([small, a.astype(np.uint8) * 255])
    return cv2.resize(rgba, (size, size), interpolation=cv2.INTER_NEAREST)


def portrait_face(patch):
    """양옆 어두운 초상화 톤 (붉은 그림자)"""
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    col = np.array([60, 80, 150], np.float32) * (0.3 + 1.0 * g)[..., None]
    a = ellipse_alpha(patch.shape[0], 0.40, 0.47, blur=8) * 0.85
    return np.dstack([col.clip(0, 255), a * 255]).astype(np.uint8)


def scene_kof(frame, faces, t):
    out = KOF_BG.copy()
    if not faces:
        return out
    for i, ((c, w, ang), (pc, pw)) in enumerate(zip(KOF_SPRITES, KOF_PORTRAITS)):
        f = faces[min(i, len(faces) - 1)]  # 한 명이면 나 vs 나
        place(out, pixel_face(face_patch(frame, f, zoom=1.15)), c, w, ang)
        place(out, portrait_face(face_patch(frame, f, zoom=1.4)), pc, pw)
    return out


# ---------------------------------------------------------------------------
# 3. 머그샷: 키 재는 벽 + 얼굴 아래 까만 명판 들고 있기
# ---------------------------------------------------------------------------
def height_chart(W, H, inch=18, top=40):
    """연회색 벽에 1인치마다 선, 3인치마다 굵은 선 + 키 눈금"""
    bg = np.full((H, W, 3), 238, np.uint8)
    for i, y in enumerate(range(top, H, inch)):
        hgt = 84 - i  # 7'0" = 84인치부터 아래로
        major = i % 3 == 0
        cv2.line(bg, (70, y), (W - 70, y), (20, 20, 20), 3 if major else 1, cv2.LINE_AA)
        if major:
            txt = f"{hgt // 12}'{hgt % 12}\""
            for x in (12, W - 62):
                cv2.putText(bg, txt, (x, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return bg


def make_plate(num, skin, PW=320, PH=170, pad=40):
    """까만 머그샷 명판 + 양손 (BGRA, 명판 중심이 캔버스 중심)"""
    c = np.zeros((PH + 2 * pad, PW + 2 * pad, 4), np.uint8)
    x0, y0, x1, y1 = pad, pad, pad + PW, pad + PH
    cv2.rectangle(c, (x0, y0), (x1, y1), (20, 20, 20, 255), -1)
    cv2.rectangle(c, (x0 + 5, y0 + 5), (x1 - 5, y1 - 5), (78, 78, 78, 255), -1)
    ink = (225, 225, 225, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    date = datetime.date.today().strftime("%y%m%d") + "04"
    cv2.putText(c, "#", (x0 + 18, y0 + 35), font, 0.7, ink, 2, cv2.LINE_AA)
    cv2.putText(c, f"{num}", (x0 + 95, y0 + 55), font, 1.0, ink, 2, cv2.LINE_AA)
    cv2.line(c, (x0 + 150, y0 + 15), (x0 + 150, y0 + 58), ink, 3)
    cv2.putText(c, "DATE", (x0 + 162, y0 + 28), font, 0.45, ink, 1, cv2.LINE_AA)
    cv2.putText(c, date, (x0 + 200, y0 + 52), font, 0.6, ink, 1, cv2.LINE_AA)
    cv2.line(c, (x0 + 15, y0 + 68), (x1 - 15, y0 + 68), ink, 2)
    cv2.putText(c, "ID", (x0 + 18, y0 + 100), font, 0.5, ink, 1, cv2.LINE_AA)
    cv2.putText(c, f"00{3146700 + num * 13}", (x0 + 70, y0 + 105), font, 1.0, ink, 2, cv2.LINE_AA)
    cv2.line(c, (x0 + 15, y0 + 120), (x1 - 15, y0 + 120), ink, 2)
    cv2.putText(c, "Central police department", (x0 + 45, y0 + 152), font, 0.55, ink, 1, cv2.LINE_AA)

    # 명판 아래 양 모서리를 쥔 주먹 (손가락 3줄)
    skin = tuple(int(v) for v in skin) + (255,)
    dark = tuple(int(v * 0.6) for v in skin[:3]) + (255,)
    for sx in (x0 + 10, x1 - 10):
        ctr = (sx, y1 - 25)
        cv2.ellipse(c, ctr, (36, 48), 0, 0, 360, (15, 15, 15, 255), -1, cv2.LINE_AA)
        cv2.ellipse(c, ctr, (33, 45), 0, 0, 360, skin, -1, cv2.LINE_AA)
        d = 1 if sx < PW / 2 + pad else -1
        for k in range(3):
            y = ctr[1] - 22 + k * 17
            cv2.line(c, (sx - d * 30, y), (sx + d * 8, y), dark, 2, cv2.LINE_AA)
    return c


def skin_color(frame, face):
    x, y, w, h = (int(v) for v in face[:4])
    roi = frame[max(y + h // 2, 0):max(y + int(h * 0.75), 1), max(x + w // 3, 0):max(x + 2 * w // 3, 1)]
    return np.median(roi.reshape(-1, 3), 0) if roi.size else (150, 170, 210)


CHART = height_chart(WIDTH, HEIGHT)


JUMPSUIT = (40, 100, 230)
PLATE_V, PLATE_W = 1.25, 1.9   # 명판 중심 위치 / 폭 (얼굴 크기 단위)
HAND_U, HAND_V = 0.71, 1.535   # make_plate 속 주먹 위치를 얼굴 단위로 환산한 값


def mug_face(frame, face, size=200, zoom=1.7):
    """머리카락까지 포함해 얼굴만 타원으로 오려냄 (플래시 톤)"""
    patch = cv2.convertScaleAbs(face_patch(frame, face, size, zoom), alpha=1.15, beta=10)
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, int(size * 0.46)), (int(size * 0.34), int(size * 0.44)), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), size * 0.02)
    return np.dstack([patch, m * 255]).astype(np.uint8)


def draw_body(out, P, s, skin):
    """얼굴 기준 좌표 P(u, v)로 목/죄수복 상체를 그림"""
    lw = max(2, int(s * 0.03))

    def poly(pts, col):
        p = np.int32([P(u, v) for u, v in pts])
        cv2.fillPoly(out, [p], col, cv2.LINE_AA)
        cv2.polylines(out, [p], True, (15, 15, 15), lw, cv2.LINE_AA)

    poly([(-0.2, 0.3), (0.2, 0.3), (0.22, 0.85), (-0.22, 0.85)], skin)                     # 목
    poly([(-0.32, 0.72), (-1.2, 0.95), (-1.5, 1.5), (-1.7, 10), (1.7, 10), (1.5, 1.5),
          (1.2, 0.95), (0.32, 0.72), (0.25, 0.92), (0, 1.0), (-0.25, 0.92)], JUMPSUIT)      # 상체
    poly([(-0.32, 0.72), (-0.25, 0.92), (0, 1.0), (0.25, 0.92), (0.32, 0.72),
          (0.22, 0.85), (0, 0.9), (-0.22, 0.85)], (235, 235, 235))                            # 흰 티 깃


def draw_arms(out, P, s):
    """어깨 -> 팔꿈치 -> 명판 쥔 주먹까지 팔"""
    th = int(s * 0.36)
    for d in (-1, 1):
        pts = np.int32([P(d * 1.2, 1.15), P(d * 1.3, 2.25), P(d * HAND_U, HAND_V)])
        cv2.polylines(out, [pts], False, (15, 15, 15), th + max(4, int(s * 0.06)), cv2.LINE_AA)
        cv2.polylines(out, [pts], False, JUMPSUIT, th, cv2.LINE_AA)


def scene_mugshot(frame, faces, t):
    """배경은 전부 키 재는 벽, 카메라에서는 얼굴만 오려 붙이고 몸/팔/명판은 얼굴에 맞춰 그림"""
    out = CHART.copy()
    for i, f in enumerate(faces):
        x, y, w, h = f[:4]
        s = max(w, h)
        a = np.radians(eye_angle(f))
        c = np.array([x + w / 2, y + h / 2])
        R, D = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        P = lambda u, v: c + (R * u + D * v) * s  # noqa: E731
        skin = tuple(int(v) for v in skin_color(frame, f))

        draw_body(out, P, s, skin)
        place(out, mug_face(frame, f), c, s * 1.7, -np.degrees(a))
        draw_arms(out, P, s)
        place(out, make_plate(45 + i * 92, skin), P(0, PLATE_V), s * PLATE_W, -np.degrees(a))
    return out


# ---------------------------------------------------------------------------
# 얼굴을 만화 톤으로 오려내기 (설거지 / 회전초밥 / 물고기 공용)
# ---------------------------------------------------------------------------
def toon_face(patch):
    s = patch
    for _ in range(2):
        s = cv2.bilateralFilter(s, 9, 50, 9)  # 피부 매끈하게
    return cv2.convertScaleAbs(s, alpha=1.1, beta=15)


def cut_face(frame, face, size=160, zoom=1.25, rx=0.38, ry=0.47):
    patch = toon_face(face_patch(frame, face, size, zoom))
    a = ellipse_alpha(size, rx, ry, blur=size * 0.025)
    return np.dstack([patch, a * 255]).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. 광란의 수족관: 사람 얼굴 물고기가 헤엄치며 동전을 떨어뜨림
# ---------------------------------------------------------------------------
def clean_aquarium(bg):
    """원래 있던 노랑/빨강 물고기를 지우고 주변 색으로 메꿈"""
    H, S, V = cv2.split(cv2.cvtColor(bg, cv2.COLOR_BGR2HSV))
    m = ((H >= 8) & (H <= 35) & (S > 110) & (V > 140)) | (((H <= 8) | (H >= 165)) & (S > 120) & (V > 110))
    m = m.astype(np.uint8) * 255
    m[:50] = 0  # 위쪽 상점 바는 그대로
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, cnts, -1, 255, -1)  # 눈알 구멍까지 채우기
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.inpaint(bg, m, 6, cv2.INPAINT_TELEA)


AQUA_BG = cv2.resize(clean_aquarium(cv2.imread(str(ASSETS / "aquarium.png"))), None, fx=2, fy=2,
                     interpolation=cv2.INTER_CUBIC)
AQUA_SCORE = (912, 58, 990, 90)  # 오른쪽 위 돈 표시 칸 (2배 좌표)
AQUA_FLOOR = 540


def make_fish(face_rgba=None, W=120, H=80):
    """오른쪽을 보는 금붕어 (머리 자리에 사람 얼굴)"""
    c = np.zeros((H, W, 4), np.uint8)
    ink, red = (20, 20, 20, 255), (70, 50, 230, 255)
    for pts in ([(4, 12), (34, 40), (4, 68), (14, 40)], [(40, 20), (60, 2), (84, 16)]):  # 꼬리, 등지느러미
        p = np.int32(pts)
        cv2.fillPoly(c, [p], red, cv2.LINE_AA)
        cv2.polylines(c, [p], True, ink, 2, cv2.LINE_AA)
    cv2.ellipse(c, (64, 42), (44, 30), 0, 0, 360, ink, -1, cv2.LINE_AA)
    cv2.ellipse(c, (64, 42), (42, 28), 0, 0, 360, (40, 185, 250, 255), -1, cv2.LINE_AA)
    cv2.ellipse(c, (62, 54), (30, 12), 0, 0, 360, (120, 225, 255, 255), -1, cv2.LINE_AA)  # 배
    cv2.ellipse(c, (58, 56), (11, 6), 30, 0, 360, red, -1, cv2.LINE_AA)                  # 옆지느러미
    if face_rgba is None:
        cv2.circle(c, (90, 36), 9, (255, 255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(c, (92, 36), 5, ink, -1, cv2.LINE_AA)
        return c
    d, (fx, fy) = 48, (84, 40)
    cv2.circle(c, (fx, fy), d // 2 + 2, ink, -1, cv2.LINE_AA)
    face = cv2.resize(face_rgba, (d, d), interpolation=cv2.INTER_AREA)
    a = face[..., 3:4].astype(np.float32) / 255
    roi = c[fy - d // 2:fy + d // 2, fx - d // 2:fx + d // 2, :3]
    roi[:] = (a * face[..., :3] + (1 - a) * roi).astype(np.uint8)
    return c


def draw_coin(img, x, y, gold, spin):
    r = 13
    rx = max(2, int(r * abs(np.cos(spin))))  # 빙글빙글 도는 동전
    base, rim = ((40, 190, 245), (20, 120, 180)) if gold else ((205, 205, 210), (130, 130, 135))
    x, y = int(x), int(y)
    cv2.ellipse(img, (x, y), (rx, r), 0, 0, 360, rim, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (max(1, rx - 3), r - 3), 0, 0, 360, base, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x - rx // 3, y - r // 3), (max(1, rx // 4), r // 4), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)


class Aquarium:
    def __init__(self, n=8):
        self.rng = np.random.default_rng(3)
        r = self.rng
        self.fish = [dict(x=r.uniform(80, 900), y=r.uniform(160, 460), vx=r.choice([-1, 1]) * r.uniform(50, 110),
                          ty=r.uniform(160, 460), ph=r.uniform(0, 6.28), size=r.uniform(100, 140),
                          drop=r.uniform(0.5, 4)) for _ in range(n)]
        self.coins = []
        self.score = 1885
        self.last = None

    def __call__(self, frame, faces, t):
        dt = 0.0 if self.last is None else min(t - self.last, 0.1)
        self.last = t
        r = self.rng
        out = AQUA_BG.copy()

        for f in self.fish:
            f["x"] += f["vx"] * dt
            if (f["x"] < 70 and f["vx"] < 0) or (f["x"] > 930 and f["vx"] > 0):
                f["vx"] = -f["vx"]  # 벽에서 방향 전환
            if r.random() < dt * 0.3:
                f["ty"] = r.uniform(160, 460)
            f["y"] += (f["ty"] - f["y"]) * dt * 0.8
            f["drop"] -= dt
            if f["drop"] <= 0:
                self.coins.append(dict(x=f["x"], y=f["y"] + 25, vy=0.0, gold=r.random() < 0.4,
                                       ph=r.uniform(0, 6.28), life=2.5))
                f["drop"] = r.uniform(2, 5)

        for c in self.coins:
            if c["y"] < AQUA_FLOOR:
                c["vy"] = min(c["vy"] + 300 * dt, 120)
                c["y"] = min(c["y"] + c["vy"] * dt, AQUA_FLOOR)
                spin = t * 6 + c["ph"]
            else:
                c["life"] -= dt
                spin = 0
            draw_coin(out, c["x"], c["y"], c["gold"], spin)
        for c in self.coins:
            if c["life"] <= 0:  # 바닥에 떨어진 동전은 잠시 뒤 자동 수거
                self.score += 35 if c["gold"] else 15
        self.coins = [c for c in self.coins if c["life"] > 0]

        sprites = [make_fish(cut_face(frame, f, 96, 1.15, 0.46, 0.46)) for f in faces] or [make_fish()]
        for i, f in enumerate(self.fish):
            spr = sprites[i % len(sprites)]
            if f["vx"] < 0:
                spr = cv2.flip(spr, 1)
            y = f["y"] + 6 * np.sin(t * 3 + f["ph"])
            place(out, spr, (f["x"], y), f["size"], 5 * np.sin(t * 2 + f["ph"]))

        x0, y0, x1, y1 = AQUA_SCORE
        cv2.rectangle(out, (x0, y0), (x1, y1), (35, 110, 35), -1)
        cv2.putText(out, str(self.score), (x0 + 10, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out


# ---------------------------------------------------------------------------
# 5. 슈의 설거지나 해: 슈 얼굴 자리에 내 얼굴
# ---------------------------------------------------------------------------
DISH_BG = cv2.imread(str(ASSETS / "dishes.png"))
SUE_FACE = ((332, 200), 132)  # (얼굴 중심, 오려낸 얼굴 폭)


def scene_dishes(frame, faces, t):
    out = DISH_BG.copy()
    if faces:
        place(out, cut_face(frame, faces[0]), *SUE_FACE)
    # 이마에서 흘러내리는 식은땀
    k = (t * 0.6) % 1
    x, y = 352, int(165 + 45 * k)
    tip = np.int32([(x - 7, y), (x, y - 16), (x + 7, y)])
    for col, grow in (((200, 140, 60), 2), ((250, 225, 170), 0)):
        cv2.circle(out, (x, y + 2), 7 + grow, col, -1, cv2.LINE_AA)
        cv2.fillPoly(out, [tip + (0, -grow)], col, cv2.LINE_AA)
    cv2.circle(out, (x - 2, y + 1), 2, (255, 255, 255), -1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# 6. 슈의 회전초밥: 손님 3명 얼굴 = 우리, 앞으로 초밥 접시가 지나감
# ---------------------------------------------------------------------------
SUSHI_BG = cv2.imread(str(ASSETS / "sushi.png"))
SUSHI_SEATS = [((137, 228), 122), ((373, 232), 110), ((610, 228), 106)]
BELT_Y, PLATE_GAP, BELT_SPEED = 338, 120, 70
TOPPING_BOXES = [10, 115, 222, 328, 435, 543, 650]  # 아래 재료 칸 x 시작 (y 378~418)
PLATE_RIMS = [(180, 110, 40), (50, 50, 200), (40, 170, 230)]


def make_nigiri(topping, rim):
    """재료 칸에서 잘라온 토핑 + 밥 + 접시"""
    c = np.zeros((70, 110, 4), np.uint8)
    cv2.ellipse(c, (55, 52), (52, 15), 0, 0, 360, rim + (255,), -1, cv2.LINE_AA)
    cv2.ellipse(c, (55, 50), (44, 10), 0, 0, 360, (240, 240, 245, 255), -1, cv2.LINE_AA)
    cv2.ellipse(c, (55, 42), (35, 14), 0, 0, 360, (170, 170, 170, 255), -1, cv2.LINE_AA)
    cv2.ellipse(c, (55, 42), (33, 12), 0, 0, 360, (235, 242, 248, 255), -1, cv2.LINE_AA)
    top = cv2.resize(topping, (78, 26), interpolation=cv2.INTER_AREA)
    m = np.zeros((26, 78), np.uint8)
    cv2.ellipse(m, (39, 13), (38, 12), 0, 0, 360, 255, -1, cv2.LINE_AA)
    a = m[..., None].astype(np.float32) / 255
    roi = c[20:46, 16:94]
    roi[..., :3] = (a * top + (1 - a) * roi[..., :3]).astype(np.uint8)
    roi[..., 3] = np.maximum(roi[..., 3], m)
    return c


NIGIRI = [make_nigiri(SUSHI_BG[383:413, x + 12:x + 78], PLATE_RIMS[i % 3]) for i, x in enumerate(TOPPING_BOXES)]


def scene_sushi(frame, faces, t):
    out = SUSHI_BG.copy()
    if faces:
        for i, (c, w) in enumerate(SUSHI_SEATS):
            place(out, cut_face(frame, faces[i % len(faces)]), c, w)
    W = out.shape[1]
    n = W // PLATE_GAP + 2
    for i in range(n):
        x = (t * BELT_SPEED + i * PLATE_GAP) % (n * PLATE_GAP) - PLATE_GAP / 2
        place(out, NIGIRI[i % len(NIGIRI)], (x, BELT_Y), 100)
    return out


# ---------------------------------------------------------------------------
SCENES = [
    ("PNN News", scene_news, cv2.INTER_LINEAR),
    ("The King of Fighters", scene_kof, cv2.INTER_NEAREST),
    ("Mugshot", scene_mugshot, cv2.INTER_LINEAR),
    ("Insaniquarium", Aquarium(), cv2.INTER_LINEAR),
    ("Sue's Dish Washing", scene_dishes, cv2.INTER_LINEAR),
    ("Sue's Sushi", scene_sushi, cv2.INTER_LINEAR),
]


def main():
    detector = create_detector()
    tracker = FaceTracker()
    scene = 0
    t0 = datetime.datetime.now()

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        output = cam.requestOutput(size=(WIDTH, HEIGHT), type=dai.ImgFrame.Type.BGR888p, fps=FPS)
        queue = output.createOutputQueue(maxSize=1, blocking=False)  # 밀린 프레임 버리고 항상 최신 것만
        pipeline.start()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 창 크기 조절 / 최대화 가능

        try:
            while pipeline.isRunning():
                msg = queue.get()
                if msg is None:
                    continue
                frame = msg.getCvFrame()

                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, dets = detector.detect(frame)
                faces = tracker.update([] if dets is None else [d[:14] for d in dets])
                faces = sorted(faces, key=lambda f: f[0])  # 화면 왼쪽 사람부터

                name, fn, interp = SCENES[scene]
                t = (datetime.datetime.now() - t0).total_seconds()
                out = fn(frame, faces, t)
                out = cv2.resize(out, (SHOW_W, int(out.shape[0] * SHOW_W / out.shape[1])), interpolation=interp)
                label(out, name, (12, 30))
                label(out, "c: next scene  q: quit", (12, out.shape[0] - 12), 0.5)
                cv2.imshow(WINDOW, out)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    scene = (scene + 1) % len(SCENES)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
