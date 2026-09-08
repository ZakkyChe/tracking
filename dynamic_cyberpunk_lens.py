"""
=========================================================================
 DYNAMIC CYBERPUNK LENS (v3)
=========================================================================
Mode:
  - Jempol + Telunjuk (2 tangan)             -> BIRU
  - Jempol + Jari Tengah (2 tangan)          -> NEGATIF (invert warna)
  - Jempol + Kelingking (2 tangan)           -> GLITCH
  - Telunjuk + Kelingking terbuka (jari
    tengah & jari manis terlipat)            -> MIRROR (flip + tint pink)
  - Telunjuk + Jari Tengah terbuka (jari
    manis & kelingking terlipat, tanpa
    jempol)                                  -> PHOTO WINDOW (lihat #6 di README)
  - Jempol + Telunjuk, salah satu tangan
    diputar 180 / menghadap ke BAWAH         -> BLUR
  - Kedua tangan mengepal penuh (BATU,
    tidak ada jari yang terangkat)           -> IMPACT FRAME (flash kontras
                                                 ekstrem + speed lines +
                                                 screen shake, background
                                                 digelapkan, ala adegan
                                                 "hantaman" di manga)

Area efek sekarang mengikuti POSISI JARI (polygon dari titik jempol &
jari aktif kedua tangan), bukan kotak lurus (bounding box) seperti versi
sebelumnya. Jadi bentuknya akan berubah-ubah mengikuti gestur tangan.
Khusus IMPACT FRAME, efeknya diterapkan ke SELURUH layar (bukan cuma
area jari) karena tujuannya bikin momen "hantaman" terasa heboh.

Kebutuhan:
  pip install "mediapipe==0.10.14" "numpy<2" "opencv-python<4.10" "opencv-contrib-python<4.10"

Jalankan:
  python dynamic_cyberpunk_lens.py
Tekan 'q' untuk keluar.
=========================================================================
"""

import time
import math
import random
from collections import Counter
import cv2
import numpy as np
import mediapipe as mp

# -------------------------------------------------------------------------
# KONFIGURASI
# -------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
WINDOW_NAME = "Dynamic Cyberpunk Lens"

NEON_MAGENTA = (255, 0, 255)
NEON_BLUE = (255, 120, 0)
NEON_GREEN = (0, 255, 140)
NEON_PURPLE = (200, 60, 160)
NEON_RED = (60, 60, 255)
NEON_PINK = (180, 60, 255)
NEON_ORANGE = (0, 165, 255)

MIN_BOX_SIZE = 40
MIN_SPREAD = 0.05           # jarak minimum (normalized) jempol-jari, hindari noise
DOWN_RATIO = 1.15           # makin besar = makin sulit ke-trigger "menghadap bawah"
BLUR_KSIZE = 50

# --- Konfigurasi IMPACT FRAME (mode "batu"/kepalan penuh) ---
IMPACT_DARKEN_ALPHA = 0.22       # seberapa gelap background dasar (0-1, makin kecil makin gelap)
IMPACT_THRESHOLD = 110           # ambang batas kontras ekstrem (grayscale threshold)
IMPACT_NUM_SPEEDLINES = 40
IMPACT_SHAKE_MAGNITUDE = 18      # px, seberapa kuat efek getar layar

# Path foto statis untuk mode "photo window". Foto ini akan di-resize
# supaya SAMA PERSIS ukurannya dengan frame kamera dan diam di tempat
# (nempel ke koordinat layar) -- gerakin tangan buat "buka" bagian foto
# yang berbeda, fotonya sendiri nggak ikut bergerak.
PHOTO_PATH = "download.jpeg"   # <-- GANTI dengan path foto kamu

WRIST = 0
THUMB_TIP = 4
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

# -------------------------------------------------------------------------
# INISIALISASI MEDIAPIPE HANDS
# -------------------------------------------------------------------------
try:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
    from mediapipe.python.solutions import drawing_styles as mp_draw_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


FINGER_TIP_BY_NAME = {
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "pinky": PINKY_TIP,
}

# urutan prioritas kalau terjadi seri antara 2 tangan dengan gestur berbeda
GESTURE_PRIORITY = ["index", "middle", "pinky"]


def get_finger_choice(hand_landmarks):
    """
    Tentukan jari mana (telunjuk / jari tengah / kelingking) yang paling
    'terbuka' pada satu tangan. Selalu mengembalikan hasil pasti.
    """
    lm = hand_landmarks.landmark
    wrist = lm[WRIST]

    scores = {
        "index": _dist(lm[INDEX_TIP], wrist) - _dist(lm[INDEX_PIP], wrist),
        "middle": _dist(lm[MIDDLE_TIP], wrist) - _dist(lm[MIDDLE_PIP], wrist),
        "pinky": _dist(lm[PINKY_TIP], wrist) - _dist(lm[PINKY_PIP], wrist),
    }
    return max(scores, key=scores.get)


def is_index_pinky_sign(hand_landmarks):
    """
    True kalau tangan ini membentuk pose TELUNJUK + KELINGKING terbuka
    sementara jari tengah & jari manis terlipat (mirip gestur "rock" /
    tanda tanduk). Ini trigger untuk efek MIRROR.
    """
    lm = hand_landmarks.landmark
    wrist = lm[WRIST]

    index_open = _dist(lm[INDEX_TIP], wrist) - _dist(lm[INDEX_PIP], wrist)
    pinky_open = _dist(lm[PINKY_TIP], wrist) - _dist(lm[PINKY_PIP], wrist)
    middle_open = _dist(lm[MIDDLE_TIP], wrist) - _dist(lm[MIDDLE_PIP], wrist)
    ring_open = _dist(lm[RING_TIP], wrist) - _dist(lm[RING_PIP], wrist)

    return index_open > 0 and pinky_open > 0 and middle_open <= 0 and ring_open <= 0


def is_index_middle_sign(hand_landmarks):
    """
    True kalau tangan ini membentuk pose TELUNJUK + JARI TENGAH terbuka
    (tanpa jempol) sementara jari manis & kelingking terlipat -- mirip
    tanda 'peace' tapi cuma 2 jari berdekatan. Trigger untuk mode
    PHOTO WINDOW.
    """
    lm = hand_landmarks.landmark
    wrist = lm[WRIST]

    index_open = _dist(lm[INDEX_TIP], wrist) - _dist(lm[INDEX_PIP], wrist)
    middle_open = _dist(lm[MIDDLE_TIP], wrist) - _dist(lm[MIDDLE_PIP], wrist)
    ring_open = _dist(lm[RING_TIP], wrist) - _dist(lm[RING_PIP], wrist)
    pinky_open = _dist(lm[PINKY_TIP], wrist) - _dist(lm[PINKY_PIP], wrist)

    return index_open > 0 and middle_open > 0 and ring_open <= 0 and pinky_open <= 0


def is_fist(hand_landmarks):
    """
    True kalau tangan ini mengepal penuh (BATU) -- telunjuk, jari
    tengah, jari manis, DAN kelingking semuanya terlipat (tidak ada
    satu pun jari yang terangkat). Ini trigger untuk efek IMPACT FRAME.
    """
    lm = hand_landmarks.landmark
    wrist = lm[WRIST]

    index_open = _dist(lm[INDEX_TIP], wrist) - _dist(lm[INDEX_PIP], wrist)
    middle_open = _dist(lm[MIDDLE_TIP], wrist) - _dist(lm[MIDDLE_PIP], wrist)
    ring_open = _dist(lm[RING_TIP], wrist) - _dist(lm[RING_PIP], wrist)
    pinky_open = _dist(lm[PINKY_TIP], wrist) - _dist(lm[PINKY_PIP], wrist)

    return (
        index_open <= 0
        and middle_open <= 0
        and ring_open <= 0
        and pinky_open <= 0
    )


def choose_gesture(finger_choices):
    """
    Gabungkan pilihan jari dari kedua tangan jadi satu gestur.
    Kalau kedua tangan sama -> gestur itu langsung dipakai.
    Kalau beda -> menang berdasarkan urutan prioritas GESTURE_PRIORITY.
    """
    counts = Counter(finger_choices)
    top_count = max(counts.values())
    candidates = [name for name, c in counts.items() if c == top_count]
    for name in GESTURE_PRIORITY:
        if name in candidates:
            return name
    return candidates[0]


def is_thumb_index_facing_down(hand_landmarks):
    """
    True jika garis JEMPOL -> TELUNJUK pada tangan ini mengarah ke BAWAH
    (telunjuk berada di bawah jempol) dan cukup vertikal -> menandakan
    tangan diputar ~180 derajat dari pose normal (frame menghadap bawah).
    """
    lm = hand_landmarks.landmark
    thumb = lm[THUMB_TIP]
    tip = lm[INDEX_TIP]

    dx = tip.x - thumb.x
    dy = tip.y - thumb.y  # y makin besar = makin ke bawah (koordinat gambar)

    if math.hypot(dx, dy) < MIN_SPREAD:
        return False

    # harus condong vertikal DAN mengarah ke bawah (dy positif)
    return dy > 0 and abs(dy) > abs(dx) * DOWN_RATIO


def get_active_points(hand_list, frame_w, frame_h, tip_ids):
    """tip_ids: tip landmark id untuk MASING-MASING tangan (index/middle)."""
    points = []
    for hand_landmarks, tip_id in zip(hand_list, tip_ids):
        for idx in (THUMB_TIP, tip_id):
            lm = hand_landmarks.landmark[idx]
            points.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    return points


def compute_polygon(points, frame_w, frame_h):
    """
    Bangun polygon (convex hull) dari titik-titik jempol & jari aktif
    kedua tangan, supaya area efek mengikuti bentuk jari, bukan selalu
    kotak lurus.
    """
    if len(points) < 4:
        return None

    pts = np.array(points, dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, frame_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, frame_h - 1)

    hull = cv2.convexHull(pts)

    x, y, w, h = cv2.boundingRect(hull)
    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
        return None

    return hull


def add_scanlines(roi, gap=4, alpha=0.2):
    overlay = roi.copy()
    for y in range(0, roi.shape[0], gap):
        cv2.line(overlay, (0, y), (roi.shape[1], y), (0, 0, 0), 1)
    return cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


# -------------------------------------------------------------------------
# EFEK (beroperasi di atas potongan ROI, dipanggil lewat apply_effect)
# -------------------------------------------------------------------------
def fx_blue(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    b = gray
    g = (gray.astype(np.float32) * 0.35).astype(np.uint8)
    r = np.zeros_like(gray)
    return add_scanlines(cv2.merge([b, g, r]))


def fx_negative(roi):
    inv = cv2.bitwise_not(roi)
    return add_scanlines(inv)


def fx_blur(roi, ksize=BLUR_KSIZE):
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(roi, (k, k), 0)


def fx_mirror(roi):
    """
    Efek mirror: gambar dibalik horizontal, lalu diberi kontras sedikit
    lebih tinggi dan dicampur (tint) dengan warna pink.
    """
    mirrored = cv2.flip(roi, 1)

    # kontras sedikit lebih tinggi (alpha > 1)
    contrasted = cv2.convertScaleAbs(mirrored, alpha=1.2, beta=0)

    pink_layer = np.full_like(contrasted, (180, 60, 255))  # BGR: pink
    tinted = cv2.addWeighted(contrasted, 0.75, pink_layer, 0.25, 0)

    return add_scanlines(tinted, alpha=0.15)


def fx_glitch(roi):
    """
    Efek glitch ala digital-distortion: channel warna digeser secara
    acak (chromatic shift) + beberapa strip horizontal digeser
    horizontal, ditambah noise garis tipis.
    """
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return roi

    out = roi.copy()

    # 1. Chromatic shift: geser channel B dan R secara horizontal
    b, g, r = cv2.split(out)
    max_shift = max(10, w // 12)
    shift_b = random.randint(-max_shift, max_shift)
    shift_r = random.randint(-max_shift, max_shift)
    b = np.roll(b, shift_b, axis=1)
    r = np.roll(r, shift_r, axis=1)
    out = cv2.merge([b, g, r])

    # 2. Geser beberapa strip horizontal secara acak
    num_strips = max(12, h // 16)
    strip_h = max(1, h // num_strips)
    for i in range(0, h, strip_h):
        if random.random() < 0.5:
            continue
        end = min(i + strip_h, h)
        strip_shift = random.randint(-max_shift * 2, max_shift * 2)
        out[i:end] = np.roll(out[i:end], strip_shift, axis=1)

    # 3. Noise garis tipis putih/hitam
    noise_lines = max(6, h // 20)
    for _ in range(noise_lines):
        y = random.randint(0, h - 1)
        color = (255, 255, 255) if random.random() < 0.5 else (0, 0, 0)
        cv2.line(out, (0, y), (w, y), color, 1)

    return add_scanlines(out, alpha=0.15)


def fx_impact_frame(frame):
    """
    Efek IMPACT FRAME ala momen "hantaman" di manga/komik: kontras warna
    ekstrem (hitam-putih penuh ATAU merah-hitam yang sangat jenuh),
    background di-gelapkan, lalu ditambah garis-garis "speed line" yang
    memancar dari tengah layar supaya terasa ada ledakan tenaga.
    Efek ini diterapkan ke SELURUH frame (bukan cuma area jari) karena
    tujuannya bikin seluruh adegan terasa menggetarkan.
    """
    h, w = frame.shape[:2]

    # 1. Kontras ekstrem: threshold grayscale jadi biner (hitam/putih)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, IMPACT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 2. Pilih skema warna ekstrem secara acak: hitam-putih atau merah-hitam
    scheme = random.choice(["bw", "red_black"])
    if scheme == "bw":
        extreme = cv2.merge([thresh, thresh, thresh])
        line_color = (255, 255, 255)
    else:
        zeros = np.zeros_like(thresh)
        extreme = cv2.merge([zeros, zeros, thresh])  # BGR -> cuma channel R
        line_color = (0, 0, 255)

    # 3. Gelapkan frame asli sebagai dasar, lalu timpa dengan versi kontras
    #    ekstrem supaya bentuk/siluet gambar aslinya masih kebaca tapi
    #    nuansanya jadi gelap & dramatis
    darkened = cv2.convertScaleAbs(frame, alpha=IMPACT_DARKEN_ALPHA, beta=0)
    out = cv2.addWeighted(extreme, 0.8, darkened, 0.2, 0)

    # 4. Speed lines memancar dari tengah layar (efek ledakan tenaga)
    center = (w // 2, h // 2)
    max_len = int(min(w, h) * 0.65)
    min_len = int(min(w, h) * 0.3)
    for _ in range(IMPACT_NUM_SPEEDLINES):
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(min_len, max_len)
        x2 = int(center[0] + length * math.cos(angle))
        y2 = int(center[1] + length * math.sin(angle))
        thickness = random.randint(1, 3)
        cv2.line(out, center, (x2, y2), line_color, thickness, cv2.LINE_AA)

    return out


def apply_screen_shake(frame, magnitude=IMPACT_SHAKE_MAGNITUDE):
    """
    Getarkan (geser acak) seluruh frame supaya momen hantaman terasa
    menggetarkan layar. Area yang kosong akibat pergeseran diisi hitam.
    """
    h, w = frame.shape[:2]
    dx = random.randint(-magnitude, magnitude)
    dy = random.randint(-magnitude, magnitude)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    shaken = cv2.warpAffine(
        frame, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return shaken


def load_static_photo(path, frame_w, frame_h):
    """
    Load foto dari disk dan resize supaya PAS sama ukuran frame kamera.
    Return None kalau file tidak ditemukan/gagal dibaca.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, (frame_w, frame_h))


def apply_photo_window(frame, polygon, static_photo):
    """
    Tampilkan potongan 'static_photo' di dalam area polygon, di posisi
    (x, y) yang SAMA dengan posisi polygon di layar. Karena foto ini
    diam di koordinat layar (bukan ikut gerak tangan), menggeser tangan
    otomatis membuka bagian foto yang berbeda -- seolah tangan jadi
    jendela yang mengintip foto statis di baliknya.
    """
    if static_photo is None:
        # Foto belum ke-load (path salah/tidak ditemukan) -> tidak
        # menampilkan apa-apa, biar tidak crash.
        return frame

    x, y, w, h = cv2.boundingRect(polygon)
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    if w <= 0 or h <= 0:
        return frame

    photo_crop = static_photo[y:y + h, x:x + w]
    if photo_crop.shape[:2] != (h, w):
        return frame

    roi = frame[y:y + h, x:x + w]

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = polygon.reshape(-1, 2) - [x, y]
    cv2.fillConvexPoly(mask, shifted, 255)
    mask3 = cv2.merge([mask, mask, mask])

    blended = np.where(mask3 == 255, photo_crop, roi)
    frame[y:y + h, x:x + w] = blended
    return frame


def apply_effect(frame, polygon, fx_func):
    """
    Terapkan fx_func hanya di dalam area 'polygon' (bentuk mengikuti
    jari), bukan di seluruh bounding box.
    """
    x, y, w, h = cv2.boundingRect(polygon)
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    if w <= 0 or h <= 0:
        return frame

    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return frame

    processed = fx_func(roi)

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = polygon.reshape(-1, 2) - [x, y]
    cv2.fillConvexPoly(mask, shifted, 255)
    mask3 = cv2.merge([mask, mask, mask])

    blended = np.where(mask3 == 255, processed, roi)
    frame[y:y + h, x:x + w] = blended
    return frame


def draw_neon_polygon(frame, polygon, color, thickness=3, glow=True):
    if glow:
        glow_layer = frame.copy()
        cv2.polylines(glow_layer, [polygon], True, color, thickness + 8, cv2.LINE_AA)
        frame = cv2.addWeighted(glow_layer, 0.25, frame, 0.75, 0)
    cv2.polylines(frame, [polygon], True, color, thickness, cv2.LINE_AA)
    return frame


# def draw_overlay(frame, fps, effect, num_hands):
#     h, w = frame.shape[:2]
#     panel = frame.copy()
#     cv2.rectangle(panel, (0, 0), (w, 90), (20, 20, 20), -1)
#     frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)

#     cv2.putText(frame, "DYNAMIC CYBERPUNK LENS", (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, NEON_MAGENTA, 2, cv2.LINE_AA)

#     status_map = {
#         "blue": ("MODE BIRU AKTIF (Jempol + Telunjuk)", NEON_BLUE),
#         "negative": ("MODE NEGATIF AKTIF (Jempol + Jari Tengah)", NEON_GREEN),
#         "glitch": ("MODE GLITCH AKTIF (Jempol + Kelingking)", NEON_RED),
#         "mirror": ("MODE MIRROR AKTIF (Telunjuk + Kelingking)", NEON_PINK),
#         "photo_window": ("MODE FOTO AKTIF (Telunjuk + Jari Tengah)", NEON_ORANGE),
#         "blur": ("MODE BLUR AKTIF (Tangan menghadap bawah)", NEON_PURPLE),
#         "impact": ("MODE IMPACT AKTIF (Kedua tangan mengepal)", (255, 255, 255)),
#     }
#     text, color = status_map.get(
#         effect, ("Gunakan 2 Tangan (Bentuk Frame)", (0, 255, 255))
#     )
#     cv2.putText(frame, text, (20, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

#     cv2.putText(frame, f"Tangan terdeteksi: {num_hands}", (w - 320, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
#     cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
#     cv2.putText(frame, "Tekan 'q' untuk keluar", (20, h - 15),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
#     return frame


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka webcam. Cek CAM_INDEX / koneksi kamera.")
        return

    prev_time = time.time()

    # Foto statis untuk mode "photo window". Di-resize belakangan begitu
    # ukuran frame kamera yang sebenarnya diketahui (bisa beda dari
    # FRAME_W/FRAME_H kalau webcam tidak mendukung resolusi itu).
    photo_raw = cv2.imread(PHOTO_PATH)
    if photo_raw is None:
        print(f"[WARNING] Foto '{PHOTO_PATH}' tidak ditemukan. "
              f"Mode photo window tidak akan menampilkan apa-apa "
              f"sampai PHOTO_PATH diperbaiki.")
    static_photo = None  # akan diisi (di-resize) begitu frame pertama datang

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Gagal membaca frame dari webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if photo_raw is not None and (
            static_photo is None
            or static_photo.shape[0] != h
            or static_photo.shape[1] != w
        ):
            static_photo = cv2.resize(photo_raw, (w, h))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        hand_list = results.multi_hand_landmarks or []
        num_hands = len(hand_list)

        # for hl in hand_list:
        #     mp_draw.draw_landmarks(
        #         frame, hl, mp_hands.HAND_CONNECTIONS,
        #         mp_draw_styles.get_default_hand_landmarks_style(),
        #         mp_draw_styles.get_default_hand_connections_style(),
        #     )

        effect = None

        if num_hands == 2:
            # 0. Cek dulu apakah KEDUA tangan mengepal penuh (BATU / fist).
            #    Ini diprioritaskan paling atas di antara semua gestur,
            #    karena kepalan penuh tidak overlap dengan gestur lain
            #    (semua gestur lain butuh minimal satu jari terbuka).
            fists = [is_fist(hl) for hl in hand_list]

            if all(fists):
                effect = "impact"
                frame = fx_impact_frame(frame)
                frame = apply_screen_shake(frame)
            else:
                # 1. Tentukan jari aktif per tangan (index / middle / pinky)
                finger_choices = [get_finger_choice(hl) for hl in hand_list]
                tip_ids = [FINGER_TIP_BY_NAME[fc] for fc in finger_choices]

                # 2. Cek rotasi "menghadap bawah" HANYA relevan untuk gestur
                #    jempol + telunjuk (index). Kalau salah satu tangan yang
                #    sedang ber-gestur index menghadap ke bawah -> BLUR.
                facing_down = any(
                    fc == "index" and is_thumb_index_facing_down(hl)
                    for hl, fc in zip(hand_list, finger_choices)
                )

                # 3. Cek pose TELUNJUK + KELINGKING (trigger MIRROR). Kalau
                #    salah satu tangan membentuk pose ini -> MIRROR, dan ini
                #    diprioritaskan di atas gestur biasa (blue/negative/glitch).
                mirror_sign = any(is_index_pinky_sign(hl) for hl in hand_list)

                # 3b. Cek pose TELUNJUK + JARI TENGAH (trigger PHOTO WINDOW).
                photo_sign = any(is_index_middle_sign(hl) for hl in hand_list)

                if mirror_sign and not facing_down:
                    mirror_points = []
                    for hl in hand_list:
                        for idx in (INDEX_TIP, PINKY_TIP):
                            lm = hl.landmark[idx]
                            mirror_points.append((int(lm.x * w), int(lm.y * h)))
                    polygon = compute_polygon(mirror_points, w, h)
                    if polygon is not None:
                        effect = "mirror"
                        frame = apply_effect(frame, polygon, fx_mirror)
                        frame = draw_neon_polygon(frame, polygon, NEON_PINK)
                elif photo_sign and not facing_down:
                    photo_points = []
                    for hl in hand_list:
                        for idx in (INDEX_TIP, MIDDLE_TIP):
                            lm = hl.landmark[idx]
                            photo_points.append((int(lm.x * w), int(lm.y * h)))
                    polygon = compute_polygon(photo_points, w, h)
                    if polygon is not None:
                        effect = "photo_window"
                        frame = apply_photo_window(frame, polygon, static_photo)
                        frame = draw_neon_polygon(frame, polygon, NEON_ORANGE)
                else:
                    # 4. Bangun polygon dari titik jempol + jari aktif tiap
                    #    tangan (mengikuti bentuk jari, bukan kotak lurus)
                    points = get_active_points(hand_list, w, h, tip_ids)
                    polygon = compute_polygon(points, w, h)

                    if polygon is not None:
                        if facing_down:
                            effect = "blur"
                            frame = apply_effect(frame, polygon, fx_blur)
                            frame = draw_neon_polygon(frame, polygon, NEON_PURPLE)
                        else:
                            gesture = choose_gesture(finger_choices)
                            if gesture == "middle":
                                effect = "negative"
                                frame = apply_effect(frame, polygon, fx_negative)
                                frame = draw_neon_polygon(frame, polygon, NEON_GREEN)
                            elif gesture == "pinky":
                                effect = "glitch"
                                frame = apply_effect(frame, polygon, fx_glitch)
                                frame = draw_neon_polygon(frame, polygon, NEON_RED)
                            else:
                                effect = "blue"
                                frame = apply_effect(frame, polygon, fx_blue)
                                frame = draw_neon_polygon(frame, polygon, NEON_BLUE)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
        prev_time = curr_time

        # frame = draw_overlay(frame, fps, effect, num_hands)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()