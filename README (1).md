# Dynamic Cyberpunk Lens — Panduan Kustomisasi

Panduan singkat: mau ubah bagian apa, edit di baris/fungsi yang mana.

---

## 1. Setup awal

```bash
pip install "mediapipe==0.10.14" "numpy<2" "opencv-python<4.10" "opencv-contrib-python<4.10"
python dynamic_cyberpunk_lens.py
```

Tekan `q` untuk keluar.

---

## 2. Daftar gestur saat ini

| Gestur | Efek |
|---|---|
| Jempol + Telunjuk (2 tangan) | Biru |
| Jempol + Jari Tengah (2 tangan) | Negatif (invert warna) |
| Jempol + Kelingking (2 tangan) | Glitch |
| Jempol + Telunjuk, salah satu tangan menghadap bawah | Blur |

---

## 3. Ngatur ketebalan / intensitas efek

Semua "kekuatan" efek diatur di bagian **KONFIGURASI** (paling atas file):

```python
BLUR_KSIZE = 25        # makin besar = makin buram
DOWN_RATIO = 1.15       # makin besar = makin susah trigger "menghadap bawah"
MIN_SPREAD = 0.05       # jarak minimum jempol-jari, hindari noise
```

Untuk efek lain, kekuatannya diatur langsung di dalam fungsi `fx_*`:

- **Biru** → `fx_blue()`
  ```python
  g = (gray.astype(np.float32) * 0.35).astype(np.uint8)  # 0.35 = kadar hijau
  ```
  Kecilkan angka `0.35` biar birunya makin "murni"/pekat, besarkan biar makin pucat.

- **Negatif** → `fx_negative()` — cuma invert penuh (`cv2.bitwise_not`), tidak ada parameter kekuatan.

- **Glitch** → `fx_glitch()`
  ```python
  max_shift = max(2, w // 20)     # seberapa jauh pergeseran channel warna
  num_strips = max(3, h // 25)    # jumlah strip horizontal yang bisa digeser
  noise_lines = max(1, h // 40)   # jumlah garis noise
  ```
  Kecilkan pembagi (`// 20`, `// 25`, `// 40`) biar efeknya makin heboh/tebal.

- **Scanline** (garis halus di semua efek) → `add_scanlines()`
  ```python
  def add_scanlines(roi, gap=4, alpha=0.2):
  ```
  `gap` = jarak antar garis, `alpha` = seberapa gelap garisnya.

---

## 4. Ganti warna

Semua warna neon didefinisikan di atas, format **BGR** (bukan RGB):

```python
NEON_MAGENTA = (255, 0, 255)
NEON_BLUE    = (255, 120, 0)
NEON_GREEN   = (0, 255, 140)
NEON_PURPLE  = (200, 60, 160)
NEON_RED     = (60, 60, 255)
```

Warna ini dipakai untuk:
1. Border neon di sekitar area efek (dipanggil lewat `draw_neon_polygon(frame, polygon, WARNA_INI)` di `main()`)
2. Teks status mode di `status_map` dalam `draw_overlay()`

Kalau mau ganti warna border/teks untuk mode tertentu, cukup ganti konstanta di atas, atau ganti variabel yang dipakai di `main()` / `status_map`.

Warna efek **biru** dan **negatif** sendiri ditentukan di dalam `fx_blue()` / `fx_negative()` (channel BGR manual), bukan dari konstanta `NEON_*` — itu warna hasil filter gambarnya, beda dari warna border.

---

## 5. Nambah postur jari baru (misal: jempol + manis / ring finger)

Ada 4 titik yang perlu disentuh:

**a. Tambah landmark ID jarinya** (dekat `WRIST = 0`):
```python
RING_TIP, RING_PIP = 16, 14
```

**b. Daftarkan di `FINGER_TIP_BY_NAME` dan `GESTURE_PRIORITY`:**
```python
FINGER_TIP_BY_NAME = {
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "pinky": PINKY_TIP,
    "ring": RING_TIP,          # <- tambahkan
}
GESTURE_PRIORITY = ["index", "middle", "pinky", "ring"]
```

**c. Tambahkan skor keterbukaannya di `get_finger_choice()`:**
```python
scores = {
    "index": _dist(lm[INDEX_TIP], wrist) - _dist(lm[INDEX_PIP], wrist),
    "middle": _dist(lm[MIDDLE_TIP], wrist) - _dist(lm[MIDDLE_PIP], wrist),
    "pinky": _dist(lm[PINKY_TIP], wrist) - _dist(lm[PINKY_PIP], wrist),
    "ring": _dist(lm[RING_TIP], wrist) - _dist(lm[RING_PIP], wrist),   # <- tambahkan
}
```

**d. Buat fungsi efeknya sendiri** (contoh, di dekat `fx_glitch`):
```python
def fx_ring_effect(roi):
    # olah roi (gambar BGR) sesuka kamu, harus return array ukuran sama
    return roi
```

**e. Hubungkan gesture baru ke efek di `main()`**, di dalam blok `if gesture == ...`:
```python
elif gesture == "ring":
    effect = "ring_effect"
    frame = apply_effect(frame, polygon, fx_ring_effect)
    frame = draw_neon_polygon(frame, polygon, NEON_ORANGE)  # definisikan warnanya juga
```

**f. (Opsional) tambahkan teksnya di `status_map`** supaya muncul di overlay:
```python
"ring_effect": ("MODE RING AKTIF (Jempol + Jari Manis)", NEON_ORANGE),
```

---

## 6. Ubah postur yang sudah ada (misal ganti trigger blur dari "menghadap bawah" jadi "menghadap atas")

Logikanya ada di `is_thumb_index_facing_down()`:

```python
def is_thumb_index_facing_down(hand_landmarks):
    ...
    dx = tip.x - thumb.x
    dy = tip.y - thumb.y   # y makin besar = makin ke bawah

    if math.hypot(dx, dy) < MIN_SPREAD:
        return False

    return dy > 0 and abs(dy) > abs(dx) * DOWN_RATIO
```

- `dy > 0` = syarat "mengarah ke bawah". Ganti jadi `dy < 0` kalau mau trigger-nya "mengarah ke atas".
- Hapus syarat `dy > 0`/`dy < 0` sepenuhnya (sisakan `abs(dy) > abs(dx) * DOWN_RATIO`) kalau mau trigger di KEDUA arah (atas maupun bawah), persis seperti versi lama.
- Ganti target jarinya dari `INDEX_TIP` ke jari lain di dalam fungsi ini kalau mau gestur rotasinya bukan jempol+telunjuk.

Lalu di `main()`, baris ini yang nentuin gestur mana yang boleh dicek rotasinya:
```python
facing_down = any(
    fc == "index" and is_thumb_index_facing_down(hl)
    for hl, fc in zip(hand_list, finger_choices)
)
```
Ganti `fc == "index"` ke gestur lain kalau mau rotasi berlaku untuk kombinasi jari lain.

---

## 7. Ubah bentuk area efek (polygon vs kotak)

Area efek dibangun di `compute_polygon()` dari titik jempol + ujung jari aktif kedua tangan (4 titik → convex hull). Kalau mau balik ke kotak lurus seperti versi lama, ganti pemanggilan `cv2.convexHull()` jadi `cv2.boundingRect()` biasa, atau tambah lebih banyak titik landmark (misal PIP tiap jari) ke `get_active_points()` supaya bentuknya lebih detail mengikuti kontur tangan.

---

## 8. Referensi ID landmark MediaPipe Hands

```
0  = wrist (pergelangan tangan)
4  = thumb tip (ujung jempol)
6  = index PIP     8  = index tip
10 = middle PIP    12 = middle tip
14 = ring PIP      16 = ring tip
18 = pinky PIP     20 = pinky tip
```
Berguna kalau mau nambah jari lain atau pola gestur baru.
