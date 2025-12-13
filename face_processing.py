import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# Import koneksi bibir resmi dari MediaPipe
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

# Import konstanta index mata dari file constants
from constants import LEFT_EYE_INDICES, RIGHT_EYE_INDICES


# ---------- KONSTAN & UTIL ----------

# Titik-titik yang membentuk lingkaran dalam bibir
# Digunakan untuk membuat "lubang" di tengah bibir agar terlihat lebih realistis
INNER_LIP_RING = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

# Ambil semua index unik dari koneksi bibir MediaPipe untuk bibir luar
LIPS_IDX = sorted({i for e in FACEMESH_LIPS for i in e})

# Map anchor points (titik jangkar) untuk setiap bagian wajah
# Dua titik ini digunakan untuk menghitung jarak dan sudut agar bagian wajah
# bisa mengikuti pergerakan, rotasi, dan skala wajah dengan akurat
ANCHOR_MAP: Dict[str, Tuple[int, int]] = {
    "left_eye": (33, 133),          # sudut dalam dan luar mata kiri
    "right_eye": (362, 263),        # sudut dalam dan luar mata kanan
    "left_eyebrow": (46, 52),       # ujung-ujung alis kiri
    "right_eyebrow": (282, 295),    # ujung-ujung alis kanan
    "nose": (98, 327),              # sayap hidung kiri dan kanan
    "mouth": (78, 308),             # sudut bibir kiri dan kanan
}


@dataclass
class FacePartData:
    """
    Data container untuk menyimpan informasi bagian wajah yang di-crop.
    Berisi gambar, posisi, dan data anchor untuk tracking.
    """
    image: np.ndarray                           # Gambar bagian wajah (BGR atau BGRA)
    center: Tuple[int, int]                     # Posisi tengah bagian wajah (x, y)
    anchor_indices: Tuple[int, int]             # Pasangan index landmark sebagai anchor
    base_anchor_mid: Tuple[float, float]        # Titik tengah anchor saat referensi
    base_anchor_dist: float                     # Jarak anchor saat referensi (untuk skala)
    base_anchor_angle: float                    # Sudut anchor saat referensi (untuk rotasi)
    offset_from_anchor: Tuple[float, float]     # Offset center dari anchor (untuk posisi)


def _px_points(landmarks: List[List[float]], indices: Iterable[int], W: int, H: int) -> np.ndarray:
    """
    Konversi landmark yang normalized (0-1) menjadi koordinat piksel.
    
    Args:
        landmarks: List landmark wajah dari MediaPipe (nilai 0-1)
        indices: Index landmark yang ingin diambil
        W: Lebar frame
        H: Tinggi frame
    
    Returns:
        Array koordinat piksel (int32)
    """
    pts = [
        (int(landmarks[i][0] * W), int(landmarks[i][1] * H))
        for i in indices
    ]
    return np.array(pts, dtype=np.int32)


def _compute_anchor_data(
    part_type: str,
    landmarks: List[List[float]],
    frame_width: int,
    frame_height: int,
    center: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[float, float], float, float, Tuple[float, float]]:
    """
    Hitung data jangkar (anchor) untuk bagian wajah tertentu.
    
    Data anchor digunakan untuk tracking bagian wajah agar bisa:
    - Mengikuti pergerakan wajah
    - Menyesuaikan skala (zoom in/out)
    - Mengikuti rotasi kepala
    
    Args:
        part_type: Jenis bagian wajah (left_eye, right_eye, dll)
        landmarks: Landmark wajah dari MediaPipe
        frame_width: Lebar frame
        frame_height: Tinggi frame
        center: Posisi center bagian wajah
    
    Returns:
        Tuple berisi:
        - anchor_indices: Pasangan index landmark
        - mid: Titik tengah anchor
        - dist: Jarak antara dua anchor points
        - angle: Sudut anchor (dalam radian)
        - offset: Offset center dari mid anchor
    """
    # Ambil pasangan anchor untuk bagian wajah ini
    anchor_indices = ANCHOR_MAP.get(part_type)
    if not anchor_indices:
        # Jika tidak ada anchor, return default values
        return (), (float(center[0]), float(center[1])), 1.0, 0.0, (0.0, 0.0)

    # Konversi anchor indices ke koordinat piksel
    pts = _px_points(landmarks, anchor_indices, frame_width, frame_height)
    if len(pts) < 2:
        return (), (float(center[0]), float(center[1])), 1.0, 0.0, (0.0, 0.0)

    # Ambil dua titik anchor
    p1, p2 = pts[0], pts[1]
    
    # Hitung vektor dari p1 ke p2
    vec = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
    
    # Hitung jarak antara dua anchor points (untuk skala)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-3:
        dist = 1.0
    
    # Hitung sudut vektor anchor (untuk rotasi)
    angle = float(math.atan2(vec[1], vec[0]))
    
    # Hitung titik tengah anchor
    mid = (float((p1[0] + p2[0]) * 0.5), float((p1[1] + p2[1]) * 0.5))
    
    # Hitung offset center dari mid anchor (untuk posisi relatif)
    offset = (float(center[0] - mid[0]), float(center[1] - mid[1]))
    
    return anchor_indices, mid, dist, angle, offset


def compute_aligned_center(
    part_data: FacePartData,
    landmarks: List[List[float]],
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[Tuple[int, int], float, float, Tuple[float, float]]]:
    """
    Hitung posisi baru bagian wajah berdasarkan perubahan anchor.
    
    Fungsi ini membandingkan anchor saat ini dengan anchor referensi (base),
    kemudian menghitung transformasi (posisi, skala, rotasi) yang diperlukan.
    
    Args:
        part_data: Data bagian wajah dengan anchor referensi
        landmarks: Landmark wajah terkini
        frame_width: Lebar frame
        frame_height: Tinggi frame
    
    Returns:
        Tuple berisi:
        - center: Posisi baru bagian wajah (x, y)
        - scale: Faktor skala (ratio jarak anchor)
        - angle_diff: Perubahan sudut (radian)
        - mid: Titik tengah anchor saat ini
    """
    # Validasi data anchor
    if not part_data.anchor_indices or part_data.base_anchor_dist <= 0.0:
        return None

    # Konversi anchor indices ke koordinat piksel
    pts = _px_points(landmarks, part_data.anchor_indices, frame_width, frame_height)
    if len(pts) < 2:
        return None

    # Ambil dua titik anchor
    p1, p2 = pts[0], pts[1]
    
    # Hitung vektor dan properti anchor saat ini
    vec = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-3:
        return None
    angle = float(math.atan2(vec[1], vec[0]))
    mid = np.array([(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5], dtype=np.float32)

    # Hitung skala: ratio jarak anchor sekarang vs referensi
    scale = dist / part_data.base_anchor_dist
    
    # Hitung perubahan sudut: sudut sekarang - sudut referensi
    angle_diff = angle - part_data.base_anchor_angle

    # Buat matrix rotasi untuk memutar offset
    cos_a = math.cos(angle_diff)
    sin_a = math.sin(angle_diff)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    
    # Aplikasikan rotasi dan skala pada offset
    offset_vec = np.array(part_data.offset_from_anchor, dtype=np.float32)
    new_offset = rot @ (offset_vec * scale)
    
    # Hitung posisi center baru
    new_center = mid + new_offset
    
    return (int(new_center[0]), int(new_center[1])), scale, angle_diff, (float(mid[0]), float(mid[1]))


def reanchor_part_data(
    part_data: FacePartData,
    landmarks: List[List[float]],
    frame_width: int,
    frame_height: int,
    target_center: Tuple[int, int],
) -> None:
    """
    Set ulang anchor referensi ke pose wajah saat ini.
    
    Digunakan ketika kita ingin "mengunci" posisi relatif bagian wajah
    terhadap anchor di pose wajah yang baru.
    
    Args:
        part_data: Data bagian wajah yang akan di-reanchor
        landmarks: Landmark wajah terkini
        frame_width: Lebar frame
        frame_height: Tinggi frame
        target_center: Posisi center yang diinginkan
    """
    if not part_data.anchor_indices:
        return
    
    # Konversi anchor indices ke koordinat piksel
    pts = _px_points(landmarks, part_data.anchor_indices, frame_width, frame_height)
    if len(pts) < 2:
        return

    # Hitung properti anchor dari pose saat ini
    p1, p2 = pts[0], pts[1]
    vec = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-3:
        return
    angle = float(math.atan2(vec[1], vec[0]))
    mid = (float((p1[0] + p2[0]) * 0.5), float((p1[1] + p2[1]) * 0.5))

    # Update anchor referensi dengan nilai saat ini
    part_data.base_anchor_mid = mid
    part_data.base_anchor_dist = dist
    part_data.base_anchor_angle = angle
    
    # Hitung offset baru dari target_center
    part_data.offset_from_anchor = (
        float(target_center[0] - mid[0]),
        float(target_center[1] - mid[1]),
    )


def _sample_skin_color(
    frame: np.ndarray,
    landmarks: List[List[float]],
    frame_width: int,
    frame_height: int,
) -> List[int]:
    """
    Ambil sampel warna kulit dari beberapa titik wajah.
    
    Mengambil patch kecil dari pipi kanan/kiri, dahi, dagu, dan rahang,
    kemudian menghitung median warna BGR untuk hasil yang robust
    terhadap variasi tone kulit dan kondisi pencahayaan.
    
    Args:
        frame: Frame video
        landmarks: Landmark wajah
        frame_width: Lebar frame
        frame_height: Tinggi frame
    
    Returns:
        List [B, G, R] warna kulit median
    """
    # Index landmark untuk sampling: pipi kanan, pipi kiri, dahi, dagu, rahang
    sample_indices = [234, 454, 10, 152, 50, 280]
    patches: List[np.ndarray] = []
    radius = 3  # Ukuran patch 7x7 pixel per titik

    # Ambil patch dari setiap titik sampling
    for idx in sample_indices:
        lx, ly = landmarks[idx]
        cx, cy = int(lx * frame_width), int(ly * frame_height)
        
        # Tentukan batas patch
        x1, x2 = max(0, cx - radius), min(frame_width, cx + radius + 1)
        y1, y2 = max(0, cy - radius), min(frame_height, cy + radius + 1)
        
        if x2 > x1 and y2 > y1:
            patch = frame[y1:y2, x1:x2]
            if patch.size > 0:
                # Flatten patch jadi list pixel
                patches.append(patch.reshape(-1, 3))

    # Hitung median dari semua pixel yang diambil
    if patches:
        all_pixels = np.concatenate(patches, axis=0)
        median_color = np.median(all_pixels, axis=0)
        return [int(c) for c in median_color.tolist()]

    # Fallback warna kulit default jika sampling gagal
    return [180, 150, 130]  # BGR


# ---------- EAR (Eye Aspect Ratio) ----------

def calculate_eye_aspect_ratio(landmarks: List[List[float]], eye_indices: Iterable[int]) -> float:
    """
    Hitung Eye Aspect Ratio (EAR) untuk deteksi mata tertutup/terbuka.
    
    Formula EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    - p2-p6, p3-p5: jarak vertikal mata (atas-bawah)
    - p1-p4: jarak horizontal mata (kiri-kanan)
    
    EAR tinggi = mata terbuka lebar
    EAR rendah = mata tertutup/berkedip
    
    Args:
        landmarks: Landmark wajah (normalized 0-1)
        eye_indices: Urutan 6 titik mata [p1,p2,p3,p4,p5,p6]
    
    Returns:
        Nilai EAR (Eye Aspect Ratio)
    """
    idx = list(eye_indices)
    # Ambil 6 titik mata
    p = [np.array(landmarks[i], dtype=np.float32) for i in idx]

    # Hitung jarak vertikal (atas-bawah mata)
    v1 = np.linalg.norm(p[1] - p[5])  # Vertikal 1
    v2 = np.linalg.norm(p[2] - p[4])  # Vertikal 2
    
    # Hitung jarak horizontal (kiri-kanan mata)
    h = np.linalg.norm(p[0] - p[3])

    if h == 0.0:
        return 0.0
    
    # Hitung EAR: rata-rata vertikal dibagi horizontal
    return (v1 + v2) / (2.0 * h)


def calculate_average_ear(landmarks: List[List[float]]) -> float:
    """
    Hitung rata-rata EAR dari kedua mata.
    Lebih reliable untuk deteksi berkedip daripada satu mata saja.
    
    Args:
        landmarks: Landmark wajah
    
    Returns:
        Rata-rata EAR dari mata kiri dan kanan
    """
    left_ear = calculate_eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
    return (left_ear + right_ear) / 2.0


# ---------- MASK WAJAH ----------

def apply_face_mask(
    frame: np.ndarray,
    landmarks: List[List[float]],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Tutup area sensitif wajah dengan warna kulit estimasi.
    
    Digunakan untuk menutupi bagian-bagian wajah (mata, hidung, mulut, alis)
    dengan warna kulit sehingga hanya bentuk wajah yang tersisa.
    Berguna untuk persiapan sebelum overlay bagian wajah yang baru.
    
    Args:
        frame: Frame video yang akan di-mask
        landmarks: Landmark wajah
        frame_width: Lebar frame
        frame_height: Tinggi frame
    
    Returns:
        Frame yang sudah di-mask
    """
    # Ambil warna kulit dari beberapa titik wajah
    skin_color = _sample_skin_color(frame, landmarks, frame_width, frame_height)

    # Daftar area yang akan ditutup dengan index landmark
    masked_areas = [
        # Mata kiri (lengkap dengan kelopak)
        [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7, 246, 161,
         130, 25, 110, 24, 23, 22, 26, 112, 243],
        # Mata kanan (lengkap dengan kelopak)
        [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249, 466, 388,
         359, 255, 339, 254, 253, 252, 256, 341, 463],
        # Hidung (batang + sayap + lubang)
        [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 327, 326, 358, 279,
         420, 429, 399, 412, 351, 419, 248, 281, 275, 49, 131, 134, 51, 363, 360],
        # Alis kiri
        [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        # Alis kanan
        [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    ]

    # Tutup setiap area dengan convex hull
    for area_indices in masked_areas:
        pts = _px_points(landmarks, area_indices, frame_width, frame_height)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(frame, [hull], skin_color)

    # Khusus untuk mulut: outer hull dikurangi inner ring
    # Agar bagian dalam bibir tetap hitam/kosong
    mouth_outer_pts = _px_points(landmarks, LIPS_IDX, frame_width, frame_height)
    if len(mouth_outer_pts) >= 3:
        outer_hull = cv2.convexHull(mouth_outer_pts)
        
        # Buat mask lokal untuk operasi subtract
        x, y, w, h = cv2.boundingRect(outer_hull)
        x2, y2 = x + w, y + h
        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        w, h = x2 - x, y2 - y
        
        if w > 0 and h > 0:
            # Buat mask local
            local_mask = np.zeros((h, w), np.uint8)
            
            # Fill outer hull (bibir luar)
            outer_local = outer_hull.reshape(-1, 2) - np.array([[x, y]])
            cv2.fillPoly(local_mask, [outer_local.astype(np.int32)], 255)

            # Subtract inner hull (bibir dalam) - buat lubang di tengah
            inner_pts = _px_points(landmarks, INNER_LIP_RING, frame_width, frame_height)
            if len(inner_pts) >= 3:
                inner_hull = cv2.convexHull(inner_pts)
                inner_local = inner_hull.reshape(-1, 2) - np.array([[x, y]])
                cv2.fillPoly(local_mask, [inner_local.astype(np.int32)], 0)

            # Sedikit blur untuk transisi halus
            local_mask = cv2.GaussianBlur(local_mask, (9, 9), 2)

            # Apply warna kulit menggunakan mask
            roi = frame[y:y+h, x:x+w]
            fill = np.full_like(roi, skin_color, dtype=np.uint8)
            roi[:] = np.where(local_mask[..., None] > 0, fill, roi)

    return frame


# ---------- CROP BAGIAN WAJAH ----------

def crop_face_part(
    frame: np.ndarray,
    landmarks: List[List[float]],
    frame_width: int,
    frame_height: int,
    part_type: str,
    mask_style: str = "polygon",
    with_alpha: bool = False,
) -> Optional[FacePartData]:
    """
    Crop bagian wajah tertentu dari frame.
    
    Fungsi ini akan:
    1. Mengambil area bagian wajah (mata, hidung, mulut, alis)
    2. Membuat mask sesuai bentuk (polygon/ellipse/rectangle)
    3. Menghitung data anchor untuk tracking
    4. Mengembalikan FacePartData lengkap
    
    Args:
        frame: Frame video sumber
        landmarks: Landmark wajah dari MediaPipe
        frame_width: Lebar frame
        frame_height: Tinggi frame
        part_type: Jenis bagian ("left_eye", "right_eye", "nose", "mouth", dll)
        mask_style: Style mask ("polygon", "ellipse", "rectangle")
        with_alpha: True jika ingin output BGRA dengan alpha channel
    
    Returns:
        FacePartData berisi gambar crop dan data tracking, atau None jika gagal
    """
    # Map index landmark untuk setiap bagian wajah
    indices_map: Dict[str, List[int]] = {
        # Mata kiri (lengkap)
        "left_eye": [
            33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7, 246, 161,
            130, 25, 110, 24, 23, 22, 26, 112, 243
        ],
        # Mata kanan (lengkap)
        "right_eye": [
            362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249, 466, 388,
            359, 255, 339, 254, 253, 252, 256, 341, 463
        ],
        # Alis kiri
        "left_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        # Alis kanan
        "right_eyebrow": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
        # Hidung (area lebar)
        "nose": [
            168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 327, 326, 358, 279,
            420, 429, 399, 412, 351, 419, 248, 281, 275, 49, 131, 134, 51, 363, 360
        ],
        # Mulut (pakai outer lips dari FACEMESH_LIPS)
        "mouth": LIPS_IDX,
    }

    # Ambil index untuk part type yang diminta
    indices = indices_map.get(part_type)
    if not indices:
        return None

    # Konversi landmark ke koordinat piksel
    points = _px_points(landmarks, indices, frame_width, frame_height)

    # Buat convex hull untuk bentuk yang lebih stabil
    if len(points) < 3:
        return None
    hull = cv2.convexHull(points)
    hull_pts = hull.reshape(-1, 2)

    # Hitung bounding box dengan padding
    x, y, w, h = cv2.boundingRect(hull_pts)
    padding = 25
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame_width - x, w + 2 * padding)
    h = min(frame_height - y, h + 2 * padding)
    if w <= 0 or h <= 0:
        return None

    # Crop ROI (Region of Interest)
    roi = frame[y:y+h, x:x+w]
    
    # Buat mask kosong
    mask = np.zeros((h, w), dtype=np.uint8)

    # Konversi koordinat hull ke koordinat lokal ROI
    roi_hull = (hull_pts - np.array([[x, y]])).astype(np.int32)

    # Tentukan mask style
    ms = (mask_style or "polygon").lower()
    if ms not in {"rectangle", "ellipse", "polygon"}:
        ms = "polygon"

    # Buat mask sesuai style
    if ms == "rectangle":
        # Rectangle: mask penuh
        mask[:] = 255
    elif ms == "polygon":
        # Polygon: ikuti bentuk hull
        cv2.fillPoly(mask, [roi_hull], 255)
    elif ms == "ellipse":
        # Ellipse: bentuk oval/lingkaran
        if len(roi_hull) >= 5:
            try:
                # Fit ellipse dari points
                ellipse = cv2.fitEllipse(roi_hull.astype(np.float32))
                cv2.ellipse(mask, ellipse, 255, -1)
            except Exception:
                # Fallback: ellipse dari center
                cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
        else:
            cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)

    # Khusus mulut: subtract inner ring untuk efek bibir ring
    if part_type == "mouth":
        inner_pts = _px_points(landmarks, INNER_LIP_RING, frame_width, frame_height)
        if len(inner_pts) >= 3:
            inner_hull = cv2.convexHull(inner_pts).reshape(-1, 2)
            inner_roi = (inner_hull - np.array([[x, y]])).astype(np.int32)
            # Buat "lubang" di tengah bibir
            cv2.fillPoly(mask, [inner_roi], 0)

    # Feathering (blur) untuk tepi yang halus
    if ms != "rectangle":
        mask = cv2.GaussianBlur(mask, (9, 9), 2)

    # Apply mask ke ROI
    if with_alpha:
        # Output dengan alpha channel (BGRA)
        bgr = roi if roi.ndim == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask  # Set alpha dari mask
        cropped = bgra
    else:
        # Output BGR biasa dengan mask
        cropped = cv2.bitwise_and(roi, roi, mask=mask)

    # Hitung center dari crop
    center = (x + w // 2, y + h // 2)
    
    # Hitung data anchor untuk tracking
    anchor_indices, base_mid, base_dist, base_angle, offset = _compute_anchor_data(
        part_type, landmarks, frame_width, frame_height, center
    )
    
    # Return FacePartData lengkap
    return FacePartData(
        image=cropped,
        center=center,
        anchor_indices=anchor_indices,
        base_anchor_mid=base_mid,
        base_anchor_dist=base_dist,
        base_anchor_angle=base_angle,
        offset_from_anchor=offset,
    )


def get_nose_position(landmarks: List[List[float]], frame_width: int, frame_height: int) -> Tuple[int, int]:
    """
    Dapatkan koordinat ujung hidung (nose tip).
    
    Args:
        landmarks: Landmark wajah dari MediaPipe
        frame_width: Lebar frame
        frame_height: Tinggi frame
    
    Returns:
        Tuple (x, y) koordinat ujung hidung dalam piksel
    """
    # Landmark index 4 adalah ujung hidung
    nose_tip = landmarks[4]
    return int(nose_tip[0] * frame_width), int(nose_tip[1] * frame_height)
