from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# Pakai koneksi bibir resmi dari MediaPipe
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

# Tetap pakai constants kamu (isi index mata sesuai proyekmu)
from constants import LEFT_EYE_INDICES, RIGHT_EYE_INDICES


# ---------- KONSTAN & UTIL ----------
# Inner-lip ring yang stabil untuk “melobangi” bagian dalam kalau dibutuhkan
INNER_LIP_RING = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

# Outer lips: ambil semua indeks unik dari FACEMESH_LIPS
LIPS_IDX = sorted({i for e in FACEMESH_LIPS for i in e})


@dataclass
class FacePartData:
    image: np.ndarray
    center: Tuple[int, int]


def _px_points(landmarks: List[List[float]], indices: Iterable[int], W: int, H: int) -> np.ndarray:
    """Konversi indeks landmark -> koordinat piksel (int32)."""
    pts = [
        (int(landmarks[i][0] * W), int(landmarks[i][1] * H))
        for i in indices
    ]
    return np.array(pts, dtype=np.int32)


# ---------- EAR ----------
def calculate_eye_aspect_ratio(landmarks: List[List[float]], eye_indices: Iterable[int]) -> float:
    """
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    landmarks: [x,y] normalized (0..1)
    eye_indices: urutan [p1,p2,p3,p4,p5,p6]
    """
    idx = list(eye_indices)
    p = [np.array(landmarks[i], dtype=np.float32) for i in idx]

    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])

    if h == 0.0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def calculate_average_ear(landmarks: List[List[float]]) -> float:
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
    Tutup area sensitif dengan warna kulit estimasi dari pipi (LM 234).
    Mulut pakai convex hull dari FACEMESH_LIPS (outer) + subtract inner ring.
    """
    # Ambil warna kulit dari pipi
    cheek = landmarks[234]
    cx, cy = int(cheek[0] * frame_width), int(cheek[1] * frame_height)
    if 0 <= cy < frame_height and 0 <= cx < frame_width:
        skin_color = frame[cy, cx].tolist()
    else:
        skin_color = [180, 150, 130]  # fallback BGR

    # Area lain (tetap seperti punyamu)
    masked_areas = [
        # Mata kiri
        [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7, 246, 161],
        # Mata kanan
        [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249, 466, 388],
        # Hidung
        [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 327, 326, 358, 279,
         420, 429, 399, 412, 351, 419, 248, 281, 275],
        # Alis kiri
        [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        # Alis kanan
        [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    ]

    # Tutup area-areas di atas
    for area_indices in masked_areas:
        pts = _px_points(landmarks, area_indices, frame_width, frame_height)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(frame, [hull], skin_color)

    # Mulut: outer hull (LIPS_IDX) – inner hull (INNER_LIP_RING)
    mouth_outer_pts = _px_points(landmarks, LIPS_IDX, frame_width, frame_height)
    if len(mouth_outer_pts) >= 3:
        outer_hull = cv2.convexHull(mouth_outer_pts)
        # Buat mask lokal biar bisa subtract inner
        x, y, w, h = cv2.boundingRect(outer_hull)
        x2, y2 = x + w, y + h
        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        w, h = x2 - x, y2 - y
        if w > 0 and h > 0:
            local_mask = np.zeros((h, w), np.uint8)
            outer_local = outer_hull.reshape(-1, 2) - np.array([[x, y]])
            cv2.fillPoly(local_mask, [outer_local.astype(np.int32)], 255)

            inner_pts = _px_points(landmarks, INNER_LIP_RING, frame_width, frame_height)
            if len(inner_pts) >= 3:
                inner_hull = cv2.convexHull(inner_pts)
                inner_local = inner_hull.reshape(-1, 2) - np.array([[x, y]])
                cv2.fillPoly(local_mask, [inner_local.astype(np.int32)], 0)

            # Sedikit feather biar halus
            local_mask = cv2.GaussianBlur(local_mask, (9, 9), 2)

            # Fill pakai warna kulit
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
    Crop area wajah per bagian:
      - mouth: pakai hull dari FACEMESH_LIPS (cover atas+bawah). (opsional subtract inner).
      - lainnya: pakai daftar indeks di bawah + convex hull.
    """
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
        # Hidung (lebar)
        "nose": [
            168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 327, 326, 358, 279,
            420, 429, 399, 412, 351, 419, 248, 281, 275, 49, 131, 134, 51, 363, 360
        ],
        # Mulut (pakai outer lips dari FACEMESH_LIPS)
        "mouth": LIPS_IDX,
    }

    indices = indices_map.get(part_type)
    if not indices:
        return None

    # Build polygon points
    points = _px_points(landmarks, indices, frame_width, frame_height)

    # Convex hull untuk stabilitas bentuk
    if len(points) < 3:
        return None
    hull = cv2.convexHull(points)
    hull_pts = hull.reshape(-1, 2)

    # Bounding box + padding
    x, y, w, h = cv2.boundingRect(hull_pts)
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame_width - x, w + 2 * padding)
    h = min(frame_height - y, h + 2 * padding)
    if w <= 0 or h <= 0:
        return None

    roi = frame[y:y+h, x:x+w]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Koordinat lokal ROI
    roi_hull = (hull_pts - np.array([[x, y]])).astype(np.int32)

    # Mask style
    ms = (mask_style or "polygon").lower()
    if ms not in {"rectangle", "ellipse", "polygon"}:
        ms = "polygon"

    if ms == "rectangle":
        mask[:] = 255
    elif ms == "polygon":
        cv2.fillPoly(mask, [roi_hull], 255)
    elif ms == "ellipse":
        if len(roi_hull) >= 5:
            try:
                ellipse = cv2.fitEllipse(roi_hull.astype(np.float32))
                cv2.ellipse(mask, ellipse, 255, -1)
            except Exception:
                cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
        else:
            cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)

    # Khusus mouth: subtract inner ring biar “ring bibir” (opsional, aktifkan kalau perlu)
    if part_type == "mouth":
        inner_pts = _px_points(landmarks, INNER_LIP_RING, frame_width, frame_height)
        if len(inner_pts) >= 3:
            inner_hull = cv2.convexHull(inner_pts).reshape(-1, 2)
            inner_roi = (inner_hull - np.array([[x, y]])).astype(np.int32)
            cv2.fillPoly(mask, [inner_roi], 0)

    # Feathering supaya tepi halus
    if ms != "rectangle":
        mask = cv2.GaussianBlur(mask, (9, 9), 2)

    # Apply mask ke ROI
    if with_alpha:
        bgr = roi if roi.ndim == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        cropped = bgra
    else:
        cropped = cv2.bitwise_and(roi, roi, mask=mask)

    center = (x + w // 2, y + h // 2)
    return FacePartData(image=cropped, center=center)

def get_nose_position(landmarks: List[List[float]], frame_width: int, frame_height: int) -> Tuple[int, int]:
    """Koordinat hidung (nose tip) LM 4."""
    nose_tip = landmarks[4]
    return int(nose_tip[0] * frame_width), int(nose_tip[1] * frame_height)
