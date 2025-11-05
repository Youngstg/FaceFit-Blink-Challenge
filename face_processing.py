from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from constants import LEFT_EYE_INDICES, RIGHT_EYE_INDICES


def calculate_eye_aspect_ratio(landmarks: List[List[float]], eye_indices: Iterable[int]) -> float:
    """Hitung Eye Aspect Ratio (EAR) untuk satu mata.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    Input landmarks dalam format relatif [x, y] (nilai 0..1) sesuai MediaPipe.
    """
    # Pastikan kita bekerja dengan list indeks agar bisa diindeks
    indices = list(eye_indices)
    # Jarak vertikal atas-bawah (dua pasang)
    v1 = np.linalg.norm(np.array(landmarks[indices[1]]) - np.array(landmarks[indices[5]]))
    v2 = np.linalg.norm(np.array(landmarks[indices[2]]) - np.array(landmarks[indices[4]]))
    # Jarak horizontal kiri-kanan (basis normalisasi)
    h = np.linalg.norm(np.array(landmarks[indices[0]]) - np.array(landmarks[indices[3]]))
    # Mengembalikan EAR sebagai float
    return (v1 + v2) / (2.0 * h)


def calculate_average_ear(landmarks: List[List[float]]) -> float:
    """Hitung EAR rata-rata dari kedua mata.

    Memanggil calculate_eye_aspect_ratio untuk mata kiri dan kanan lalu mengambil rata-rata.
    """
    left_ear = calculate_eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
    return (left_ear + right_ear) / 2.0


def apply_face_mask(
    frame: np.ndarray,
    landmarks: List[List[float]],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Tutup area sensitif wajah dengan warna kulit hasil estimasi.

    - Mengambil warna kulit dari titik pipi (landmark 234) jika berada dalam frame.
    - Menutup (fillPoly) beberapa region wajah untuk menyamarkan/melindungi area sensitif.
    - Mengembalikan frame yang telah dimodifikasi.
    """
    # Ambil koordinat pipi (digunakan untuk estimasi warna kulit)
    cheek_point = landmarks[234]
    cheek_x = int(cheek_point[0] * frame_width)
    cheek_y = int(cheek_point[1] * frame_height)

    # Jika titik pipi valid di dalam frame, gunakan warna pixel tersebut, kalau tidak fallback warna
    if 0 <= cheek_y < frame_height and 0 <= cheek_x < frame_width:
        skin_color = frame[cheek_y, cheek_x].tolist()
    else:
        # Warna kulit default (BGR)
        skin_color = [180, 150, 130]

    # Daftar region landmark yang akan diisi dengan warna kulit
    masked_areas = [
        [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7],
        [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249],
        [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 326, 327, 358, 279, 420, 429],
        [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],
        [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    ]

    for area_indices in masked_areas:
        # Konversi landmark relatif ke koordinat pixel
        points = np.array(
            [
                (int(landmarks[i][0] * frame_width), int(landmarks[i][1] * frame_height))
                for i in area_indices
            ]
        )
        # Isi polygon dengan warna kulit (BGR)
        cv2.fillPoly(frame, [points], skin_color)

    return frame


def crop_face_part(
    frame: np.ndarray,
    landmarks: List[List[float]],
    frame_width: int,
    frame_height: int,
    part_type: str,
) -> Optional[np.ndarray]:
    """Potong (crop) area wajah berdasarkan tipe bagian (left_eye, right_eye, dll).

    - Mengembalikan crop image (copy) dari frame atau None jika part_type tidak dikenali.
    - Menggunakan boundingRect + sedikit padding agar area tidak terlalu ketat.
    """
    # Peta nama bagian ke daftar indeks landmark yang sesuai
    indices_map: Dict[str, List[int]] = {
        "left_eye": [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7],
        "right_eye": [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249],
        "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
        # Beberapa indeks hidung diduplikasi di daftar asli; tetap digunakan untuk cakupan area
        "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 326, 327, 358, 279, 420, 429, 358, 327, 326],
        "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],
    }

    indices = indices_map.get(part_type)
    if not indices:
        # Part type tidak valid
        return None

    # Hitung koordinat titik polygon dalam pixel
    points = np.array(
        [
            (int(landmarks[i][0] * frame_width), int(landmarks[i][1] * frame_height))
            for i in indices
        ]
    )
    # Dapatkan bounding rectangle (x, y, w, h)
    x, y, width, height = cv2.boundingRect(points)

    # Tambahkan padding kecil agar crop tidak terlalu pas
    padding = 3
    x = max(0, x - padding)
    y = max(0, y - padding)
    width = min(frame_width - x, width + padding * 2)
    height = min(frame_height - y, height + padding * 2)

    # Kembalikan salinan area crop untuk menghindari referensi ke frame asli
    return frame[y : y + height, x : x + width].copy()


def get_nose_position(landmarks: List[List[float]], frame_width: int, frame_height: int) -> Tuple[int, int]:
    """Ambil koordinat hidung (nose tip) sebagai titik referensi.

    Mengembalikan koordinat piksel (x, y).
    """
    nose_tip = landmarks[4]
    return int(nose_tip[0] * frame_width), int(nose_tip[1] * frame_height)
