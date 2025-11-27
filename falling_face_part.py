import math
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from face_processing import FacePartData, compute_aligned_center, reanchor_part_data


class FallingFacePart:
    """
    Class untuk membuat bagian wajah yang jatuh dari atas (seperti game).
    Bagian wajah ini bisa ditempelkan kembali ke wajah dan mengikuti pergerakan kepala.
    """

    def __init__(
        self,
        part_data: FacePartData,
        part_type: str,
        frame_width: int,
        frame_height: int,
        spawn_x: Optional[int] = None,
    ) -> None:
        self.part_data = part_data
        # === DATA GAMBAR ===
        self._original_image = part_data.image
        self.image = part_data.image  # Gambar bagian wajah (mata/hidung/mulut)
        self.part_type = part_type  # Nama bagian: "mata", "hidung", "mulut"
        
        # === UKURAN GAMBAR ===
        self.width = self.image.shape[1] if self.image is not None else 50  # Lebar gambar
        self.height = self.image.shape[0] if self.image is not None else 50  # Tinggi gambar
        self._base_size = (self.width, self.height)
        self._current_scale = 1.0
        self._angle_smoothed = 0.0
        self._lock_frames = 0  # tahan update agar tidak lompat tepat setelah stop
        
        # === POSISI SPAWN (MUNCULNYA) ===
        # Tentukan batas kiri-kanan agar tidak terlalu pinggir
        min_x = max(50, (self.image.shape[1] // 2) if self.image is not None else 50)
        max_x = max(min_x + 1, frame_width - min_x)
        
        # Pilih posisi X secara acak atau gunakan posisi yang ditentukan
        random_x = random.randint(min_x, max_x)
        self.spawn_x = float(np.clip(spawn_x, min_x, max_x)) if spawn_x is not None else float(random_x)
        
        # === POSISI AWAL ===
        self.x: float = self.spawn_x  # Posisi horizontal (kiri-kanan)
        self.y: float = -float(self.height)  # Posisi vertikal: mulai di atas layar (negatif = belum kelihatan)
        
        # === KECEPATAN JATUH ===
        self.speed = random.randint(4, 8)  # Kecepatan jatuh acak antara 4-8 pixel per frame
        
        # === STATUS ===
        self.is_falling = True  # True = masih jatuh, False = sudah nempel di wajah

    def update(self) -> None:
        """
        Update posisi setiap frame.
        Kalau masih jatuh, gerakkan ke bawah.
        """
        if self.is_falling:
            self.y += self.speed  # Tambah posisi Y = gerak ke bawah

    def reset_start_position(self) -> None:
        """
        Kembalikan ke posisi awal (untuk dijatuhkan lagi).
        Digunakan ketika bagian wajah ini ingin dijatuhkan ulang.
        """
        # Kembalikan ukuran ke dasar saat dijatuhkan ulang
        if self._original_image is not None:
            self.image = self._original_image
            self.width, self.height = self._base_size
            self._current_scale = 1.0
        self.y = -self.height  # Kembali ke atas layar
        self.x = self.spawn_x  # Kembali ke kolom awal
        self.is_falling = True  # Aktifkan status jatuh

    def apply_scale(self, scale: float) -> None:
        """Skalakan gambar sesuai perubahan jarak anchor."""
        if self._original_image is None:
            return
        # Hindari skala ekstrem
        scale = max(0.5, min(2.0, float(scale)))
        self._current_scale = scale
        new_w = max(1, int(self._base_size[0] * scale))
        new_h = max(1, int(self._base_size[1] * scale))
        if new_w == self.width and new_h == self.height:
            return
        self.image = cv2.resize(self._original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.width, self.height = new_w, new_h

    def stop(self, target_center: Tuple[int, int]) -> None:
        """Hentikan jatuhnya bagian wajah dan tempelkan di posisi tertentu."""
        self.is_falling = False
        self.x, self.y = target_center
        # Saat menempel, jadikan pusat sekarang sebagai referensi anchor baru
        self._lock_frames = 2  # tahan 2 frame pertama agar tidak lompat

    def reanchor_to_current_landmarks(
        self,
        landmarks: List[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Perbarui anchor agar mengikuti perubahan ukuran/orientasi wajah setelah ditempel."""
        if landmarks is None:
            return
        center = (int(self.x), int(self.y))
        reanchor_part_data(self.part_data, landmarks, frame_width, frame_height, center)
        # Reset smoothing angle ke 0 agar delta selanjutnya diukur dari referensi baru
        self._angle_smoothed = 0.0
        self._lock_frames = 0

    def update_position_from_landmarks(
        self,
        landmarks: List[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """
        Update posisi bagian wajah mengikuti pergerakan kepala berbasis dua anchor.
        Dipanggil setelah bagian wajah sudah nempel.
        """
        if self.is_falling:
            return
        if self._lock_frames > 0:
            self._lock_frames -= 1
            return
        result = compute_aligned_center(
            self.part_data, landmarks, frame_width, frame_height
        )
        if result:
            (cx, cy), scale, angle_diff, mid = result
            # Smooth scale/sudut/posisi; hidung dibuat lebih responsif (alpha lebih besar)
            if self.part_type == "nose":
                alpha_scale = 0.6
                alpha_angle = 0.6
                alpha_pos = 0.6
            else:
                alpha_scale = 0.35
                alpha_angle = 0.35
                alpha_pos = 0.35

            smoothed_scale = self._current_scale * (1 - alpha_scale) + float(scale) * alpha_scale
            self._angle_smoothed = self._angle_smoothed * (1 - alpha_angle) + float(angle_diff) * alpha_angle

            # Hitung ulang center dengan sudut yang dismoothing
            offset_vec = np.array(self.part_data.offset_from_anchor, dtype=np.float32)
            cos_a = math.cos(self._angle_smoothed)
            sin_a = math.sin(self._angle_smoothed)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            new_offset = rot @ (offset_vec * smoothed_scale)
            new_center = np.array([mid[0], mid[1]], dtype=np.float32) + new_offset

            # Smooth posisi juga
            smoothed_x = self.x * (1 - alpha_pos) + float(new_center[0]) * alpha_pos
            smoothed_y = self.y * (1 - alpha_pos) + float(new_center[1]) * alpha_pos

            self.apply_scale(smoothed_scale)
            self.x, self.y = smoothed_x, smoothed_y

    def draw(self, frame: np.ndarray) -> None:
        """
        Gambar bagian wajah ke dalam frame video.
        
        Args:
            frame: Frame video tempat gambar akan ditempel
        """
        if self.image is None:
            return  # Tidak ada gambar, skip

        # === HITUNG POSISI GAMBAR ===
        # Posisi X: tengahkan gambar di koordinat x
        x = int(self.x - self.width // 2)
        
        # Posisi Y: 
        # - Kalau masih jatuh: gunakan y langsung (dari atas)
        # - Kalau sudah nempel: tengahkan di y
        y = int(self.y - self.height // 2) if not self.is_falling else int(self.y)

        # === CEK BATAS LAYAR ===
        # Jangan gambar kalau posisinya keluar dari frame
        if y < 0 or x < 0:
            return  # Posisi di luar batas atas/kiri
        if y + self.height > frame.shape[0] or x + self.width > frame.shape[1]:
            return  # Posisi keluar batas bawah/kanan

        # === TEMPELKAN GAMBAR ===
        # Jika gambar punya alpha channel, lakukan alpha compositing manual
        if self.image.shape[2] == 4:
            overlay = self.image
            alpha = overlay[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            roi = frame[y : y + self.height, x : x + self.width]
            if roi.shape[:2] != overlay.shape[:2]:
                return
            for c in range(3):
                roi[:, :, c] = (alpha * overlay[:, :, c] + alpha_inv * roi[:, :, c]).astype(np.uint8)
            frame[y : y + self.height, x : x + self.width] = roi
        else:
            # Ambil area di frame tempat gambar akan ditempel (Region of Interest)
            roi = frame[y : y + self.height, x : x + self.width]

            # Cek apakah ukuran ROI sama dengan ukuran gambar
            if roi.shape[:2] != self.image.shape[:2]:
                return  # Ukuran tidak cocok, skip

            # Blending ringan supaya terlihat integrasi dengan frame
            frame[y : y + self.height, x : x + self.width] = cv2.addWeighted(
                roi,        # Frame asli
                0.2,        # Sedikit lebih banyak frame agar tidak terlalu mencolok
                self.image, # Gambar bagian wajah
                0.8,        # Dominan gambar bagian wajah
                0,          # Gamma correction
            )
