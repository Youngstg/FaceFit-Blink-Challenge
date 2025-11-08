import random
from typing import Optional, Tuple

import cv2
import numpy as np


class FallingFacePart:
    """
    Class untuk membuat bagian wajah yang jatuh dari atas (seperti game).
    Bagian wajah ini bisa ditempelkan kembali ke wajah dan mengikuti pergerakan kepala.
    """

    def __init__(
        self,
        image: np.ndarray,
        part_type: str,
        frame_width: int,
        frame_height: int,
        spawn_x: Optional[int] = None,
    ) -> None:
        # === DATA GAMBAR ===
        self.image = image  # Gambar bagian wajah (mata/hidung/mulut)
        self.part_type = part_type  # Nama bagian: "mata", "hidung", "mulut"
        
        # === UKURAN GAMBAR ===
        self.width = image.shape[1] if image is not None else 50  # Lebar gambar
        self.height = image.shape[0] if image is not None else 50  # Tinggi gambar
        
        # === POSISI SPAWN (MUNCULNYA) ===
        # Tentukan batas kiri-kanan agar tidak terlalu pinggir
        min_x = max(50, (image.shape[1] // 2) if image is not None else 50)
        max_x = max(min_x + 1, frame_width - min_x)
        
        # Pilih posisi X secara acak atau gunakan posisi yang ditentukan
        random_x = random.randint(min_x, max_x)
        self.spawn_x = int(np.clip(spawn_x, min_x, max_x)) if spawn_x is not None else random_x
        
        # === POSISI AWAL ===
        self.x = self.spawn_x  # Posisi horizontal (kiri-kanan)
        self.y = -self.height  # Posisi vertikal: mulai di atas layar (negatif = belum kelihatan)
        
        # === KECEPATAN JATUH ===
        self.speed = random.randint(4, 8)  # Kecepatan jatuh acak antara 4-8 pixel per frame
        
        # === STATUS ===
        self.is_falling = True  # True = masih jatuh, False = sudah nempel di wajah
        
        # === TRACKING WAJAH ===
        # Simpan jarak dari hidung setelah nempel (biar bisa ikut gerak kepala)
        self.offset_from_nose: Optional[Tuple[int, int]] = None

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
        self.y = -self.height  # Kembali ke atas layar
        self.x = self.spawn_x  # Kembali ke kolom awal
        self.is_falling = True  # Aktifkan status jatuh

    def stop(self, stop_x: int, stop_y: int, nose_pos: Tuple[int, int]) -> None:
        """
        Hentikan jatuhnya bagian wajah dan tempelkan di posisi tertentu.
        
        Args:
            stop_x: Posisi X akhir (tempat nempel)
            stop_y: Posisi Y akhir (tempat nempel)
            nose_pos: Posisi hidung saat ini (x, y) - sebagai titik referensi
        """
        self.is_falling = False  # Matikan status jatuh
        self.x = stop_x  # Simpan posisi nempel
        self.y = stop_y
        
        # Hitung dan simpan jarak dari hidung
        # Jadi nanti kalau kepala gerak, bagian wajah ikut gerak juga
        self.offset_from_nose = (stop_x - nose_pos[0], stop_y - nose_pos[1])

    def update_position_from_tracking(self, nose_pos: Tuple[int, int]) -> None:
        """
        Update posisi bagian wajah mengikuti pergerakan kepala.
        Digunakan setelah bagian wajah sudah nempel.
        
        Args:
            nose_pos: Posisi hidung saat ini (x, y)
        """
        if self.offset_from_nose is None:
            return  # Belum nempel, skip
        
        # Posisi baru = posisi hidung sekarang + jarak yang tersimpan
        self.x = nose_pos[0] + self.offset_from_nose[0]
        self.y = nose_pos[1] + self.offset_from_nose[1]

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
        # Ambil area di frame tempat gambar akan ditempel (Region of Interest)
        roi = frame[y : y + self.height, x : x + self.width]
        
        # Cek apakah ukuran ROI sama dengan ukuran gambar
        if roi.shape[:2] != self.image.shape[:2]:
            return  # Ukuran tidak cocok, skip

        # Tempelkan gambar dengan blending (campuran):
        # - 10% dari frame asli (biar transparan sedikit)
        # - 90% dari gambar bagian wajah (biar kelihatan jelas)
        frame[y : y + self.height, x : x + self.width] = cv2.addWeighted(
            roi,           # Frame asli
            0.1,           # 10% frame asli
            self.image,    # Gambar bagian wajah
            0.9,           # 90% gambar bagian wajah
            0,             # Gamma correction (tidak dipakai)
        )
