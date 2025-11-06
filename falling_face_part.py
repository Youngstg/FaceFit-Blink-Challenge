import random
from typing import Optional, Tuple

import cv2
import numpy as np


class FallingFacePart:
    """Representasi komponen wajah yang jatuh dan dapat ditempelkan kembali."""

    def __init__(self, image: np.ndarray, part_type: str, frame_width: int, frame_height: int) -> None:
        # Simpan gambar bagian wajah (misalnya mata, hidung, mulut)
        self.image = image
        self.part_type = part_type  # Jenis bagian wajah
        
        # Posisi awal X secara acak (tidak terlalu di pinggir)
        self.x = random.randint(100, frame_width - 100)
        # Posisi awal Y di atas layar (negatif, sehingga belum terlihat)
        self.y = -image.shape[0] if image is not None else -50
        
        # Kecepatan jatuh acak
        self.speed = random.randint(4, 8)
        self.is_falling = True  # Status apakah masih jatuh atau sudah menempel
        
        # Simpan ukuran gambar
        self.width = image.shape[1] if image is not None else 50
        self.height = image.shape[0] if image is not None else 50
        
        # Offset dari hidung (digunakan untuk tracking setelah menempel)
        self.offset_from_nose: Optional[Tuple[int, int]] = None

    def update(self) -> None:
        """Update posisi jika masih dalam status jatuh."""
        if self.is_falling:
            self.y += self.speed  # Gerakkan ke bawah

    def reset_start_position(self, frame_width: int) -> None:
        """Reset posisi awal ketika objek kembali dijatuhkan."""
        self.y = -self.height  # Kembalikan ke atas layar
        self.x = random.randint(100, frame_width - 100)  # Posisi X baru secara acak
        self.is_falling = True

    def stop(self, stop_x: int, stop_y: int, nose_pos: Tuple[int, int]) -> None:
        """Hentikan pergerakan dan simpan offset relatif terhadap hidung."""
        self.is_falling = False  # Berhenti jatuh
        self.x = stop_x  # Simpan posisi akhir
        self.y = stop_y
        # Hitung jarak dari hidung agar bisa follow pergerakan wajah
        self.offset_from_nose = (stop_x - nose_pos[0], stop_y - nose_pos[1])

    def update_position_from_tracking(self, nose_pos: Tuple[int, int]) -> None:
        """Sinkronkan posisi dengan pergerakan kepala."""
        if self.offset_from_nose is None:
            return
        # Update posisi berdasarkan posisi hidung saat ini + offset yang tersimpan
        self.x = nose_pos[0] + self.offset_from_nose[0]
        self.y = nose_pos[1] + self.offset_from_nose[1]

    def draw(self, frame: np.ndarray) -> None:
        """Gambar bagian wajah ke frame video."""
        if self.image is None:
            return

        # Hitung posisi untuk menggambar (tengah gambar di koordinat x,y)
        x = int(self.x - self.width // 2)
        # Jika masih jatuh, gunakan posisi y langsung. Jika sudah menempel, tengahkan
        y = int(self.y - self.height // 2) if not self.is_falling else int(self.y)

        # Cek apakah posisi masih dalam batas frame (tidak keluar layar)
        if y < 0 or x < 0:
            return
        if y + self.height > frame.shape[0] or x + self.width > frame.shape[1]:
            return

        # Ambil region of interest (area di frame tempat gambar akan ditempel)
        roi = frame[y : y + self.height, x : x + self.width]
        # Cek apakah ukuran ROI sesuai dengan ukuran gambar
        if roi.shape[:2] != self.image.shape[:2]:
            return

        # Tempelkan gambar dengan blending (10% frame asli + 90% gambar bagian wajah)
        frame[y : y + self.height, x : x + self.width] = cv2.addWeighted(
            roi,
            0.1,  # Bobot frame asli (10%)
            self.image,
            0.9,  # Bobot gambar bagian wajah (90%)
            0,  # Gamma correction
        )