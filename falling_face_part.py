import math
import random
import cv2
import numpy as np
from typing import Tuple, List, Optional

from face_processing import FacePartData, compute_aligned_center, reanchor_part_data


class FallingFacePart:
    """
    Class untuk objek bagian wajah yang jatuh.
    Digunakan untuk animasi bagian wajah (mata, hidung, mulut) yang jatuh dari atas
    dan menempel ke posisi yang tepat di wajah.
    """
    
    def __init__(
        self,
        part_data: FacePartData,
        part_type: str,
        screen_width: int,
        screen_height: int,
        spawn_x: Optional[int] = None,
        fall_speed: float = 10.0,
    ):
        """
        Initialize falling face part object.
        
        Args:
            part_data: Data bagian wajah (gambar, center, dll)
            part_type: Jenis bagian wajah (left_eye, right_eye, dll)
            screen_width: Lebar layar
            screen_height: Tinggi layar
            spawn_x: Posisi X spawn (default: random atau center dari part_data)
            fall_speed: Kecepatan jatuh (pixel per frame)
        """
        # Simpan data dasar
        self.part_data = part_data
        self.part_type = part_type
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fall_speed = fall_speed
        
        # Tentukan posisi awal (spawn point)
        if spawn_x is None:
            spawn_x = part_data.center[0]
        self.start_x = spawn_x  # Posisi X awal (tetap selama jatuh)
        self.x = float(spawn_x)
        self.y = -50.0  # Mulai dari atas layar (di luar viewport)
        
        # Status objek
        self.is_falling = True  # Apakah sedang jatuh atau sudah menempel
        self.current_scale = 1.0  # Skala ukuran saat ini
        
        # Simpan gambar original untuk keperluan scaling nanti
        self.original_image = part_data.image.copy() if part_data.image is not None else None
        
    def update(self) -> None:
        """
        Update posisi objek setiap frame.
        Objek jatuh vertikal ke bawah, posisi X tetap konstan.
        """
        if self.is_falling:
            # Gerakkan ke bawah sesuai kecepatan jatuh
            self.y += self.fall_speed
            # Pastikan X tetap sama dengan posisi awal (jatuh lurus)
            self.x = float(self.start_x)
    
    def reset_start_position(self) -> None:
        """
        Reset posisi objek ke posisi awal di atas layar.
        Digunakan untuk mengulang animasi jatuh.
        """
        self.x = float(self.start_x)
        self.y = -50.0  # Kembali ke atas layar
        self.is_falling = True
    
    def stop(self, target_center: Tuple[int, int]) -> None:
        """
        Hentikan objek jatuh dan tempelkan di posisi target.
        Dipanggil ketika objek mencapai posisi yang diinginkan di wajah.
        
        Args:
            target_center: Posisi target (x, y) untuk ditempel
        """
        # Set posisi final
        self.x = float(target_center[0])
        self.y = float(target_center[1])
        self.is_falling = False  # Tandai sudah tidak jatuh lagi
        # Update center di part_data agar sinkron
        self.part_data.center = target_center
    
    def apply_scale(self, scale: float) -> None:
        """
        Terapkan skala (zoom in/out) pada gambar bagian wajah.
        Digunakan untuk menyesuaikan ukuran dengan jarak wajah ke kamera.
        
        Args:
            scale: Faktor skala (1.0 = ukuran asli, >1 = lebih besar, <1 = lebih kecil)
        """
        if self.original_image is None or scale <= 0:
            return
        
        # Cek apakah perubahan skala cukup signifikan (> 5%)
        # Hindari resize terlalu sering untuk performa
        if abs(scale - self.current_scale) < 0.05:
            return
        
        self.current_scale = scale
        
        # Hitung ukuran baru
        new_h = int(self.original_image.shape[0] * scale)
        new_w = int(self.original_image.shape[1] * scale)
        
        # Resize gambar jika ukuran valid
        if new_h > 0 and new_w > 0:
            self.part_data.image = cv2.resize(
                self.original_image,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )
    
    def update_position_from_landmarks(
        self,
        landmarks: List[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """
        Update posisi objek yang sudah menempel agar mengikuti gerakan wajah.
        Dipanggil setiap frame setelah objek berhenti jatuh.
        
        Args:
            landmarks: Landmark wajah terkini dari MediaPipe
            frame_width: Lebar frame video
            frame_height: Tinggi frame video
        """
        # Skip jika masih dalam mode jatuh
        if self.is_falling:
            return
        
        # Hitung posisi baru berdasarkan landmark wajah saat ini
        result = compute_aligned_center(
            self.part_data,
            landmarks,
            frame_width,
            frame_height,
        )
        
        if result:
            new_center, scale, _, _ = result
            # Update posisi mengikuti wajah
            self.x = float(new_center[0])
            self.y = float(new_center[1])
            self.part_data.center = (int(new_center[0]), int(new_center[1]))
            
            # Update skala sesuai jarak wajah
            self.apply_scale(scale)
    
    def reanchor_to_current_landmarks(
        self,
        landmarks: List[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """
        Set ulang titik anchor reference ke pose wajah saat ini.
        Digunakan untuk "mengunci" posisi relatif terhadap wajah baru.
        
        Args:
            landmarks: Landmark wajah terkini
            frame_width: Lebar frame
            frame_height: Tinggi frame
        """
        target_center = (int(self.x), int(self.y))
        reanchor_part_data(
            self.part_data,
            landmarks,
            frame_width,
            frame_height,
            target_center,
        )
    
    def draw(self, frame: np.ndarray) -> None:
        """
        Gambar objek bagian wajah ke frame video.
        Mendukung transparansi (alpha channel) untuk blending yang smooth.
        
        Args:
            frame: Frame video target untuk digambar
        """
        if self.part_data.image is None:
            return
        
        img = self.part_data.image
        h, w = img.shape[:2]
        
        # Hitung posisi top-left dari gambar (center dikurangi setengah ukuran)
        x1 = int(self.x - w // 2)
        y1 = int(self.y - h // 2)
        x2 = x1 + w
        y2 = y1 + h
        
        # Cek apakah objek masih dalam batas frame
        frame_h, frame_w = frame.shape[:2]
        if x2 <= 0 or y2 <= 0 or x1 >= frame_w or y1 >= frame_h:
            return  # Objek di luar layar, skip drawing
        
        # Crop bagian gambar yang keluar dari batas frame
        img_x1 = max(0, -x1)  # Potong dari kiri jika x1 negatif
        img_y1 = max(0, -y1)  # Potong dari atas jika y1 negatif
        img_x2 = w - max(0, x2 - frame_w)  # Potong dari kanan jika lewat batas
        img_y2 = h - max(0, y2 - frame_h)  # Potong dari bawah jika lewat batas
        
        # Koordinat di frame
        frame_x1 = max(0, x1)
        frame_y1 = max(0, y1)
        frame_x2 = min(frame_w, x2)
        frame_y2 = min(frame_h, y2)
        
        # Validasi ukuran crop
        if img_x2 <= img_x1 or img_y2 <= img_y1:
            return
        
        # Ambil bagian gambar yang akan ditampilkan
        img_crop = img[img_y1:img_y2, img_x1:img_x2]
        
        # Blend dengan alpha channel jika ada (untuk transparansi)
        if img.shape[2] == 4:
            # Ekstrak alpha channel (0-255) dan normalisasi ke 0-1
            alpha = img_crop[:, :, 3:4] / 255.0
            # Ambil RGB channels
            img_rgb = img_crop[:, :, :3]
            # Ambil background dari frame
            bg = frame[frame_y1:frame_y2, frame_x1:frame_x2]
            
            # Blending: foreground * alpha + background * (1 - alpha)
            blended = (img_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            frame[frame_y1:frame_y2, frame_x1:frame_x2] = blended
        else:
            # Jika tidak ada alpha channel, langsung copy
            frame[frame_y1:frame_y2, frame_x1:frame_x2] = img_crop
