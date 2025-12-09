import math
import random
import cv2
import numpy as np
from typing import Tuple, List, Optional

from face_processing import FacePartData, compute_aligned_center, reanchor_part_data


class FallingFacePart:
    """Class untuk objek bagian wajah yang jatuh."""
    
    def _init_(
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
        self.part_data = part_data
        self.part_type = part_type
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fall_speed = fall_speed
        
        # Posisi awal
        if spawn_x is None:
            spawn_x = part_data.center[0]
        self.start_x = spawn_x
        self.x = float(spawn_x)
        self.y = -50.0  # Mulai dari atas layar
        
        # Status
        self.is_falling = True
        self.current_scale = 1.0
        
        # Simpan gambar original untuk scaling
        self.original_image = part_data.image.copy() if part_data.image is not None else None
        
    def update(self) -> None:
        """Update posisi objek (jatuh ke bawah, X tetap)."""
        if self.is_falling:
            self.y += self.fall_speed
            # Pastikan X tetap sama dengan posisi awal
            self.x = float(self.start_x)
    
    def reset_start_position(self) -> None:
        """Reset posisi ke atas layar."""
        self.x = float(self.start_x)
        self.y = -50.0
        self.is_falling = True
    
    def stop(self, target_center: Tuple[int, int]) -> None:
        """
        Hentikan objek jatuh dan set posisi final.
        
        Args:
            target_center: Posisi target (x, y) untuk ditempel
        """
        self.x = float(target_center[0])
        self.y = float(target_center[1])
        self.is_falling = False
        # Update center di part_data
        self.part_data.center = target_center
    
    def apply_scale(self, scale: float) -> None:
        """
        Terapkan skala pada gambar.
        
        Args:
            scale: Faktor skala (1.0 = ukuran asli)
        """
        if self.original_image is None or scale <= 0:
            return
        
        # Cek perubahan skala signifikan (> 5%)
        if abs(scale - self.current_scale) < 0.05:
            return
        
        self.current_scale = scale
        
        # Resize gambar
        new_h = int(self.original_image.shape[0] * scale)
        new_w = int(self.original_image.shape[1] * scale)
        
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
        Update posisi objek yang sudah ditempel mengikuti wajah.
        
        Args:
            landmarks: Landmark wajah terkini
            frame_width: Lebar frame
            frame_height: Tinggi frame
        """
        if self.is_falling:
            return
        
        # Hitung posisi baru berdasarkan anchor
        result = compute_aligned_center(
            self.part_data,
            landmarks,
            frame_width,
            frame_height,
        )
        
        if result:
            new_center, scale, _, _ = result
            self.x = float(new_center[0])
            self.y = float(new_center[1])
            self.part_data.center = (int(new_center[0]), int(new_center[1]))
            
            # Update skala
            self.apply_scale(scale)
    
    def reanchor_to_current_landmarks(
        self,
        landmarks: List[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """
        Set ulang anchor reference ke pose wajah saat ini.
        
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
        Gambar objek ke frame.
        
        Args:
            frame: Frame target untuk digambar
        """
        if self.part_data.image is None:
            return
        
        img = self.part_data.image
        h, w = img.shape[:2]
        
        # Hitung posisi top-left
        x1 = int(self.x - w // 2)
        y1 = int(self.y - h // 2)
        x2 = x1 + w
        y2 = y1 + h
        
        # Pastikan dalam batas frame
        frame_h, frame_w = frame.shape[:2]
        if x2 <= 0 or y2 <= 0 or x1 >= frame_w or y1 >= frame_h:
            return
        
        # Crop jika sebagian keluar frame
        img_x1 = max(0, -x1)
        img_y1 = max(0, -y1)
        img_x2 = w - max(0, x2 - frame_w)
        img_y2 = h - max(0, y2 - frame_h)
        
        frame_x1 = max(0, x1)
        frame_y1 = max(0, y1)
        frame_x2 = min(frame_w, x2)
        frame_y2 = min(frame_h, y2)
        
        if img_x2 <= img_x1 or img_y2 <= img_y1:
            return
        
        img_crop = img[img_y1:img_y2, img_x1:img_x2]
        
        # Blend dengan alpha channel jika ada
        if img.shape[2] == 4:
            alpha = img_crop[:, :, 3:4] / 255.0
            img_rgb = img_crop[:, :, :3]
            bg = frame[frame_y1:frame_y2, frame_x1:frame_x2]
            
            blended = (img_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            frame[frame_y1:frame_y2, frame_x1:frame_x2] = blended
        else:
            frame[frame_y1:frame_y2, frame_x1:frame_x2] = img_crop
