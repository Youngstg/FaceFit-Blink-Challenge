from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from constants import (
    CAPTURE_COUNTDOWN,
    EAR_THRESHOLD,
    FACE_PART_SEQUENCE,
    STATE_CAPTURE,
    STATE_PLAYING,
)
from falling_face_part import FallingFacePart
from face_processing import apply_face_mask, calculate_average_ear, crop_face_part, get_nose_position


class FaceFilterGame:
    """Main loop pengelolaan permainan Face Filter Blink Challenge."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index  # Index kamera yang digunakan (default 0 = webcam)
        self._face_mesh_solution = mp.solutions.face_mesh  # Inisialisasi MediaPipe Face Mesh
        self._part_sequence: Sequence[str] = FACE_PART_SEQUENCE  # Urutan bagian wajah yang akan dijatuhkan
        self.reset_state()  # Set state awal game

    def run(self) -> None:
        """Fungsi utama untuk menjalankan game."""
        # Buka kamera
        capture = cv2.VideoCapture(self.camera_index)

        # Inisialisasi face mesh dengan konfigurasi
        with self._face_mesh_solution.FaceMesh(
            min_detection_confidence=0.5,  # Minimum confidence untuk deteksi wajah
            min_tracking_confidence=0.5,   # Minimum confidence untuk tracking wajah
            max_num_faces=1,                # Hanya deteksi 1 wajah
        ) as face_mesh:
            # Loop utama game
            while capture.isOpened():
                frame_available, frame = capture.read()  # Baca frame dari kamera
                if not frame_available:
                    break

                # Flip frame agar seperti cermin
                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape
                
                # Konversi ke RGB untuk MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)  # Proses deteksi wajah

                # Proses berdasarkan state saat ini
                if self.current_state == STATE_CAPTURE:
                    # State untuk mengcapture bagian wajah
                    self._process_capture_state(frame, results, frame_width, frame_height)
                else:
                    # State untuk bermain (menjatuhkan dan menangkap bagian wajah)
                    self._process_play_state(frame, results, frame_width, frame_height)

                # Tampilkan frame
                cv2.imshow("Face Filter - Crop & Drop", frame)
                key = cv2.waitKey(1) & 0xFF

                # Handle keyboard input
                if key == ord("q"):  # Tekan 'q' untuk keluar
                    break
                if key == ord("r"):  # Tekan 'r' untuk reset game
                    self.reset_state()

        # Bersihkan resources
        capture.release()
        cv2.destroyAllWindows()

    def reset_state(self) -> None:
        """Reset semua state game ke kondisi awal."""
        self.current_state = STATE_CAPTURE  # Mulai dari state capture
        self.face_parts: Dict[str, np.ndarray] = {}  # Dictionary untuk menyimpan gambar bagian wajah
        self.falling_objects: List[FallingFacePart] = []  # List objek yang sedang jatuh
        self.placed_objects: List[FallingFacePart] = []  # List objek yang sudah ditangkap/ditempel
        self.current_part_index = 0  # Index bagian wajah yang sedang aktif
        self.capture_countdown = CAPTURE_COUNTDOWN  # Countdown sebelum capture
        self.last_countdown_time = cv2.getTickCount()  # Waktu terakhir countdown

    def _process_capture_state(
        self,
        frame: np.ndarray,
        results: Any,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Proses state capture: countdown dan ambil foto bagian wajah."""
        if results and results.multi_face_landmarks:
            # Hitung waktu elapsed untuk countdown
            current_time = cv2.getTickCount()
            elapsed = (current_time - self.last_countdown_time) / cv2.getTickFrequency()

            # Update countdown setiap 1 detik
            if elapsed >= 1.0:
                self.capture_countdown -= 1
                self.last_countdown_time = current_time

            # Jika countdown belum selesai, tampilkan prompt
            if self.capture_countdown > 0:
                self._draw_capture_prompt(frame, frame_width, frame_height, self.capture_countdown)
                return

            # Countdown selesai, capture bagian wajah
            self.face_parts.clear()
            frame_for_capture = frame.copy()
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [[lm.x, lm.y] for lm in face_landmarks.landmark]
                self._capture_face_parts(frame_for_capture, landmarks, frame_width, frame_height)

            # Pindah ke state playing
            self.current_state = STATE_PLAYING
            self._spawn_next_falling_object(frame_width, frame_height)
        else:
            # Tidak ada wajah terdeteksi
            cv2.putText(
                frame,
                "Tidak ada wajah terdeteksi!",
                (frame_width // 2 - 200, frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    def _process_play_state(
        self,
        frame: np.ndarray,
        results: Any,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Proses state playing: deteksi kedipan, update objek jatuh, dan tracking wajah."""
        blink_detected = False  # Flag untuk deteksi kedipan
        current_nose_pos: Optional[Tuple[int, int]] = None  # Posisi hidung saat ini

        # Jika wajah terdeteksi
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [[lm.x, lm.y] for lm in face_landmarks.landmark]
                
                # Terapkan mask wajah (blur area di luar wajah)
                apply_face_mask(frame, landmarks, frame_width, frame_height)
                # Dapatkan posisi hidung untuk tracking
                current_nose_pos = get_nose_position(landmarks, frame_width, frame_height)

                # Hitung Eye Aspect Ratio untuk deteksi kedipan
                avg_ear = calculate_average_ear(landmarks)
                if avg_ear < EAR_THRESHOLD:  # Jika mata tertutup (kedip)
                    blink_detected = True
                    cv2.putText(
                        frame,
                        "BLINK!",
                        (frame_width - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

        # Update posisi semua objek yang sudah ditempel agar mengikuti wajah
        if current_nose_pos:
            for obj in self.placed_objects:
                obj.update_position_from_tracking(current_nose_pos)

        # Dapatkan objek yang sedang jatuh saat ini
        current_falling_obj = self._get_current_falling_object()
        if current_falling_obj:
            current_falling_obj.update()  # Update posisi (jatuh ke bawah)

            # Jika kedipan terdeteksi, hentikan objek dan tempel ke wajah
            if blink_detected and current_nose_pos:
                current_falling_obj.stop(current_falling_obj.x, current_falling_obj.y, current_nose_pos)
                self.placed_objects.append(current_falling_obj)  # Pindahkan ke list placed
                self.current_part_index += 1  # Lanjut ke bagian wajah berikutnya
                self._spawn_next_falling_object(frame_width, frame_height)  # Spawn objek baru

            # Jika objek jatuh keluar layar, reset posisinya ke atas
            if current_falling_obj.y > frame_height + 100:
                current_falling_obj.reset_start_position(frame_width)

        # Gambar semua objek (yang sudah ditempel dan yang sedang jatuh)
        for obj in self.placed_objects:
            obj.draw(frame)
        if current_falling_obj:
            current_falling_obj.draw(frame)

        # Gambar HUD (instruksi dan progress)
        self._draw_game_hud(frame, frame_width, frame_height)

    def _capture_face_parts(
        self,
        frame: np.ndarray,
        landmarks: List[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Capture semua bagian wajah sesuai urutan yang ditentukan."""
        for part_type in self._part_sequence:
            # Crop setiap bagian wajah dan simpan ke dictionary
            cropped = crop_face_part(frame, landmarks, frame_width, frame_height, part_type)
            if cropped is not None:
                self.face_parts[part_type] = cropped

    def _spawn_next_falling_object(self, frame_width: int, frame_height: int) -> None:
        """Spawn objek jatuh berikutnya dari bagian wajah yang sudah dicapture."""
        # Jika sudah semua bagian, tidak spawn lagi
        if self.current_part_index >= len(self._part_sequence):
            return

        # Ambil jenis bagian wajah berikutnya
        part_type = self._part_sequence[self.current_part_index]
        part_image = self.face_parts.get(part_type)
        if part_image is None:
            return

        # Buat objek FallingFacePart baru dan tambahkan ke list
        self.falling_objects.append(FallingFacePart(part_image, part_type, frame_width, frame_height))

    def _get_current_falling_object(self) -> Optional[FallingFacePart]:
        """Dapatkan objek yang sedang dalam status jatuh (belum ditangkap)."""
        for obj in self.falling_objects:
            if obj.is_falling:
                return obj
        return None

    @staticmethod
    def _draw_capture_prompt(
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        remaining_seconds: int,
    ) -> None:
        """Tampilkan countdown dan instruksi sebelum capture."""
        cv2.putText(
            frame,
            f"Capture dalam {remaining_seconds}...",
            (frame_width // 2 - 150, frame_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            "Tunjukkan wajah Anda!",
            (frame_width // 2 - 150, frame_height // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    def _draw_game_hud(self, frame: np.ndarray, frame_width: int, frame_height: int) -> None:
        """Tampilkan HUD game (instruksi dan progress)."""
        # Instruksi cara bermain
        cv2.putText(
            frame,
            "Kedipkan mata untuk menangkap!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        # Progress bagian wajah yang sudah ditangkap
        cv2.putText(
            frame,
            f"Bagian: {self.current_part_index}/{len(self._part_sequence)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Jika sudah selesai semua, tampilkan pesan selesai
        if self.current_part_index >= len(self._part_sequence):
            cv2.putText(
                frame,
                "SELESAI! Tekan 'R' untuk reset",
                (frame_width // 2 - 200, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )