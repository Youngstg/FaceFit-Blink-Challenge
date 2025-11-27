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
    EAR_DYNAMIC_ENABLED,
    EAR_DYNAMIC_SAMPLES,
    EAR_DYNAMIC_FACTOR,
    EAR_MIN_THRESHOLD,
    EAR_HYSTERESIS,
)
from falling_face_part import FallingFacePart
from face_processing import (
    FacePartData,
    apply_face_mask,
    calculate_average_ear,
    compute_aligned_center,
    reanchor_part_data,
    crop_face_part,
)


class FaceFilterGame:
    """Main loop pengelolaan permainan Face Filter Blink Challenge."""

    def __init__(self, camera_index: int = 0):
        """Initialize the game with camera index"""
        self.camera_index = camera_index
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self._face_mesh_solution = mp.solutions.face_mesh  # Inisialisasi MediaPipe Face Mesh
        self._part_sequence: Sequence[str] = FACE_PART_SEQUENCE  # Urutan bagian wajah yang akan dijatuhkan
        self.reset_state()  # Set state awal game
        # Variabel kalibrasi EAR dinamis
        self._ear_samples = []  # type: List[float]
        self._ear_threshold = EAR_THRESHOLD
        self._ear_state_closed = False  # untuk hysteresis buka/tutup

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
        self.face_parts: Dict[str, FacePartData] = {}  # Simpan gambar + koordinat target bagian wajah
        self.falling_objects: List[FallingFacePart] = []  # List objek yang sedang jatuh
        self.placed_objects: List[FallingFacePart] = []  # List objek yang sudah ditangkap/ditempel
        self.current_part_index = 0  # Index bagian wajah yang sedang aktif
        self.capture_countdown = CAPTURE_COUNTDOWN  # Countdown sebelum capture
        self.last_countdown_time = cv2.getTickCount()  # Waktu terakhir countdown
        self.blink_count = 0  # Hitung jumlah kedipan yang terdeteksi
        self._blink_active = False  # Flag untuk mencegah double count kedipan yang sama
        self._ear_samples = []
        self._ear_threshold = EAR_THRESHOLD
        self._ear_state_closed = False
        self._last_landmarks: Optional[List[List[float]]] = None

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

            # Selama countdown, kumpulkan sampel EAR untuk kalibrasi jika diaktifkan
            if EAR_DYNAMIC_ENABLED and results and results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [[lm.x, lm.y] for lm in face_landmarks.landmark]
                    ear_sample = calculate_average_ear(landmarks)
                    if len(self._ear_samples) < EAR_DYNAMIC_SAMPLES:
                        self._ear_samples.append(ear_sample)

            # Jika countdown belum selesai, tampilkan prompt & progress sampel
            if self.capture_countdown > 0:
                self._draw_capture_prompt(
                    frame,
                    frame_width,
                    frame_height,
                    self.capture_countdown,
                    len(self._ear_samples),
                    EAR_DYNAMIC_SAMPLES,
                )
                return

            # Kalibrasi ambang EAR setelah countdown selesai (sekali saja)
            if EAR_DYNAMIC_ENABLED and self._ear_samples:
                # Gunakan median untuk robust terhadap noise
                baseline = float(np.median(self._ear_samples))
                calibrated = max(EAR_MIN_THRESHOLD, baseline * EAR_DYNAMIC_FACTOR)
                self._ear_threshold = calibrated
                print(f"EAR baseline: {baseline:.4f} -> threshold kalibrasi: {self._ear_threshold:.4f}")

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
        blink_event = False  # Trigger satu kali ketika kedipan baru terdeteksi
        blink_display = False  # Menandakan mata sedang tertutup untuk tampilan teks

        # Jika wajah terdeteksi
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [[lm.x, lm.y] for lm in face_landmarks.landmark]
                self._last_landmarks = landmarks
                
                # Terapkan mask wajah (blur area di luar wajah)
                apply_face_mask(frame, landmarks, frame_width, frame_height)

                # Hitung Eye Aspect Ratio untuk deteksi kedipan
                avg_ear = calculate_average_ear(landmarks)
                threshold = self._ear_threshold
                # Hysteresis: dua ambang - tutup & buka
                close_level = threshold
                open_level = threshold + EAR_HYSTERESIS
                if not self._ear_state_closed:
                    # Mata dianggap menutup ketika turun di bawah close_level
                    if avg_ear < close_level:
                        self._ear_state_closed = True
                        blink_display = True
                        if not self._blink_active:
                            self._blink_active = True
                            self.blink_count += 1
                            blink_event = True
                            print(f"Blink terdeteksi (EAR={avg_ear:.3f} < {close_level:.3f}): {self.blink_count}")
                    else:
                        self._blink_active = False
                else:
                    # Mata dianggap kembali terbuka ketika EAR naik di atas open_level
                    if avg_ear > open_level:
                        self._ear_state_closed = False
                        self._blink_active = False
                    else:
                        blink_display = True
        else:
            self._blink_active = False
            self._last_landmarks = None

        if blink_display:
            cv2.putText(
                frame,
                "BLINK!",
                (frame_width - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        # Tampilkan ambang EAR terkini untuk debugging kecil
        cv2.putText(
            frame,
            f"EAR th={self._ear_threshold:.2f}",
            (frame_width - 170, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        # Update posisi semua objek yang sudah ditempel agar mengikuti wajah
        if self._last_landmarks:
            for obj in self.placed_objects:
                obj.update_position_from_landmarks(self._last_landmarks, frame_width, frame_height)

        # Dapatkan objek yang sedang jatuh saat ini
        current_falling_obj = self._get_current_falling_object()
        if current_falling_obj:
            current_falling_obj.update()  # Update posisi (jatuh ke bawah)

            # Jika ada kedipan baru, hentikan objek dan tempel tepat di posisi jatuh saat ini
            # (jangan snap ke lokasi capture awal). Skala bisa disesuaikan dari anchor, tapi pusat tetap posisi sekarang.
            if blink_event and self._last_landmarks:
                target_center = (int(current_falling_obj.x), int(current_falling_obj.y))
                scale_at_stop = 1.0
                snap_result = compute_aligned_center(
                    current_falling_obj.part_data,
                    self._last_landmarks,
                    frame_width,
                    frame_height,
                )
                if snap_result:
                    _, scale_at_stop, _, _ = snap_result
                current_falling_obj.apply_scale(scale_at_stop)
                current_falling_obj.stop(target_center)
                # Set ulang anchor referensi ke pose wajah sekarang agar tracking skala/rotasi mengikuti
                current_falling_obj.reanchor_to_current_landmarks(
                    self._last_landmarks, frame_width, frame_height
                )
                self.placed_objects.append(current_falling_obj)  # Pindahkan ke list placed
                self.current_part_index += 1  # Lanjut ke bagian wajah berikutnya
                self._spawn_next_falling_object(frame_width, frame_height)  # Spawn objek baru

            # Jika objek jatuh keluar layar, reset posisinya ke atas
            if current_falling_obj.y > frame_height + 100:
                current_falling_obj.reset_start_position()

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
            # Gunakan mask polygon + alpha agar mengikuti kontur (tidak kotak)
            part_data = crop_face_part(
                frame,
                landmarks,
                frame_width,
                frame_height,
                part_type,
                mask_style="polygon",
                with_alpha=True,
            )
            if part_data is not None:
                self.face_parts[part_type] = part_data

    def _spawn_next_falling_object(self, frame_width: int, frame_height: int) -> None:
        """Spawn objek jatuh berikutnya dari bagian wajah yang sudah dicapture."""
        # Jika sudah semua bagian, tidak spawn lagi
        if self.current_part_index >= len(self._part_sequence):
            return

        # Ambil jenis bagian wajah berikutnya
        part_type = self._part_sequence[self.current_part_index]
        part_data = self.face_parts.get(part_type)
        if part_data is None:
            return

        # Buat objek FallingFacePart baru dan tambahkan ke list
        self.falling_objects.append(
            FallingFacePart(
                part_data,
                part_type,
                frame_width,
                frame_height,
                spawn_x=part_data.center[0],
            )
        )

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
        ear_collected: int,
        ear_target: int,
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
        cv2.putText(
            frame,
            f"Kalibrasi EAR: {ear_collected}/{ear_target}",
            (frame_width // 2 - 160, frame_height // 2 + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
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

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
