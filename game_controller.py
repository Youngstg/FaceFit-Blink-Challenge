from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame

from constants import (
    CAPTURE_COUNTDOWN,
    EAR_THRESHOLD,
    FACE_PART_SEQUENCE,
    STATE_CAPTURE,
    STATE_PLAYING,
    STATE_MENU,
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


class SVGImageLoader:
    """Load dan cache SVG images yang sudah diconvert ke PNG."""
    
    def __init__(self):  # UBAH dari _init_ menjadi __init__
        self._cache: Dict[str, Optional[np.ndarray]] = {}
        self.assets_dir = Path(__file__).parent / "Assets"
    
    def load_svg_as_png(self, filename: str, width: int, height: int) -> Optional[np.ndarray]:
        """Load SVG file dan convert ke PNG array untuk display di OpenCV."""
        cache_key = f"{filename}_{width}_{height}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            svg_path = self.assets_dir / filename
            if not svg_path.exists():
                print(f"[ERROR] SVG file not found: {svg_path}")
                return None
            
            # Try load as PNG first (if PNG version exists)
            png_path = svg_path.with_suffix('.png')
            if png_path.exists():
                img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Resize if needed
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                    self._cache[cache_key] = img
                    return img
            
            # Fallback: return None if no PNG available
            print(f"[WARNING] PNG not found for {filename}")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to load image {filename}: {e}")
            return None


class FaceFilterGame:
    """Main loop pengelolaan permainan Face Filter Blink Challenge."""

    def _adjust_landmarks_for_display(self, results, offset_x, offset_y):
        """Adjust landmark coordinates untuk fullscreen display dengan offset."""
        if results is None or results.multi_face_landmarks is None:
            return results
        
        # MediaPipe returns normalized coordinates (0-1)
        # Kami perlu store offset info untuk convert ke display coordinates nanti
        # Store offset di class untuk digunakan di methods lainnya
        self._display_offset = (offset_x, offset_y)
        
        return results
    
    def _normalize_to_display(self, x_norm, y_norm, frame_width, frame_height):
        """Convert normalized landmark coordinates ke display coordinates."""
        if not hasattr(self, '_display_offset'):
            self._display_offset = (0, 0)
        
        offset_x, offset_y = self._display_offset
        
        # Normalized (0-1) to pixel coordinates pada resized frame
        x_pixel = int(x_norm * frame_width)
        y_pixel = int(y_norm * frame_height)
        
        # Add offset untuk fullscreen position
        x_display = x_pixel + offset_x
        y_display = y_pixel + offset_y
        
        return x_display, y_display

    def __init__(self, camera_index: int = 0):  # UBAH dari _init_ menjadi __init__
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
        self._svg_loader = SVGImageLoader()  # Initialize SVG image loader
        self._camera_area: Tuple[int, int, int, int] = (0, 0, 950, 800)  # (x, y, width, height)
        self.reset_state()  # Set state awal game
        # Variabel kalibrasi EAR dinamis
        self._ear_samples = []  # type: List[float]
        self._ear_threshold = EAR_THRESHOLD
        self._ear_state_closed = False  # untuk hysteresis buka/tutup
        
        # Initialize audio
        self._init_audio()

    def run(self) -> None:
        """Fungsi utama untuk menjalankan game."""
        # Buka kamera
        capture = cv2.VideoCapture(self.camera_index)

        # Set camera properties untuk better performance
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)

        # Define fullscreen display size
        fullscreen_width, fullscreen_height = 1920, 1080
        
        # Define centered camera area size
        camera_width, camera_height = 950, 800
        camera_x = (fullscreen_width - camera_width) // 2
        camera_y = (fullscreen_height - camera_height) // 2
        
        # Store camera area info di class untuk digunakan di state methods
        self._camera_area = (camera_x, camera_y, camera_width, camera_height)
        self._display_offset = (camera_x, camera_y)
        self._camera_frame_dims = (camera_width, camera_height)

        # Inisialisasi face mesh dengan konfigurasi
        with self._face_mesh_solution.FaceMesh(
            min_detection_confidence=0.5,  # Minimum confidence untuk deteksi wajah
            min_tracking_confidence=0.5,   # Minimum confidence untuk tracking wajah
            max_num_faces=1,                # Hanya deteksi 1 wajah
        ) as face_mesh:
            # Loop utama game
            first_frame = True
            music_started = False  # Track apakah music sudah diplay
            
            # Play background music di awal
            self._play_background_music()
            music_started = True
            
            while capture.isOpened():
                frame_available, frame = capture.read()
                if not frame_available:
                    break

                # Flip frame agar seperti cermin
                frame = cv2.flip(frame, 1)
                
                # === BUAT BACKGROUND GRADIEN CYAN KE BIRU TUA ===
                display_frame = np.zeros((fullscreen_height, fullscreen_width, 3), dtype=np.uint8)
                
                # Warna gradien (sama seperti menu)
                color_start = np.array([255, 230, 168], dtype=np.float32)  # Cyan
                color_end = np.array([138, 58, 30], dtype=np.float32)  # Biru tua
                
                # Buat gradien vertikal
                for y in range(fullscreen_height):
                    ratio = y / fullscreen_height
                    color = color_start * (1 - ratio) + color_end * ratio
                    display_frame[y, :] = color.astype(np.uint8)
                
                # Resize camera frame ke camera area size
                resized_frame = cv2.resize(frame, (camera_width, camera_height))
                
                # Paste camera frame ke tengah fullscreen display
                display_frame[camera_y:camera_y+camera_height, camera_x:camera_x+camera_width] = resized_frame
                
                # Create working frame (camera area only) untuk processing
                working_frame = resized_frame.copy()
                
                # Proses berdasarkan state saat ini
                if self.current_state == STATE_MENU:
                    # State menu dengan tombol START - overlay di atas display_frame
                    self._process_menu_state(display_frame, fullscreen_width, fullscreen_height)
                else:
                    # Convert frame ke RGB untuk MediaPipe
                    rgb_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)  # Proses deteksi wajah

                    # Proses berdasarkan state saat ini (dengan working_frame yang hanya camera area)
                    if self.current_state == STATE_CAPTURE:
                        # State untuk mengcapture bagian wajah
                        self._process_capture_state(working_frame, results, camera_width, camera_height)
                    else:
                        # State untuk bermain (menjatuhkan dan menangkap bagian wajah)
                        self._process_play_state(working_frame, results, camera_width, camera_height)
                    
                    # Copy processed working frame kembali ke display frame
                    display_frame[camera_y:camera_y+camera_height, camera_x:camera_x+camera_width] = working_frame

                # Tampilkan frame
                if first_frame:
                    window_name = "FaceFit Blink Challenge"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.setMouseCallback(window_name, self._on_mouse_click)
                    first_frame = False
                
                cv2.imshow("FaceFit Blink Challenge", display_frame)
                
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
        self.current_state = STATE_MENU  # Mulai dari menu
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
        self._start_button_rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

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
                cv2.FONT_HERSHEY_DUPLEX,
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
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                2,
            )
        # Tampilkan ambang EAR terkini untuk debugging kecil
        cv2.putText(
            frame,
            f"EAR th={self._ear_threshold:.2f}",
            (frame_width - 170, 80),
            cv2.FONT_HERSHEY_DUPLEX,
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
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            "Tunjukkan wajah Anda!",
            (frame_width // 2 - 150, frame_height // 2 + 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Kalibrasi EAR: {ear_collected}/{ear_target}",
            (frame_width // 2 - 160, frame_height // 2 + 90),
            cv2.FONT_HERSHEY_DUPLEX,
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
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        # Progress bagian wajah yang sudah ditangkap
        cv2.putText(
            frame,
            f"Bagian: {self.current_part_index}/{len(self._part_sequence)}",
            (10, 60),
            cv2.FONT_HERSHEY_DUPLEX,
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
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

    def _process_menu_state(
        self,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Tampilkan menu dengan background cream dan dekorasi UI."""
        # === BUAT BACKGROUND GRADIEN DARI CYAN KE BIRU TUA ===
        # Warna awal (cyan/biru muda #A8E6FF ≈ RGB: 168, 230, 255) dan akhir (biru tua #1E3A8A ≈ RGB: 30, 58, 138)
        color_start = np.array([255, 230, 168], dtype=np.float32)  # Cyan (BGR)
        color_end = np.array([138, 58, 30], dtype=np.float32)  # Biru tua (BGR)
        
        # Buat gradien vertikal (atas ke bawah)
        for y in range(frame_height):
            # Hitung ratio posisi (0.0 di atas, 1.0 di bawah)
            ratio = y / frame_height
            
            # Interpolasi warna
            color = color_start * (1 - ratio) + color_end * ratio
            
            # Set seluruh baris dengan warna yang sama
            frame[y, :] = color.astype(np.uint8)
        
        # Camera area di tengah: 950x800
        camera_width, camera_height = 950, 800
        camera_x = (frame_width - camera_width) // 2
        camera_y = (frame_height - camera_height) // 2
        
        # Draw camera area border dengan rounded corners (putih dengan shadow)
        # Untuk rounded rectangle, kita bisa draw manual atau gunakan overlay
        overlay = frame.copy()
        corner_radius = 30
        
        # Draw filled rounded rectangle untuk background putih
        pts = [
            # Top edge dengan rounded corners
            (camera_x + corner_radius, camera_y),
            (camera_x + camera_width - corner_radius, camera_y),
            # Right edge dengan rounded corners
            (camera_x + camera_width, camera_y + corner_radius),
            (camera_x + camera_width, camera_y + camera_height - corner_radius),
            # Bottom edge dengan rounded corners
            (camera_x + camera_width - corner_radius, camera_y + camera_height),
            (camera_x + corner_radius, camera_y + camera_height),
            # Left edge dengan rounded corners
            (camera_x, camera_y + camera_height - corner_radius),
            (camera_x, camera_y + corner_radius),
        ]
        
        # Gambar rectangle dengan rounded corners (sederhana: pakai rectangle + circles di corners)
        cv2.rectangle(
            overlay,
            (camera_x + corner_radius, camera_y),
            (camera_x + camera_width - corner_radius, camera_y + camera_height),
            (255, 255, 255),
            -1  # Filled
        )
        cv2.rectangle(
            overlay,
            (camera_x, camera_y + corner_radius),
            (camera_x + camera_width, camera_y + camera_height - corner_radius),
            (255, 255, 255),
            -1  # Filled
        )
        
        # Draw circles di 4 corners
        cv2.circle(overlay, (camera_x + corner_radius, camera_y + corner_radius), corner_radius, (255, 255, 255), -1)
        cv2.circle(overlay, (camera_x + camera_width - corner_radius, camera_y + corner_radius), corner_radius, (255, 255, 255), -1)
        cv2.circle(overlay, (camera_x + corner_radius, camera_y + camera_height - corner_radius), corner_radius, (255, 255, 255), -1)
        cv2.circle(overlay, (camera_x + camera_width - corner_radius, camera_y + camera_height - corner_radius), corner_radius, (255, 255, 255), -1)
        
        # Blend overlay dengan transparency
        alpha = 0.95
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw border outline
        # Untuk rounded rectangle outline, gunakan polylines atau approx dengan circles
        border_color = (200, 200, 200)  # Abu-abu terang
        border_thickness = 3

        # === LOGO IMAGE ===
        logo_width = 761
        logo_height = 287
        logo_img = self._svg_loader.load_svg_as_png(
            "image-removebg-preview (2) 1.png", 
            logo_width, 
            logo_height
        )
        
        if logo_img is not None:
            # Posisi logo di tengah atas area putih
            logo_x = frame_width // 2 - logo_width // 2
            logo_y = camera_y + 40  # 40px dari top border
            
            self._paste_image_with_alpha(frame, logo_img, logo_x, logo_y)
        
        # === TEKS "Cara Bermain" - DINAIKKAN 50 PIXEL LAGI ===
        text_start_y = camera_y + 40 + logo_height - 15  # Dikurangi dari 10 menjadi -40 (naik 50px lagi)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        color_dark_blue = (99, 2, 2)  # Warna #020263 dalam BGR format
        
        # Judul "Cara Bermain"
        subtitle_text = "Cara Bermain"
        subtitle_size = 0.9
        subtitle_thickness = 2
        subtitle_width = cv2.getTextSize(subtitle_text, font, subtitle_size, subtitle_thickness)[0][0]
        subtitle_x = frame_width // 2 - subtitle_width // 2
        subtitle_y = text_start_y
        cv2.putText(
            frame,
            subtitle_text,
            (subtitle_x, subtitle_y),
            font,
            subtitle_size,
            color_dark_blue,  # Ubah warna ke #020263
            subtitle_thickness,
            cv2.LINE_AA
        )
        
        # === GAMBAR CARA KERJA - POSISI TETAP (dari subtitle lama) ===
        # Hitung posisi subtitle yang lama untuk referensi
        old_subtitle_y = camera_y + 40 + logo_height + 10
        cara_kerja_y = old_subtitle_y + 20  # 20px spacing dari subtitle lama (posisi tetap)
        
        # Load gambar cara_kerja.png (sesuaikan ukuran)
        cara_kerja_width = 219  # Sesuaikan dengan kebutuhan
        cara_kerja_height = 187   # Sesuaikan dengan kebutuhan
        cara_kerja_img = self._svg_loader.load_svg_as_png(
            "cara_kerja.png",
            cara_kerja_width,
            cara_kerja_height
        )
        
        if cara_kerja_img is not None:
            cara_kerja_x = frame_width // 2 - cara_kerja_width // 2
            self._paste_image_with_alpha(frame, cara_kerja_img, cara_kerja_x, cara_kerja_y)
            
            # Update button_y untuk di bawah gambar cara_kerja - DITURUNKAN 50 PIXEL LAGI
            button_y = cara_kerja_y + cara_kerja_height + 100  # Ditambah dari 50 menjadi 100 (turun 50px lagi)
        else:
            # Fallback jika gambar tidak ada
            button_y = subtitle_y + 270  # Juga disesuaikan (220 + 50)
        
        # Draw START button
        self._draw_svg_button_combined(frame, frame_width, button_y)

    def _draw_svg_button_combined(self, frame: np.ndarray, frame_width: int, button_y: int) -> None:
        """Draw START button dengan rounded corners dan outline style."""
        # Load START button image (280x70)
        btn_width, btn_height = 280, 70
        btn_img = self._svg_loader.load_svg_as_png("start.png", btn_width, btn_height)
        
        # Position button di tengah
        btn_x = frame_width // 2 - btn_width // 2
        btn_y = button_y
        
        # Paste button image
        if btn_img is not None:
            self._paste_image_with_alpha(frame, btn_img, btn_x, btn_y)
        
        # Store button rect untuk click detection
        self._start_button_rect = (btn_x, btn_y, btn_width, btn_height)

    def _paste_image_with_alpha(self, frame: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
        """Paste image dengan alpha channel ke frame."""
        if img is None:
            return
        
        h, w = img.shape[:2]
        has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False
        
        # Boundary check
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        
        if x1 >= x2 or y1 >= y2:
            return
        
        # Image coordinates
        img_x1 = max(0, -x)
        img_y1 = max(0, -y)
        img_x2 = w - max(0, x + w - frame.shape[1])
        img_y2 = h - max(0, y + h - frame.shape[0])
        
        if has_alpha:
            # Extract alpha channel
            alpha = img[img_y1:img_y2, img_x1:img_x2, 3].astype(float) / 255.0
            
            # Blend with frame
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    frame[y1:y2, x1:x2, c] * (1 - alpha) + 
                    img[img_y1:img_y2, img_x1:img_x2, c] * alpha
                ).astype(np.uint8)
        else:
            # No alpha, just copy
            frame[y1:y2, x1:x2] = img[img_y1:img_y2, img_x1:img_x2]

    def _init_audio(self) -> None:
        """Initialize pygame mixer untuk background music."""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            
            # Path ke audio file
            audio_path = Path(__file__).parent / "Assets" / "Cartoon Bounce.mp3"
            
            if audio_path.exists():
                pygame.mixer.music.load(str(audio_path))
                self._audio_loaded = True
                print(f"[AUDIO] Loaded: {audio_path}")
            else:
                self._audio_loaded = False
                print(f"[WARNING] Audio file not found: {audio_path}")
        except Exception as e:
            self._audio_loaded = False
            print(f"[ERROR] Failed to init audio: {e}")
    
    def _play_background_music(self) -> None:
        """Play background music on loop."""
        if self._audio_loaded:
            try:
                pygame.mixer.music.play(-1)  # -1 untuk infinite loop
                print("[AUDIO] Background music started (loop)")
            except Exception as e:
                print(f"[ERROR] Failed to play music: {e}")
    
    def _stop_background_music(self) -> None:
        """Stop background music."""
        try:
            pygame.mixer.music.stop()
            print("[AUDIO] Background music stopped")
        except Exception as e:
            print(f"[ERROR] Failed to stop music: {e}")

    def _on_mouse_click(self, event, x, y, flags, param):
        """Handle mouse click untuk deteksi klik tombol START."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Cek apakah klik ada di area tombol START
            if self._start_button_rect is not None:
                btn_x, btn_y, btn_w, btn_h = self._start_button_rect
                if btn_x <= x <= btn_x + btn_w and btn_y <= y <= btn_y + btn_h:
                    # Klik tombol START -> pindah ke STATE_CAPTURE
                    self.current_state = STATE_CAPTURE
                    print("[INFO] Game dimulai! Countdown dimulai...")

    def __del__(self):  # UBAH dari _del_ menjadi __del__
        """Cleanup resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
