# === IMPORT LIBRARY ===
import cv2  # OpenCV untuk akses kamera
from game_controller import FaceFilterGame  # Class utama game yang sudah dibuat


def main() -> None:
    """
    Fungsi utama untuk menjalankan permainan Face Filter Blink Challenge.
    
    Fungsi ini bertanggung jawab untuk:
    1. Mencari kamera yang tersedia
    2. Menginisialisasi game
    3. Menjalankan game
    """
    
    # === KONFIGURASI KAMERA ===
    preferred_index = 1  # Index 1 = kamera eksternal (webcam USB biasanya)
    fallback_index = 0   # Index 0 = kamera default (kamera laptop biasanya)
    selected_index = preferred_index  # Mulai dengan mencoba kamera eksternal
    
    # === CEK KETERSEDIAAN KAMERA ===
    # Coba buka kamera eksternal dulu (index 1)
    cap = cv2.VideoCapture(preferred_index)
    
    if not cap.isOpened():
        # Kamera eksternal tidak tersedia
        print("Kamera index 1 tidak tersedia, menggunakan index 0.")
        selected_index = fallback_index  # Ganti ke kamera default
        cap.release()  # Tutup koneksi kamera yang gagal
        
        # Coba buka kamera default (index 0)
        cap = cv2.VideoCapture(fallback_index)
        
        if not cap.isOpened():
            # Tidak ada kamera yang tersedia sama sekali
            raise RuntimeError("Tidak ada kamera yang tersedia pada index 1 maupun 0.")
    
    # Tutup koneksi kamera sementara
    # (nanti akan dibuka lagi oleh FaceFilterGame)
    cap.release()
    
    # === INISIALISASI GAME ===
    # Buat objek game dengan kamera yang sudah dipilih
    # Di sini semua resource game (detector wajah, gambar, dll) akan disiapkan
    game = FaceFilterGame(camera_index=selected_index)
    
    # === JALANKAN GAME ===
    # Mulai loop utama game yang mengelola:
    # - Capture video dari kamera
    # - Deteksi wajah dan mata
    # - Game logic (jatuh, nempel, scoring)
    # - Tampilan UI
    # - Input dari keyboard
    game.run()


# === ENTRY POINT PROGRAM ===
if __name__ == "__main__":
    """
    Blok ini memastikan fungsi main() hanya dijalankan ketika file ini
    dieksekusi langsung (bukan ketika di-import sebagai module).
    
    Contoh:
    - python main.py        → main() akan dijalankan ✓
    - import main           → main() TIDAK dijalankan ✗
    """
    main()
