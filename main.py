# Import kelas pengendali permainan utama
from game_controller import FaceFilterGame


def main() -> None:
    """Entry point untuk menjalankan permainan Face Filter Blink Challenge."""
    # Inisialisasi permainan sehingga semua resource siap digunakan
    game = FaceFilterGame()
    # Memulai loop utama yang mengelola alur permainan dan interaksi pengguna
    game.run()


if __name__ == "__main__":
    # Pastikan kode hanya berjalan saat file ini dieksekusi langsung, bukan diimpor
    main()
