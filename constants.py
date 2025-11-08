# Ambang EAR (Eye Aspect Ratio) default di bawah nilai ini dianggap sebagai mata tertutup (kedipan)
# Catatan: Jika dynamic EAR aktif, nilai ini akan ditimpa hasil kalibrasi.
EAR_THRESHOLD = 0.21

# Opsi kalibrasi ambang EAR secara dinamis berdasarkan kondisi mata terbuka pengguna.
# Ketika aktif, sistem akan mengumpulkan sampel EAR saat countdown (mata terbuka),
# lalu menetapkan ambang = median(ear_sampel) * EAR_DYNAMIC_FACTOR (dengan batas bawah MIN).
EAR_DYNAMIC_ENABLED = True
EAR_DYNAMIC_SAMPLES = 30        # jumlah sampel yang ditargetkan saat countdown
EAR_DYNAMIC_FACTOR = 0.78       # threshold = baseline * faktor (semakin besar, semakin sensitif)
EAR_MIN_THRESHOLD = 0.17        # batas bawah agar tidak terlalu kecil
EAR_HYSTERESIS = 0.02           # jarak aman untuk membuka kembali (anti flicker)

# Status aplikasi / state machine
STATE_CAPTURE = 0  # Mode persiapan / menangkap (countdown)
STATE_PLAYING = 1  # Mode bermain / aktif setelah capture

# Waktu mundur (detik) sebelum mengambil gambar / memulai aksi capture
CAPTURE_COUNTDOWN = 3

# Indeks landmark untuk mata kiri dan kanan (sesuai output MediaPipe Face Mesh)
# Urutan 6 titik biasanya dipakai untuk menghitung EAR: [p1, p2, p3, p4, p5, p6]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Urutan bagian wajah yang akan diproses / ditampilkan dalam aplikasi
FACE_PART_SEQUENCE = [
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
]
