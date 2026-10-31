# Load aspect keywords
with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
    ASPECT_KEYWORDS = json.load(f)

def detect_aspect(text: str, aspect: str) -> bool:
    """Cek apakah ada minimal 1 kata kunci aspek (case-insensitive substring match)."""
    text_lower = str(text).lower()
    keywords = ASPECT_KEYWORDS[aspect]
    return any(kw.lower() in text_lower for kw in keywords)

# Test
test = "saya makan sayur dan pergi ke dokter"
for asp in ASPECTS:
    print(f"  '{test}' → {asp}: {detect_aspect(test, asp)}")

# ── Test score_windowed (sekarang ASPECT_KEYWORDS sudah tersedia) ──
test_cases = [
    ("makanan ini enak dan bergizi",              "pola_konsumsi"),
    ("antri bpjs lama pelayanan buruk",           "akses_layanan_kesehatan"),
    ("olahraga pagi tidak menyenangkan",          "aktivitas_fisik"),
    ("vaksin ini aman dan efektif",               "pengetahuan_pengobatan"),
]
print(f"\n{'Teks':<40} {'Aspek':<28} {'Skor':>6}  Label")
print("-" * 85)
for teks, aspek in test_cases:
    s = score_windowed(teks, aspek)
    print(f"{teks:<40} {aspek:<28} {s:>6.2f}  {score_to_label(s)}")