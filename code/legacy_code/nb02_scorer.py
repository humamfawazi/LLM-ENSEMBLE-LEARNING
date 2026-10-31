# Kata negasi untuk pembalikan skor
NEGATION_WORDS = {'tidak', 'bukan', 'kurang', 'jangan', 'tak', 'nggak', 'gak'}
NEGATION_WINDOW = 2  # Jarak maksimal kata negasi sebelum kata sentimen


def score_sentence(text: str) -> float:
    """Hitung skor sentimen lexicon-based dengan penanganan negasi.
    
    1. Tokenisasi: lowercase, split spasi, buang tanda baca
    2. Untuk tiap token, cek kamus positif/negatif → ambil bobot
    3. Jika kata negasi muncul dalam jarak NEGATION_WINDOW token sebelum
       kata sentimen → kalikan bobot × -1
    4. Jumlahkan semua bobot → skor akhir
    """
    text_lower = str(text).lower()
    # Tokenisasi: hapus tanda baca, split spasi
    tokens = re.sub(r'[^\w\s]', '', text_lower).split()
    
    total_score = 0.0
    for i, token in enumerate(tokens):
        weight = 0.0
        if token in pos_lexicon:
            weight = pos_lexicon[token]
        elif token in neg_lexicon:
            weight = neg_lexicon[token]
        else:
            continue
        
        # Cek negasi dalam jendela sebelumnya
        start = max(0, i - NEGATION_WINDOW)
        preceding = tokens[start:i]
        if any(neg in NEGATION_WORDS for neg in preceding):
            weight *= -1
        
        total_score += weight
    
    return total_score


def score_to_label(score: float, threshold: float = 0.5) -> str:
    """Konversi skor sentimen ke label."""
    if score > threshold:
        return 'positif'
    elif score < -threshold:
        return 'negatif'
    else:
        return 'netral'


# Test cepat
test_texts = [
    "makanan ini enak dan bergizi",
    "obat ini tidak bagus efek sampingnya parah",
    "hari ini cerah"
]
for t in test_texts:
    s = score_sentence(t)
    print(f"  '{t}' → score={s:.2f} → {score_to_label(s)}")

