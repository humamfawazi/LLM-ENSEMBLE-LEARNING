def score_windowed(text: str, aspect: str, window: int = 5) -> float:
    """Skor sentimen hanya dalam jendela sekitar kata kunci aspek.
    Fallback ke score_sentence jika tidak ada kata kunci ditemukan.
    Memerlukan ASPECT_KEYWORDS, pos_lexicon, neg_lexicon sudah dimuat.
    """
    tokens = re.sub(r'[^\w\s]', '', str(text).lower()).split()
    keywords = [k.lower() for k in ASPECT_KEYWORDS[aspect]]

    # Cari posisi semua kata kunci aspek
    hits = [i for i, t in enumerate(tokens) if any(k in t for k in keywords)]
    if not hits:
        return score_sentence(text)  # fallback

    total = 0.0
    counted = set()
    for pos in hits:
        for i in range(max(0, pos - window), min(len(tokens), pos + window + 1)):
            if i in counted:
                continue
            counted.add(i)
            tok = tokens[i]
            if tok in pos_lexicon:   w = pos_lexicon[tok]
            elif tok in neg_lexicon: w = neg_lexicon[tok]
            else: continue
            if any(n in NEGATION_WORDS for n in tokens[max(0, i - NEGATION_WINDOW):i]):
                w *= -1
            total += w
    return total

print("score_windowed didefinisikan. Test akan dijalankan setelah ASPECT_KEYWORDS dimuat (cell 3.3).")