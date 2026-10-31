# ── Silver labeling: untuk tiap tweet × tiap aspek ──
# THRESHOLD diambil dari hasil sweep theta (BEST_THETA)
THRESHOLD = BEST_THETA if 'BEST_THETA' in dir() else 2.0
print(f"Menggunakan THRESHOLD = {THRESHOLD}")

silver_rows = []

for _, row in tqdm(df_for_silver.iterrows(), total=len(df_for_silver), desc="Silver labeling"):
    # Gunakan text_v3 untuk scoring (sudah lowercase, huruf berulang dinormalisasi)
    text_for_scoring = str(row.get('text_v3', row['text']))
    text_for_detect  = str(row['text'])  # Deteksi aspek pada teks asli

    for aspect in ASPECTS:
        if detect_aspect(text_for_detect, aspect):
            # Aspek terdeteksi → score_windowed: skor hanya di sekitar kata kunci aspek
            lexicon_score = score_windowed(text_for_scoring, aspect)
            label = score_to_label(lexicon_score, threshold=THRESHOLD)
        else:
            # Aspek tidak terdeteksi
            label = 'tidak_disebutkan'
            lexicon_score = 0.0

        silver_rows.append({
            'tweet_id'     : row['tweet_id'],
            'text'         : row['text'],
            'aspek'        : aspect,
            'label_silver' : label,
            'lexicon_score': round(lexicon_score, 4)
        })

df_silver = pd.DataFrame(silver_rows)
print(f"\nSilver labeled: {df_silver.shape}")
print(df_silver['label_silver'].value_counts())