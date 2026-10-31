def build_pairs(df_labeled, label_col, text_version='text_v1'):
    """
    Bangun sentence-pair dari dataframe berlabel.
    
    Parameters:
        df_labeled: DataFrame dengan kolom tweet_id, text (atau text_vX), aspek, dan label_col
        label_col: nama kolom label (label_gold atau label_silver)
        text_version: kolom teks yang dipakai sebagai sentence_a
    
    Returns:
        DataFrame dengan kolom: pair_id, tweet_id, sentence_a, sentence_b, label
    """
    pairs = []
    
    # Jika text_version tidak ada di df_labeled, cari dari df utama
    if text_version not in df_labeled.columns:
        # Merge dengan df preprocessed untuk mendapatkan text_version
        text_lookup = df[['tweet_id', text_version]].drop_duplicates(subset='tweet_id')
        df_labeled = df_labeled.merge(text_lookup, on='tweet_id', how='left')
    
    for _, row in df_labeled.iterrows():
        aspect = row['aspek']
        label_str = row[label_col]
        
        # Skip jika label kosong
        if pd.isna(label_str) or str(label_str).strip() == '':
            continue
        
        pairs.append({
            'pair_id': f"{row['tweet_id']}_{aspect}",
            'tweet_id': row['tweet_id'],
            'sentence_a': str(row.get(text_version, row['text'])),
            'sentence_b': ASPECT_DESCRIPTIONS[aspect],
            'label': LABEL_MAP[label_str]
        })
    
    return pd.DataFrame(pairs)

print("✓ Fungsi build_pairs siap.")