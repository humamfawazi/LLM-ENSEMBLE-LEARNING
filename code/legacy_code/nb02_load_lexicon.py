def load_lexicon(filepath):
    """Load InSet lexicon TSV (word<TAB>weight)."""
    lexicon = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                try:
                    word, weight = parts[0].strip(), float(parts[1].strip())
                    lexicon[word.lower()] = weight
                except ValueError:
                    continue  # Skip header row
    return lexicon

pos_lexicon_path = os.path.join(LEXICON_DIR, 'positive.tsv')
neg_lexicon_path = os.path.join(LEXICON_DIR, 'negative.tsv')

if os.path.exists(pos_lexicon_path) and os.path.exists(neg_lexicon_path):
    pos_lexicon = load_lexicon(pos_lexicon_path)
    neg_lexicon = load_lexicon(neg_lexicon_path)
    print(f"Positive lexicon: {len(pos_lexicon):,} kata")
    print(f"Negative lexicon: {len(neg_lexicon):,} kata")
else:
    print("⚠️  File lexicon tidak ditemukan!")
    print(f"   Unduh dari: https://github.com/fajri91/InSet")
    print(f"   Simpan ke : {LEXICON_DIR}/positive.tsv & negative.tsv")
    pos_lexicon = {}
    neg_lexicon = {}