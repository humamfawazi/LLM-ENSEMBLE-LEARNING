def aggregate_seeds(model_name: str, split: str, seeds: list, pred_dir: str):
    """
    Rata-ratakan skor softmax dari semua seed untuk satu model pada satu split.
    
    Return: DataFrame dengan pair_id dan prob_ columns yang sudah dirata-rata.
    """
    prob_cols = ['prob_tidak_disebutkan', 'prob_positif', 'prob_negatif', 'prob_netral']
    dfs = []
    
    for seed in seeds:
        path = os.path.join(pred_dir, f"{model_name}_seed{seed}_{split}.csv")
        if not os.path.exists(path):
            print(f"  ⚠️  File tidak ditemukan: {path}")
            continue
        dfs.append(pd.read_csv(path))
    
    if not dfs:
        return None
    
    base = dfs[0][['pair_id']].copy()
    avg_probs = np.mean([df[prob_cols].values for df in dfs], axis=0)
    
    for i, col in enumerate(prob_cols):
        base[col] = avg_probs[:, i]
    
    base['pred_label'] = np.argmax(avg_probs, axis=1).astype(int)
    
    # Tambahkan true_label jika ada
    if 'true_label' in dfs[0].columns:
        base['true_label'] = dfs[0]['true_label'].values
    
    return base


print("Mengagregasi prediksi antar-seed...")

for model_name in MODELS:
    for split in ['dev', 'test']:
        out_path = os.path.join(PRED_DIR, f"{model_name}_{split}_avg.csv")
        df_avg = aggregate_seeds(model_name, split, SEEDS, PRED_DIR)
        if df_avg is not None:
            df_avg.to_csv(out_path, index=False, encoding='utf-8-sig')
            macro_f1 = f1_score(
                df_avg['true_label'], df_avg['pred_label'],
                average='macro', zero_division=0
            ) if 'true_label' in df_avg.columns else float('nan')
            print(f"  {model_name:15s} {split:4s} avg → {out_path.split(os.sep)[-1]}  macro-F1={macro_f1:.4f}")

print("\n✓ Agregasi selesai.")