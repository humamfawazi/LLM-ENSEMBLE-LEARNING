# ── Siapkan dict probs untuk semua model yang tersedia ──
available_models = [m for m in MODELS if m in test_preds]
print(f"Model tersedia untuk ensemble: {available_models}")

if len(available_models) < 2:
    print("⚠️  Minimal 2 model dibutuhkan untuk ensemble. Periksa prediksi dari NB3.")
else:
    # Align semua prediksi ke urutan df_test
    probs_dict = {}
    for model_name in available_models:
        df_pred = test_preds[model_name]
        df_aligned = df_test[['pair_id']].merge(
            df_pred[['pair_id'] + PROB_COLS], on='pair_id', how='left'
        )
        probs_dict[model_name] = df_aligned[PROB_COLS].fillna(0.25).values.astype(np.float32)

    print(f"\nMenjalankan ensemble ({len(available_models)} model, N={len(df_test):,}) di {DEVICE}...")
    results = vectorized_ensemble_gpu(
        probs_dict       = probs_dict,
        weights_per_aspect = model_weights,
        pair_ids         = df_test['pair_id'].tolist(),
        aspek_series     = df_test['aspek'],
        device           = DEVICE
    )
    print("✓ Ensemble selesai!")

    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()