# ── EKSPERIMEN 3: Temperature Scaling ───────────────────────────────
print('\nEKSPERIMEN 3 — Temperature Scaling (kalibrasi softmax)')
print('-' * 78)

# Cari T terbaik di dev set
print('  Tuning temperature T pada dev set...')
probs_dev_all = {m: get_probs(m, split='dev') for m in available}
y_dev = df_dev['label'].values
mask_dev = y_dev != 0

best_T, best_T_asc = 1.0, -1.0
T_candidates = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
print(f'  {"T":5s}  {"dev ASC F1":12s}')
for T in T_candidates:
    scaled_dev = apply_temperature(probs_dev_all, T=T)
    sv_dev = soft_vote_uniform(list(scaled_dev.values()))
    if mask_dev.sum() > 0:
        # Gunakan ASC_LABELS konsisten dengan compute_metrics() dan NB07
        asc = f1_score(y_dev[mask_dev], sv_dev[mask_dev],
                       average='macro', labels=ASC_LABELS, zero_division=0)
        marker = ' <- best' if asc > best_T_asc else ''
        print(f'  {T:<5.1f}  {asc:.4f}{marker}')
        if asc > best_T_asc:
            best_T_asc, best_T = asc, T

print(f'  Best T dari dev: T={best_T}  (dev ASC F1={best_T_asc:.4f})')
print()

# Terapkan ke test
T_test_list = sorted(set(T_candidates + [best_T]))
for T_val in T_test_list:
    scaled_test = apply_temperature(probs_all, T=T_val)

    hv_t = hard_vote([p.argmax(axis=1) for p in scaled_test.values()])
    r1 = compute_metrics(y_true, hv_t, label=f'Hard Vote + Temp T={T_val}')
    all_results[f'09_HardVote_T{T_val}'] = r1

    sv_t = soft_vote_uniform(list(scaled_test.values()))
    r2 = compute_metrics(y_true, sv_t, label=f'Soft Vote uniform + Temp T={T_val}')
    all_results[f'10_SoftVote_uniform_T{T_val}'] = r2

    sv_tw = weighted_soft_vote_aspect(scaled_test, model_weights_dev, df_test['aspek'])
    r3 = compute_metrics(y_true, sv_tw, label=f'Soft Vote weighted + Temp T={T_val}')
    all_results[f'11_SoftVote_weighted_T{T_val}'] = r3
