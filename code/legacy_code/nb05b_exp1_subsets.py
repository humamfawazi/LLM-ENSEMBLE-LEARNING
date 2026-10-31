# ── EKSPERIMEN 1: Subset 2 Model ────────────────────────────────────
print('\nEKSPERIMEN 1 — Subset 2 Model (drop model terlemah)')
print('-' * 78)
for m1, m2 in combinations(available, 2):
    subset = [m1, m2]
    short  = f'{m1[:6]}+{m2[:6]}'

    hv = hard_vote([preds_all[m] for m in subset])
    r  = compute_metrics(y_true, hv, label=f'Hard Vote ({short})')
    all_results[f'04_HardVote_{m1}_{m2}'] = r

    sv = soft_vote_uniform([probs_all[m] for m in subset])
    r  = compute_metrics(y_true, sv, label=f'Soft Vote uniform ({short})')
    all_results[f'05_SoftVote_uniform_{m1}_{m2}'] = r

    # Weighted per aspek, hanya 2 model
    probs_sub = {m: probs_all[m] for m in subset}
    wv = weighted_soft_vote_aspect(probs_sub, model_weights_dev, df_test['aspek'])
    r  = compute_metrics(y_true, wv, label=f'Soft Vote weighted ({short})')
    all_results[f'06_SoftVote_weighted_{m1}_{m2}'] = r
