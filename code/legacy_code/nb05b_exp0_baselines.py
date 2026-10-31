all_results = {}

# ── Baseline: individu ──────────────────────────────────────────────
print('BASELINE — Model Individu')
print('-' * 78)
for m in available:
    r = compute_metrics(y_true, preds_all[m], label=f'Individual {m}')
    all_results[f'01_Individual_{m}'] = r

# ── Baseline NB05 ───────────────────────────────────────────────────
print('\nBASELINE — NB05')
print('-' * 78)
hv_baseline = hard_vote([preds_all[m] for m in available])
r = compute_metrics(y_true, hv_baseline, label='Hard Vote 3 model (NB05 baseline)')
all_results['02_HardVote_NB05_baseline'] = r

sv_baseline = weighted_soft_vote_aspect(probs_all, model_weights_dev, df_test['aspek'])
r = compute_metrics(y_true, sv_baseline, label='Weighted Soft Vote linear (NB05 baseline)')
all_results['03_SoftVote_linear_NB05_baseline'] = r
