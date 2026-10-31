# ── EKSPERIMEN 2: Power Weighting ───────────────────────────────────
print('\nEKSPERIMEN 2 — Power Weighting (bobot dipangkatkan)')
print('-' * 78)
for power in [1.5, 2.0, 3.0, 4.0, 5.0]:
    pw = make_power_weights(model_weights_dev, power=power)
    preds_pw = weighted_soft_vote_aspect(probs_all, pw, df_test['aspek'])
    r = compute_metrics(y_true, preds_pw, label=f'Soft Vote power={power:.1f} (3 model)')
    all_results[f'07_SoftVote_power{power}'] = r

    # Power weighting, hanya 2 model terbaik (drop mbert)
    best2 = [m for m in available if m != 'mbert']
    if len(best2) == 2:
        probs_best2 = {m: probs_all[m] for m in best2}
        pw2 = make_power_weights(
            {asp: {m: v for m, v in w.items() if m in best2}
             for asp, w in model_weights_dev.items()},
            power=power
        )
        preds_pw2 = weighted_soft_vote_aspect(probs_best2, pw2, df_test['aspek'])
        r2 = compute_metrics(y_true, preds_pw2,
                             label=f'Soft Vote power={power:.1f} (2 model, no mbert)')
        all_results[f'08_SoftVote_power{power}_nombert'] = r2
