# ── EKSPERIMEN 4: Kombinasi Power + Temperature ──────────────────────
print('\nEKSPERIMEN 4 — Kombinasi Power Weight + Temperature Scaling')
print('-' * 78)

for T_val in [best_T, 0.7, 0.5]:
    for power in [2.0, 3.0, 4.0]:
        scaled = apply_temperature(probs_all, T=T_val)
        pw = make_power_weights(model_weights_dev, power=power)
        preds_combo = weighted_soft_vote_aspect(scaled, pw, df_test['aspek'])
        label = f'Soft Vote power={power:.0f} + T={T_val} (3 model)'
        r = compute_metrics(y_true, preds_combo, label=label)
        all_results[f'12_SoftVote_p{power}_T{T_val}'] = r

        # Dengan 2 model (no mbert)
        if 'mbert' in available:
            best2 = [m for m in available if m != 'mbert']
            scaled2 = {m: scaled[m] for m in best2}
            pw2 = make_power_weights(
                {asp: {m: v for m, v in w.items() if m in best2}
                 for asp, w in model_weights_dev.items()},
                power=power
            )
            preds_c2 = weighted_soft_vote_aspect(scaled2, pw2, df_test['aspek'])
            r2 = compute_metrics(y_true, preds_c2,
                                 label=f'Soft Vote power={power:.0f} + T={T_val} (2 model)')
            all_results[f'13_SoftVote_p{power}_T{T_val}_nombert'] = r2
