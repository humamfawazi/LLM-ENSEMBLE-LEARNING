model_weights_dev = {asp: {} for asp in ASPECTS}

for asp in ASPECTS:
    dev_asp = df_dev[df_dev['aspek'] == asp].copy()
    for model_name in MODELS:
        if model_name not in dev_preds:
            model_weights_dev[asp][model_name] = 0.0
            continue
        merged = dev_asp.merge(
            dev_preds[model_name][['pair_id', 'pred_label']], on='pair_id', how='left'
        )
        mask = merged['pred_label'].notna()
        if mask.sum() == 0:
            model_weights_dev[asp][model_name] = 0.0
            continue
        f1 = f1_score(
            merged.loc[mask, 'label'],
            merged.loc[mask, 'pred_label'].astype(int),
            average='macro', zero_division=0
        )
        model_weights_dev[asp][model_name] = round(f1, 4)

print('Bobot model (dev F1 per aspek) — baseline NB05:')
df_weights = pd.DataFrame(model_weights_dev).T
print(df_weights.round(4).to_string())
