# PENTING: bobot dihitung dari DEV set, BUKAN test set
model_weights = {asp: {} for asp in ASPECTS}
model_f1_dev  = {asp: {} for asp in ASPECTS}

for asp in ASPECTS:
    dev_asp = df_dev[df_dev['aspek'] == asp].copy()

    for model_name, df_pred in dev_preds.items():
        merged = dev_asp.merge(
            df_pred[['pair_id', 'pred_label']],
            on='pair_id', how='left'
        )
        mask = merged['pred_label'].notna()
        if mask.sum() == 0:
            model_weights[asp][model_name] = 0.0
            continue

        f1 = f1_score(
            merged.loc[mask, 'label'],
            merged.loc[mask, 'pred_label'].astype(int),
            average='macro', zero_division=0
        )
        model_weights[asp][model_name] = round(f1, 4)
        model_f1_dev[asp][model_name]  = round(f1, 4)

print("Model weights (macro-F1 dev per aspek):")
for asp, w in model_weights.items():
    print(f"  {asp}: {w}")