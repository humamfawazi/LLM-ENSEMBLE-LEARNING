# Merge semua prediksi ke dataframe utama
df_analysis = df_dev[['pair_id', 'tweet_id', 'aspek', 'label']].copy()
df_analysis = df_analysis.rename(columns={'label': 'true_label'})

for model_name, df_pred in preds_dev.items():
    df_analysis = df_analysis.merge(
        df_pred[['pair_id', 'pred_label']].rename(columns={'pred_label': f'pred_{model_name}'}),
        on='pair_id', how='left'
    )

print(f"Analysis df: {df_analysis.shape}")
print(f"Kolom: {list(df_analysis.columns)}")
df_analysis.head()