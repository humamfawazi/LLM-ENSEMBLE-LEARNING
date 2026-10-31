# ── Bangun DataFrame hasil ensemble ──
ensemble_rows = []
for i in range(len(df_test)):
    row = df_test.iloc[i]
    hard_label = int(results['hard_labels'][i])
    is_bulat   = bool(results['vote_type_bulat'][i])
    soft_label = int(results['soft_labels'][i])
    margin     = float(results['margins'][i])
    p_ens      = results['probs_ensemble'][i].tolist()

    # vote_type: 'bulat' hanya jika semua len(available_models) model setuju
    vote_type = 'bulat' if is_bulat else 'terpecah'

    erow = {
        'pair_id'           : row['pair_id'],
        'aspek'             : row['aspek'],
        'true_label'        : int(row['label']) if 'label' in row else None,
        'hard_vote_label'   : hard_label,
        'vote_type'         : vote_type,
        'soft_vote_label'   : soft_label,
        'soft_vote_margin'  : round(margin, 6),
        'prob_ensemble_json': json.dumps([round(p, 6) for p in p_ens]),
    }
    for model_name in available_models:
        erow[f'pred_{model_name}'] = int(results['preds_per_model'][model_name][i])

    ensemble_rows.append(erow)

df_ensemble = pd.DataFrame(ensemble_rows)

# Simpan
ens_path = os.path.join(PRED_DIR, 'ensemble_test.csv')
df_ensemble.to_csv(ens_path, index=False, encoding='utf-8-sig')

print(f"✓ Ensemble test disimpan: {ens_path}")
print(f"  Shape: {df_ensemble.shape}")
print(f"\nDistribusi vote_type:")
print(df_ensemble['vote_type'].value_counts())
print(f"\nDistribusi soft_vote_margin:")
print(df_ensemble['soft_vote_margin'].describe().round(4))
df_ensemble.head()