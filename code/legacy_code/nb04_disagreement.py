def disagreement_rate(df, col_a: str, col_b: str) -> float:
    """Persentase pasangan di mana kedua model prediksi berbeda."""
    mask = df[col_a].notna() & df[col_b].notna()
    if mask.sum() == 0:
        return float('nan')
    return (df.loc[mask, col_a] != df.loc[mask, col_b]).mean()


# ── Matriks disagreement 3×3 ──
model_cols = {m: f'pred_{m}' for m in MODELS}
disagr_matrix = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)

for m1 in MODELS:
    for m2 in MODELS:
        if m1 == m2:
            disagr_matrix.loc[m1, m2] = 0.0
        else:
            c1, c2 = model_cols.get(m1), model_cols.get(m2)
            if c1 and c2 and c1 in df_analysis.columns and c2 in df_analysis.columns:
                disagr_matrix.loc[m1, m2] = disagreement_rate(df_analysis, c1, c2)
            else:
                disagr_matrix.loc[m1, m2] = float('nan')

print("Disagreement Rate Matrix (antar pasangan model):")
print((disagr_matrix * 100).round(2).to_string(), "(%)")