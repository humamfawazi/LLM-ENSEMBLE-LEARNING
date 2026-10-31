y_true = df_test['label'].values
available = [m for m in MODELS if m in test_preds]

# Probs & preds semua model
probs_all = {m: get_probs(m, split='test') for m in available}
preds_all = {m: probs_all[m].argmax(axis=1) for m in available}

print('=' * 78)
print(f'  {"Model":42s}  {"ACD F1":8s}  {"ASC F1":8s}  {"Overall":8s}')
print('=' * 78)
diag_results = {}
for m in available:
    r = compute_metrics(y_true, preds_all[m], label=m)
    diag_results[m] = r
print('=' * 78)

# Visualisasi
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
mk_list = ['acd', 'asc', 'overall']
titles  = ['ACD Macro-F1', 'ASC Macro-F1', 'Overall Macro-F1']
colors  = sns.color_palette('Set2', len(available))
for ax, mk, title in zip(axes, mk_list, titles):
    vals = [diag_results.get(m, {}).get(mk, 0) for m in available]
    bars = ax.bar(available, vals, color=colors, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Macro-F1')
    plt.setp(ax.get_xticklabels(), rotation=15)
plt.suptitle('F1 Tiap Model Individu di Test Set', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
