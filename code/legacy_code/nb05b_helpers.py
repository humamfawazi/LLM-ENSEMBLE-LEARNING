# Label kelas sentimen — WAJIB eksplisit, identik dengan NB07 compute_asc_f1()
# Tanpa ini, kelas 0 ikut macro average jika ada prediksi 0 di subset ASC
ASC_LABELS = [1, 2, 3]  # positif, negatif, netral


def compute_metrics(y_true, y_pred, label=''):
    """Hitung ACD F1, ASC F1, Overall F1 — konsisten dengan NB07."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # ACD: biner tidak_disebutkan=0 vs disebutkan=1
    y_acd_true = (y_true != 0).astype(int)
    y_acd_pred = (y_pred != 0).astype(int)
    acd_f1 = f1_score(y_acd_true, y_acd_pred, average='macro', zero_division=0)

    # ASC: filter baris bukan tidak_disebutkan, labels=[1,2,3] seperti NB07
    mask = y_true != 0
    asc_f1 = f1_score(
        y_true[mask], y_pred[mask],
        average='macro', labels=ASC_LABELS, zero_division=0
    ) if mask.sum() > 0 else 0.0

    overall = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if label:
        print(f'  {label:45s}  ACD={acd_f1:.4f}  ASC={asc_f1:.4f}  Overall={overall:.4f}')
    return {'acd': acd_f1, 'asc': asc_f1, 'overall': overall}


def get_probs(model_name, split='test'):
    """Ambil probability matrix (N, 4) untuk split tertentu."""
    src = test_preds if split == 'test' else dev_preds
    df_ref = df_test if split == 'test' else df_dev
    df_pred = src.get(model_name)
    if df_pred is None:
        return np.full((len(df_ref), NUM_LABELS), 1.0 / NUM_LABELS, dtype=np.float32)
    df_aligned = df_ref[['pair_id']].merge(
        df_pred[['pair_id'] + PROB_COLS], on='pair_id', how='left'
    )
    return df_aligned[PROB_COLS].fillna(0.25).values.astype(np.float32)


def soft_vote_uniform(probs_list):
    """Simple average soft voting."""
    return np.mean(probs_list, axis=0).argmax(axis=1)


def hard_vote(pred_list):
    """Hard voting (majority vote) dari list of (N,) arrays."""
    stacked = np.stack(pred_list, axis=1)
    return np.array([Counter(row).most_common(1)[0][0] for row in stacked])


def apply_temperature(probs_dict, T=1.0):
    """Temperature scaling: T<1 = lebih confident, T>1 = lebih smooth."""
    scaled = {}
    for m, p in probs_dict.items():
        logits = np.log(np.clip(p, 1e-10, 1.0)) / T
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        exp_l = np.exp(logits)
        scaled[m] = exp_l / exp_l.sum(axis=1, keepdims=True)
    return scaled


def weighted_soft_vote_aspect(probs_dict, weights_per_aspect, aspek_series):
    """Weighted soft voting dengan bobot berbeda per aspek."""
    N = len(aspek_series)
    combined = np.zeros((N, NUM_LABELS), dtype=np.float32)
    aspek_arr = np.array(aspek_series)
    for i, asp in enumerate(aspek_arr):
        w_asp = weights_per_aspect.get(asp, {})
        total_w = sum(w_asp.values()) or 1.0
        for m, p in probs_dict.items():
            w = w_asp.get(m, 0.0) / total_w
            combined[i] += w * p[i]
    return combined.argmax(axis=1)


def make_power_weights(base_weights, power=2.0):
    """Pangkatkan bobot F1 agar model terkuat lebih dominan."""
    new_w = {}
    for asp, w_dict in base_weights.items():
        powered = {m: max(v, 0.0) ** power for m, v in w_dict.items()}
        total = sum(powered.values()) or 1.0
        new_w[asp] = {m: v / total for m, v in powered.items()}
    return new_w

print('Helper functions siap.')
