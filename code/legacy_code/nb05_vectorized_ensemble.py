def hard_vote(pred_a: int, pred_b: int, pred_c: int) -> tuple:
    """Return (label_mayoritas, 'bulat' atau 'terpecah')"""
    counts = Counter([pred_a, pred_b, pred_c])
    label, count = counts.most_common(1)[0]
    vote_type = 'bulat' if count == 3 else 'terpecah'
    return int(label), vote_type


def vectorized_ensemble_gpu(
        probs_dict: dict,          # {model_name: np.ndarray (N, 4)}
        weights_per_aspect: dict,  # {aspek: {model_name: float}}
        pair_ids: list,
        aspek_series: pd.Series,
        device
) -> dict:
    """
    Vectorized ensemble voting sepenuhnya di GPU.
    
    - Hard voting menggunakan mode argmax per tensor (GPU)
    - Weighted soft voting menggunakan matrix multiplication di GPU
    - Satu transfer GPU→CPU di akhir
    
    Returns dict berisi array-array hasil:
      hard_labels, vote_types, soft_labels, margins, probs_ensemble
    """
    N = len(pair_ids)
    model_list = list(probs_dict.keys())
    M = len(model_list)

    # ── Transfer semua probs ke GPU sekaligus ──
    # Tensor shape: (M, N, 4)
    probs_gpu = torch.stack([
        torch.from_numpy(probs_dict[m].astype(np.float32)).to(device)
        for m in model_list
    ], dim=0)  # (M, N, 4)

    # ── Hard Voting (GPU) ──
    # argmax per model per sampel → (M, N)
    preds_gpu = probs_gpu.argmax(dim=-1)  # (M, N) int64

    # Mode voting: one-hot count → argmax
    # shape (M, N, num_classes) → sum across M
    one_hot = F.one_hot(preds_gpu, num_classes=NUM_LABELS).float()  # (M, N, 4)
    vote_counts = one_hot.sum(dim=0)      # (N, 4) — jumlah vote per kelas
    hard_preds_gpu = vote_counts.argmax(dim=-1)        # (N,)
    max_vote_counts = vote_counts.max(dim=-1).values   # (N,)

    # Vote type: bulat jika semua M model setuju
    vote_type_bulat = (max_vote_counts == M)  # (N,) bool

    # ── Weighted Soft Voting (GPU) ──
    # Bobot berbeda per aspek — harus dihitung per sampel
    # Buat weight tensor: (N, M)
    aspek_arr = aspek_series.values
    weight_matrix = torch.zeros(N, M, dtype=torch.float32, device=device)

    for j, model_name in enumerate(model_list):
        for asp in ASPECTS:
            w = weights_per_aspect.get(asp, {}).get(model_name, 1.0 / M)
            asp_mask = torch.tensor(
                aspek_arr == asp, dtype=torch.float32, device=device
            )
            weight_matrix[:, j] += asp_mask * w

    # Normalisasi bobot per sampel: (N, M)
    weight_sum = weight_matrix.sum(dim=1, keepdim=True).clamp(min=1e-9)
    weight_matrix = weight_matrix / weight_sum  # (N, M) normalized

    # Weighted sum: (N, M) x (M, N, 4) → (N, 4)
    # Gunakan einsum untuk efisiensi
    probs_ensemble_gpu = torch.einsum(
        'nm,mnd->nd', weight_matrix, probs_gpu
    )  # (N, 4)

    soft_preds_gpu = probs_ensemble_gpu.argmax(dim=-1)  # (N,)

    # Margin: selisih prob tertinggi - prob kedua tertinggi
    top2_probs = probs_ensemble_gpu.topk(2, dim=-1).values  # (N, 2)
    margins_gpu = top2_probs[:, 0] - top2_probs[:, 1]       # (N,)

    # ── Transfer ke CPU sekaligus ──
    hard_preds_np     = hard_preds_gpu.cpu().numpy().astype(int)
    vote_type_bulat_np = vote_type_bulat.cpu().numpy()
    soft_preds_np     = soft_preds_gpu.cpu().numpy().astype(int)
    margins_np        = margins_gpu.cpu().numpy()
    probs_ens_np      = probs_ensemble_gpu.cpu().numpy()
    preds_per_model_np = preds_gpu.cpu().numpy()  # (M, N)

    return {
        'hard_labels'       : hard_preds_np,
        'vote_type_bulat'   : vote_type_bulat_np,
        'soft_labels'       : soft_preds_np,
        'margins'           : margins_np,
        'probs_ensemble'    : probs_ens_np,
        'preds_per_model'   : {m: preds_per_model_np[j] for j, m in enumerate(model_list)},
    }

print("✓ Fungsi vectorized ensemble GPU siap.")