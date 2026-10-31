"""Shared helpers for the revision notebooks (no training, no API).

All metrics are computed from 4x4 confusion matrices so that cluster
bootstrap and cluster permutation can be vectorised.  Equivalence with
sklearn (as used in notebooks/07_Evaluasi_dan_Pelaporan.ipynb) is asserted
in every notebook before any inference is drawn.

Label coding (from the original pipeline):
    0 = tidak_disebutkan, 1 = positif, 2 = negatif, 3 = netral
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [0, 1, 2, 3]
LABEL_NAMES = {0: 'tidak_disebutkan', 1: 'positif', 2: 'negatif', 3: 'netral'}
LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}
ASPECTS = ['pola_konsumsi', 'aktivitas_fisik', 'akses_layanan_kesehatan', 'pengetahuan_pengobatan']
MODELS = ['indobert', 'xlm_roberta', 'mbert']

SCHEMES = ['IndoBERT', 'XLM-RoBERTa', 'mBERT', 'Hard Voting',
           'Weighted Soft Voting', 'Soft Voting + LLM']

# Numbers printed in the manuscript: ACD F1 / ASC F1 / Overall F1 / Accuracy
PAPER_NUMBERS = {
    'IndoBERT':             (0.8638, 0.3990, 0.4880, 0.7161),
    'XLM-RoBERTa':          (0.8580, 0.3851, 0.4761, 0.7071),
    'mBERT':                (0.8562, 0.3537, 0.4556, 0.6946),
    'Hard Voting':          (0.8674, 0.3814, 0.4783, 0.7125),
    'Weighted Soft Voting': (0.8641, 0.3755, 0.4716, 0.7054),
    'Soft Voting + LLM':    (0.8804, 0.4502, 0.5366, 0.7536),
}
METRIC_KEYS = ['acd_f1', 'asc_f1', 'overall_f1', 'accuracy']

# Sumber label "Soft Voting + LLM":
#   'snapshot' (UTAMA)  = run ulang qwen3-max-2026-01-23, dibangun oleh 00_build_snapshot_hybrid.ipynb dari cache
#   'alias'   (sensitivitas) = run asli NB6 dengan alias qwen3-max (21 Sep 2026), angka naskah lama
HYBRID_SOURCES = ('snapshot', 'alias')
SNAPSHOT_HYBRID_FILE = 'revision/outputs/hybrid_snapshot_final_labels_test.csv'
HYBRID_NUMBERS = {
    'alias':    (0.8804, 0.4502, 0.5366, 0.7536),
    'snapshot': (0.8804, 0.4517, 0.5372, 0.7536),
}
DECISION_COUNTS = {'alias': {'retain': 41, 'revise': 53}, 'snapshot': {'retain': 42, 'revise': 52}}


def expected_numbers(hybrid: str = 'snapshot') -> dict:
    """Angka acuan untuk assertion: model individu & voting = naskah; hybrid sesuai sumber."""
    assert hybrid in HYBRID_SOURCES
    return {**PAPER_NUMBERS, 'Soft Voting + LLM': HYBRID_NUMBERS[hybrid]}


def llm_cache_key(model, messages, temperature, max_tokens, extra_body) -> str:
    """Kunci cache yang sama dengan CachedLoggingClient di Notebook 04."""
    import hashlib
    blob = json.dumps({'model': model, 'messages': messages, 'temperature': temperature,
                       'max_tokens': max_tokens, 'extra_body': extra_body}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


# ─────────────────────────────────────────────────────────────
# Paths & loading
# ─────────────────────────────────────────────────────────────
def find_dirs() -> tuple[Path, Path]:
    """Return (REV_DIR, BASE_DIR) whether cwd is revision/ or the project root."""
    cwd = Path.cwd()
    rev = cwd if (cwd / 'revision_utils.py').exists() else cwd / 'revision'
    assert (rev / 'revision_utils.py').exists(), f'revision/ not found from {cwd}'
    return rev, rev.parent


def load_test_frame(base_dir: Path, hybrid: str = 'snapshot') -> pd.DataFrame:
    """One row per test pair with gold label and the predictions of the six schemes.

    Model individu & voting: file yang sama dengan notebooks/07_Evaluasi_dan_Pelaporan.ipynb.
    'Soft Voting + LLM': hybrid='snapshot' (utama) dari SNAPSHOT_HYBRID_FILE, atau
    hybrid='alias' dari outputs/predictions/final_labels_test.csv (run asli NB6).
    """
    assert hybrid in HYBRID_SOURCES, hybrid
    pairs = pd.read_csv(base_dir / 'data/pairs/test_pairs.csv', dtype={'tweet_id': str})
    pairs['pair_id'] = pairs['pair_id'].astype(str)
    df = pairs[['pair_id', 'tweet_id', 'sentence_a', 'label']].rename(columns={'label': 'y_true'})
    df['aspek'] = df['pair_id'].map(lambda p: next(a for a in ASPECTS if p.endswith('_' + a)))

    pred_dir = base_dir / 'outputs/predictions'
    names = {'indobert': 'IndoBERT', 'xlm_roberta': 'XLM-RoBERTa', 'mbert': 'mBERT'}
    for m, nice in names.items():
        p = pd.read_csv(pred_dir / f'{m}_test_avg.csv')
        p['pair_id'] = p['pair_id'].astype(str)
        df = df.merge(p[['pair_id', 'pred_label']].rename(columns={'pred_label': nice}),
                      on='pair_id', how='left')

    ens = pd.read_csv(pred_dir / 'ensemble_test.csv')
    ens['pair_id'] = ens['pair_id'].astype(str)
    ens = ens.rename(columns={'hard_vote_label': 'Hard Voting',
                              'soft_vote_label': 'Weighted Soft Voting'})
    df = df.merge(ens[['pair_id', 'Hard Voting', 'Weighted Soft Voting', 'vote_type',
                       'soft_vote_margin', 'pred_indobert', 'pred_xlm_roberta', 'pred_mbert']],
                  on='pair_id', how='left')

    fin_path = pred_dir / 'final_labels_test.csv' if hybrid == 'alias' else base_dir / SNAPSHOT_HYBRID_FILE
    if not fin_path.exists():
        raise FileNotFoundError(f'{fin_path} belum ada — jalankan 00_build_snapshot_hybrid.ipynb dulu')
    fin = pd.read_csv(fin_path)
    fin['pair_id'] = fin['pair_id'].astype(str)
    fin = fin.rename(columns={'final_label_int': 'Soft Voting + LLM', 'reasoning': 'llm_reasoning'})
    df = df.merge(fin[['pair_id', 'Soft Voting + LLM', 'decision', 'llm_reasoning']],
                  on='pair_id', how='left')

    assert len(df) == 560 and df[SCHEMES].notna().all().all()
    for s in SCHEMES:
        df[s] = df[s].astype(int)
    df['flagged'] = (df['vote_type'] == 'terpecah') | (df['soft_vote_margin'] < 0.15)
    df.attrs['hybrid_source'] = hybrid
    return df


# ─────────────────────────────────────────────────────────────
# Metrics (sklearn reference + confusion-matrix vectorised version)
# ─────────────────────────────────────────────────────────────
def sklearn_metrics(y_true, y_pred) -> dict:
    """Exact replica of compute_all_metrics() in notebook 07 (unrounded)."""
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return {
        'acd_f1': f1_score((y_true != 0).astype(int), (y_pred != 0).astype(int),
                           average='macro', zero_division=0),
        'asc_f1': f1_score(y_true[mask], y_pred[mask], average='macro',
                           labels=[1, 2, 3], zero_division=0),
        'overall_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'accuracy': float((y_true == y_pred).mean()),
    }


def confusion(y_true, y_pred, k: int = 4) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def cluster_confusions(df: pd.DataFrame, pred_cols: list[str], cluster_col: str = 'tweet_id',
                       true_col: str = 'y_true') -> tuple[np.ndarray, np.ndarray]:
    """Return (cluster_ids, C) with C[c, s, i, j] = #pairs in cluster c, gold i, scheme s predicts j."""
    ids, codes = np.unique(df[cluster_col].values, return_inverse=True)
    C = np.zeros((len(ids), len(pred_cols), 4, 4), dtype=np.int64)
    yt = df[true_col].values.astype(int)
    for s, col in enumerate(pred_cols):
        np.add.at(C, (codes, s, yt, df[col].values.astype(int)), 1)
    return ids, C


def _safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.zeros(np.broadcast(a, b).shape)
    np.divide(a, b, out=out, where=b > 0)
    return out


def per_class_from_cm(cm: np.ndarray) -> dict:
    """Precision/recall/F1 per class with zero_division=0 and fixed labels [0..3]. cm: (...,4,4)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    return {'precision': _safe_div(tp, tp + fp), 'recall': _safe_div(tp, tp + fn),
            'f1': _safe_div(2 * tp, 2 * tp + fp + fn), 'support': cm.sum(axis=-1)}


def metrics_from_cm(cm: np.ndarray) -> dict:
    """ACD / ASC / Overall macro-F1 and accuracy from confusion matrices (...,4,4).

    Fixed label lists and zero_division=0 (a class absent from both gold and
    prediction in a resample contributes F1 = 0).
    """
    cm = np.asarray(cm, dtype=float)
    n = cm.sum(axis=(-1, -2))
    acc = _safe_div(np.trace(cm, axis1=-2, axis2=-1), n)
    overall = per_class_from_cm(cm)['f1'].mean(axis=-1)

    # ASC: rows with gold in {1,2,3}; labels [1,2,3]; predicted 0 counts as a miss
    f1s = []
    for k in (1, 2, 3):
        tp = cm[..., k, k]
        fn = cm[..., k, :].sum(axis=-1) - tp
        fp = cm[..., 1:, k].sum(axis=-1) - tp
        f1s.append(_safe_div(2 * tp, 2 * tp + fp + fn))
    asc = np.mean(f1s, axis=0)

    # ACD: binary mentioned (gold != 0) vs not
    n00 = cm[..., 0, 0]
    n01 = cm[..., 0, 1:].sum(axis=-1)
    n10 = cm[..., 1:, 0].sum(axis=-1)
    n11 = cm[..., 1:, 1:].sum(axis=(-1, -2))
    f1_0 = _safe_div(2 * n00, 2 * n00 + n10 + n01)
    f1_1 = _safe_div(2 * n11, 2 * n11 + n01 + n10)
    acd = (f1_0 + f1_1) / 2
    return {'acd_f1': acd, 'asc_f1': asc, 'overall_f1': overall, 'accuracy': acc}


def kappa_from_cm(cm: np.ndarray) -> np.ndarray:
    """Cohen's kappa from (...,k,k) confusion matrices (NaN if expected agreement == 1)."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum(axis=(-1, -2))
    po = _safe_div(np.trace(cm, axis1=-2, axis2=-1), n)
    pe = _safe_div((cm.sum(axis=-1) * cm.sum(axis=-2)).sum(axis=-1), n ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        k = (po - pe) / (1 - pe)
    return np.where(np.isclose(pe, 1), np.nan, k)


# ─────────────────────────────────────────────────────────────
# Cluster bootstrap & cluster permutation
# ─────────────────────────────────────────────────────────────
def bootstrap_weights(n_clusters: int, B: int, seed: int) -> np.ndarray:
    """W[b, c] = how often cluster c is drawn in resample b (sampling clusters with replacement)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=B)


def boot_cms(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Resampled confusion matrices: (B, ...) from W (B, n_c) and C (n_c, ...)."""
    return np.tensordot(W, C, axes=([1], [0]))


def pct_ci(x: np.ndarray, alpha: float = 0.05, axis: int = 0):
    lo, hi = np.nanpercentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=axis)
    return lo, hi


def cluster_permutation_test(CA: np.ndarray, CB: np.ndarray, metric_fn, B: int, seed: int):
    """Paired two-sided permutation test at cluster level.

    CA, CB: (n_c, ..., 4, 4) per-cluster confusion matrices of systems A and B
    computed on the same items.  For every cluster the two systems' predictions
    are swapped with probability 1/2.  Statistic = metric(A) - metric(B).
    Returns (observed_delta, p_value, null_distribution).
    """
    rng = np.random.default_rng(seed)
    D = (CB - CA).astype(float)
    totA, totB = CA.sum(axis=0).astype(float), CB.sum(axis=0).astype(float)
    obs = float(metric_fn(totA) - metric_fn(totB))
    S = rng.integers(0, 2, size=(B, CA.shape[0])).astype(float)
    shift = np.tensordot(S, D, axes=([1], [0]))
    null = metric_fn(totA + shift) - metric_fn(totB - shift)
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (B + 1)
    return obs, float(p), null


def holm(pvals) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    return multipletests(np.asarray(pvals, float), method='holm')[1]


def durkalski_mcnemar(correct_a, correct_b, clusters) -> dict:
    """Clustered McNemar test (Durkalski et al., 2003, Stat Med 22:2417-28)."""
    from scipy.stats import chi2
    d = pd.DataFrame({'b': (correct_a & ~correct_b).astype(int),
                      'c': (~correct_a & correct_b).astype(int), 'k': clusters})
    g = d.groupby('k')[['b', 'c']].sum()
    num = (g['b'].sum() - g['c'].sum()) ** 2
    den = ((g['b'] - g['c']) ** 2).sum()
    stat = num / den if den > 0 else 0.0
    return {'b': int(g['b'].sum()), 'c': int(g['c'].sum()), 'chi2': float(stat),
            'p': float(chi2.sf(stat, 1)) if den > 0 else 1.0}


def tweet_level_sign_test(correct_a, correct_b, clusters) -> dict:
    """McNemar/sign test on tweets: per tweet compare #aspects correct under A vs B."""
    from scipy.stats import binomtest
    d = pd.DataFrame({'a': correct_a.astype(int), 'b': correct_b.astype(int), 'k': clusters})
    g = d.groupby('k')[['a', 'b']].sum()
    n_a, n_b = int((g['a'] > g['b']).sum()), int((g['b'] > g['a']).sum())
    p = binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b > 0 else 1.0
    return {'tweets_A_better': n_a, 'tweets_B_better': n_b,
            'tweets_tied': int((g['a'] == g['b']).sum()), 'p': float(p)}


# ─────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────
def to_markdown(df: pd.DataFrame, floatfmt: str = '.4f', index: bool = False) -> str:
    d = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return 'NA' if (v is None or np.isnan(v)) else format(v, floatfmt)
        return str(v)
    cols = [str(c) for c in d.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in d.iterrows():
        lines.append('| ' + ' | '.join(fmt(v).replace('|', '/') for v in r.values) + ' |')
    return '\n'.join(lines)


def save_table(df: pd.DataFrame, out_dir: Path, name: str, title: str = '',
               floatfmt: str = '.4f', index: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f'{name}.csv', index=index, encoding='utf-8-sig')
    md = (f'### {title}\n\n' if title else '') + to_markdown(df, floatfmt, index) + '\n'
    (out_dir / f'{name}.md').write_text(md, encoding='utf-8')
    print(f'[saved] {name}.csv / {name}.md')


def fmt_ci(est, lo, hi, nd: int = 4) -> str:
    return f'{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]'


def notebook_cell_sources(nb_path: Path) -> list[str]:
    nb = json.loads(Path(nb_path).read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def find_cell(nb_path: Path, *needles: str) -> str:
    """Return the unique code cell of an existing notebook that contains all needles."""
    hits = [s for s in notebook_cell_sources(nb_path) if all(n in s for n in needles)]
    assert len(hits) == 1, f'{len(hits)} cells in {Path(nb_path).name} match {needles}'
    return hits[0]
