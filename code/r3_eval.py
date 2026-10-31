"""Fungsi evaluasi R3_06/R3_07 — logika sama dengan revision/01, 02, 06 (metrik & uji dari revision/revision_utils.py)."""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

import config as C

if str(C.REVISION_DIR) not in sys.path:
    sys.path.insert(0, str(C.REVISION_DIR))
import revision_utils as ru  # noqa: E402

SCHEMES6 = ['IndoBERT', 'XLM-RoBERTa', 'mBERT', 'Hard Voting', 'Weighted Soft Voting', 'Soft Voting + LLM']
REF = 'Soft Voting + LLM'
METRICS = ru.METRIC_KEYS                                   # acd_f1, asc_f1, overall_f1, accuracy
METRIC_LABEL = {'acd_f1': 'ACD F1', 'asc_f1': 'ASC F1', 'overall_f1': 'Overall F1', 'accuracy': 'Akurasi'}
CLASS_NAMES = [C.LABEL_MAP_INV[k] for k in range(4)]


def fmt_ci(e, lo, hi, nd=4):
    return f'{e:.{nd}f} [{lo:.{nd}f}; {hi:.{nd}f}]'


def check_sklearn_equivalence(df, cols):
    """Metrik vektor (confusion matrix) harus sama dengan definisi NB07 (sklearn)."""
    for c in cols:
        sk = ru.sklearn_metrics(df.y_true.values, df[c].values)
        vec = ru.metrics_from_cm(ru.confusion(df.y_true.values, df[c].values))
        for k in METRICS:
            if not (k == 'overall_f1' and len(set(df.y_true) | set(df[c])) < 4):
                assert abs(sk[k] - float(vec[k])) < 1e-12, (c, k)


# ── revision/01 §1: metrik + CI cluster bootstrap ──
def metrics_ci(df, cols, B=C.B_BOOT, seed=C.STAT_SEED):
    ids, Cm = ru.cluster_confusions(df, cols)
    BC = ru.boot_cms(ru.bootstrap_weights(len(ids), B, seed), Cm)
    bt, pt = ru.metrics_from_cm(BC), ru.metrics_from_cm(Cm.sum(0))
    rows = []
    for i, c in enumerate(cols):
        r = {'skema': c, 'n_pasangan': len(df), 'n_tweet': len(ids)}
        for k in METRICS:
            lo, hi = ru.pct_ci(bt[k][:, i])
            r[k], r[f'{k}_lo'], r[f'{k}_hi'] = float(pt[k][i]), lo, hi
        rows.append(r)
    gold_present = BC[:, 0].sum(-1) > 0
    miss = int((~gold_present).any(axis=1).sum())
    return pd.DataFrame(rows), miss


def pretty_metrics(tab):
    out = tab[['skema']].copy()
    for k in METRICS:
        out[METRIC_LABEL[k]] = [fmt_ci(e, l, h) for e, l, h in zip(tab[k], tab[f'{k}_lo'], tab[f'{k}_hi'])]
    return out


# ── revision/01 §2: per kelas ──
def per_class_ci(df, cols, B=C.B_BOOT, seed=C.STAT_SEED):
    ids, Cm = ru.cluster_confusions(df, cols)
    BC = ru.boot_cms(ru.bootstrap_weights(len(ids), B, seed), Cm)
    pp, pb = ru.per_class_from_cm(Cm.sum(0)), ru.per_class_from_cm(BC)
    rows = []
    for i, c in enumerate(cols):
        for k in range(4):
            r = {'skema': c, 'kelas': CLASS_NAMES[k], 'support_gold': int(pp['support'][i, k]), 'n_prediksi': int(Cm.sum(0)[i, :, k].sum())}
            for m in ['precision', 'recall', 'f1']:
                lo, hi = ru.pct_ci(pb[m][:, i, k]); r[m], r[f'{m}_lo'], r[f'{m}_hi'] = float(pp[m][i, k]), lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def asc_per_class_ci(df, cols, B=C.B_BOOT, seed=C.STAT_SEED):
    ids, Cm = ru.cluster_confusions(df, cols)
    BC = ru.boot_cms(ru.bootstrap_weights(len(ids), B, seed), Cm).astype(float)
    tot = Cm.sum(0).astype(float)
    rows = []
    for i, c in enumerate(cols):
        for k in (1, 2, 3):
            def f(cm):
                tp = cm[..., k, k]; fn = cm[..., k, :].sum(-1) - tp; fp = cm[..., 1:, k].sum(-1) - tp
                return ru._safe_div(tp, tp + fp), ru._safe_div(tp, tp + fn), ru._safe_div(2 * tp, 2 * tp + fp + fn)
            pe, pb = f(tot[i]), f(BC[:, i])
            r = {'skema': c, 'kelas': CLASS_NAMES[k], 'support_gold': int(tot[i, k].sum())}
            for nm, e, b in zip(['precision', 'recall', 'f1'], pe, pb):
                lo, hi = ru.pct_ci(b); r[nm], r[f'{nm}_lo'], r[f'{nm}_hi'] = float(e), lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


# ── revision/01 §3: 15 perbandingan berpasangan ──
def pairwise(df, cols=SCHEMES6, metric='asc_f1', ref=REF, B_boot=C.B_BOOT, B_perm=C.B_PERM, seed=C.STAT_SEED, seed_offset=1000):
    ids, Cm = ru.cluster_confusions(df, cols)
    BC = ru.boot_cms(ru.bootstrap_weights(len(ids), B_boot, seed), Cm)
    bt, pt = ru.metrics_from_cm(BC), ru.metrics_from_cm(Cm.sum(0))
    f = lambda cm: ru.metrics_from_cm(cm)[metric]
    pairs = [(b, a) if b == ref else (a, b) for a, b in itertools.combinations(cols, 2)]
    rows = []
    for j, (a, b) in enumerate(pairs):
        ia, ib = cols.index(a), cols.index(b)
        d = bt[metric][:, ia] - bt[metric][:, ib]
        obs, p, _ = ru.cluster_permutation_test(Cm[:, ia], Cm[:, ib], f, B_perm, seed + seed_offset + j)
        lo, hi = ru.pct_ci(d)
        rows.append({'skema_A': a, 'skema_B': b, 'A': float(pt[metric][ia]), 'B': float(pt[metric][ib]),
                     'delta_observasi': obs, 'delta_rata2_bootstrap': float(d.mean()), 'ci_lo': lo, 'ci_hi': hi,
                     'p_permutasi': p})
    out = pd.DataFrame(rows)
    out['p_holm'] = ru.holm(out.p_permutasi)
    out['signifikan_holm_005'] = out.p_holm < C.ALPHA
    out['ci_tidak_memuat_0'] = (out.ci_lo > 0) | (out.ci_hi < 0)
    out['status'] = np.where(out.signifikan_holm_005, 'SIGNIFIKAN (p Holm < 0,05)', 'tidak signifikan setelah Holm')
    return out


# ── revision/01 §4: McNemar naskah + versi berkluster ──
def mcnemar_family(df, cols=SCHEMES6, pw_acc=None):
    from statsmodels.stats.contingency_tables import mcnemar
    yt = df.y_true.values
    rows = []
    for a, b in itertools.combinations(cols, 2):
        ca, cb = df[a].values == yt, df[b].values == yt
        n11, n10, n01, n00 = int((ca & cb).sum()), int((ca & ~cb).sum()), int((~ca & cb).sum()), int((~ca & ~cb).sum())
        p_exact = float(mcnemar([[n11, n10], [n01, n00]], exact=True).pvalue)
        dk = ru.durkalski_mcnemar(ca, cb, df.tweet_id.values)
        st = ru.tweet_level_sign_test(ca, cb, df.tweet_id.values)
        r = {'skema_A': a, 'skema_B': b, 'b_A_benar_B_salah': n10, 'c_A_salah_B_benar': n01, 'p_mcnemar_naskah': p_exact,
             'chi2_durkalski': dk['chi2'], 'p_durkalski': dk['p'], 'tweet_A_lebih_baik': st['tweets_A_better'],
             'tweet_B_lebih_baik': st['tweets_B_better'], 'p_tanda_tweet': st['p']}
        if pw_acc is not None:
            pa = pw_acc[((pw_acc.skema_A == a) & (pw_acc.skema_B == b)) | ((pw_acc.skema_A == b) & (pw_acc.skema_B == a))].iloc[0]
            r['p_permutasi_akurasi'] = pa.p_permutasi
        rows.append(r)
    mc = pd.DataFrame(rows)
    mc['p_holm_naskah'] = ru.holm(mc.p_mcnemar_naskah)
    mc['p_holm_durkalski'] = ru.holm(mc.p_durkalski)
    mc['p_holm_tanda_tweet'] = ru.holm(mc.p_tanda_tweet)
    if 'p_permutasi_akurasi' in mc:
        mc['p_holm_permutasi_akurasi'] = ru.holm(mc.p_permutasi_akurasi)
    return mc


def significance_table(pw_asc, pw_overall, mc):
    """Tabel 15: satu baris per perbandingan, ditandai jelas signifikan/tidak setelah Holm."""
    rows = []
    for _, r in pw_asc.iterrows():
        a, b = r.skema_A, r.skema_B
        o = pw_overall[(pw_overall.skema_A == a) & (pw_overall.skema_B == b)].iloc[0]
        m = mc[((mc.skema_A == a) & (mc.skema_B == b)) | ((mc.skema_A == b) & (mc.skema_B == a))].iloc[0]
        mark = lambda p: f'{p:.4f} ' + ('✔ SIGNIFIKAN' if p < C.ALPHA else '✘ tidak signifikan')
        rows.append({'perbandingan': f'{a} vs {b}', 'ΔASC F1 [CI 95%]': fmt_ci(r.delta_observasi, r.ci_lo, r.ci_hi),
                     'ASC: p perm': round(r.p_permutasi, 4), 'ASC: p Holm': mark(r.p_holm),
                     'ΔOverall F1 [CI 95%]': fmt_ci(o.delta_observasi, o.ci_lo, o.ci_hi), 'Overall: p Holm': mark(o.p_holm),
                     'Akurasi: McNemar naskah p Holm': mark(m.p_holm_naskah), 'Akurasi: McNemar berkluster p Holm': mark(m.p_holm_durkalski)})
    return pd.DataFrame(rows)


# ── revision/02: kasus ditandai ──
def subset_metrics(df, mask, label, cols=SCHEMES6, before='Weighted Soft Voting', after=REF, B=C.B_BOOT, seed=C.STAT_SEED):
    sub = df[mask]
    ids, Cm = ru.cluster_confusions(sub, cols)
    BC = ru.boot_cms(ru.bootstrap_weights(len(ids), B, seed), Cm)
    bt, pt = ru.metrics_from_cm(BC), ru.metrics_from_cm(Cm.sum(0))
    rows = []
    for i, c in enumerate(cols):
        r = {'subset': label, 'n_pasangan': len(sub), 'n_tweet': len(ids), 'n_gold_sentimen': int((sub.y_true != 0).sum()), 'skema': c}
        for k in METRICS:
            lo, hi = ru.pct_ci(bt[k][:, i]); r[k], r[f'{k}_lo'], r[f'{k}_hi'] = float(pt[k][i]), lo, hi
        rows.append(r)
    ia, ib = cols.index(after), cols.index(before)
    delta = {'subset': label}
    for k in METRICS:
        lo, hi = ru.pct_ci(bt[k][:, ia] - bt[k][:, ib])
        delta[f'delta_{k}'] = float(pt[k][ia] - pt[k][ib]); delta[f'{k}_lo'] = lo; delta[f'{k}_hi'] = hi
    return pd.DataFrame(rows), delta


def transitions(df, before='Weighted Soft Voting', after=REF):
    d = df.assign(soft_benar=df[before] == df.y_true, final_benar=df[after] == df.y_true,
                  kelompok=np.where(~df.flagged, 'auto_accept', df.decision))

    def tr(g):
        sb, fb, ch = g.soft_benar, g.final_benar, g[after] != g[before]
        return pd.Series({'n': len(g), 'benar→benar (tetap)': int((sb & fb).sum()), 'salah→benar (diperbaiki)': int((~sb & fb).sum()),
                          'benar→salah (dirusak)': int((sb & ~fb).sum()), 'salah→salah, label diubah': int((~sb & ~fb & ch).sum()),
                          'salah→salah, label tetap': int((~sb & ~fb & ~ch).sum()), 'akurasi sebelum': sb.mean(), 'akurasi sesudah': fb.mean()})
    t = d.groupby('kelompok').apply(tr)
    t.loc['ditandai (retain+revise)'] = tr(d[d.flagged])
    t.loc['semua'] = tr(d)
    order = [x for x in ['retain', 'revise', 'ditandai (retain+revise)', 'auto_accept', 'semua'] if x in t.index]
    return t.loc[order]


def flagged_tests(df, before='Weighted Soft Voting', after=REF, B=C.B_PERM, seed=C.STAT_SEED):
    from statsmodels.stats.contingency_tables import mcnemar
    sub = df[df.flagged]
    ca, cb = (sub[after] == sub.y_true).values, (sub[before] == sub.y_true).values
    tab = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
    rows = [{'uji': 'McNemar eksak (akurasi)', 'delta': ca.mean() - cb.mean(), 'p': float(mcnemar(tab, exact=True).pvalue)},
            {'uji': 'McNemar berkluster Durkalski (akurasi)', 'delta': ca.mean() - cb.mean(), 'p': ru.durkalski_mcnemar(ca, cb, sub.tweet_id.values)['p']}]
    ids, Cm = ru.cluster_confusions(sub, [after, before])
    for k in ['accuracy', 'asc_f1', 'overall_f1']:
        obs, p, _ = ru.cluster_permutation_test(Cm[:, 0], Cm[:, 1], lambda cm, k=k: ru.metrics_from_cm(cm)[k], B, seed + 7)
        rows.append({'uji': f'permutasi cluster ({k})', 'delta': obs, 'p': p})
    return pd.DataFrame(rows)


# ── revision/06: hybrid vs LLM-only / LLM-always ──
def compare_vs(df, cols, ref, subsets, B=C.B_BOOT, seed=C.STAT_SEED):
    rows = []
    for si, (label, mask) in enumerate(subsets):
        sub = df[mask]
        ids, Cm = ru.cluster_confusions(sub, cols)
        BC = ru.boot_cms(ru.bootstrap_weights(len(ids), B, seed + 100 * si), Cm)
        bt, pt = ru.metrics_from_cm(BC), ru.metrics_from_cm(Cm.sum(0))
        ir = cols.index(ref)
        for i, c in enumerate(cols):
            for k in METRICS:
                lo, hi = ru.pct_ci(bt[k][:, i])
                r = {'subset': label, 'n_pasangan': len(sub), 'skema': c, 'metrik': k, 'nilai': float(pt[k][i]), 'lo': lo, 'hi': hi}
                if c != ref:
                    dlo, dhi = ru.pct_ci(bt[k][:, i] - bt[k][:, ir])
                    obs, p, _ = ru.cluster_permutation_test(Cm[:, i], Cm[:, ir], lambda cm, k=k: ru.metrics_from_cm(cm)[k],
                                                            B, seed + 100 * si + 17 + i)
                    r.update({'delta_vs_ref': obs, 'delta_lo': dlo, 'delta_hi': dhi, 'p_permutasi': p})
                rows.append(r)
    return pd.DataFrame(rows)


# ── R3_07: router alternatif ──
def router_rankings(df, ens_probs, indobert_probs):
    """Urutan prioritas routing (indeks posisi df). Seri dipecah oleh urutan baris (stabil)."""
    n = len(df)
    pos = np.arange(n)
    split = (df.vote_type == 'terpecah').values
    margin = df.soft_vote_margin.values
    p = np.clip(ens_probs, 1e-12, 1)
    entropy = -(p * np.log(p)).sum(1)
    return {
        '(a) router utama (vote terpecah, lalu margin terkecil)': np.lexsort((pos, margin, ~split)),
        '(b) confidence IndoBERT terendah': np.lexsort((pos, indobert_probs.max(1))),
        '(c) confidence soft voting terendah': np.lexsort((pos, ens_probs.max(1))),
        '(d) entropi soft voting tertinggi': np.lexsort((pos, -entropy)),
    }


def router_eval(df, rankings, budgets, llm_col='LLM-always', base_col='Weighted Soft Voting', B=C.B_BOOT, seed=C.STAT_SEED):
    rows, preds = [], {}
    for budget in budgets:
        for name, order in rankings.items():
            routed = np.zeros(len(df), bool); routed[order[:budget]] = True
            y = np.where(routed, df[llm_col].values, df[base_col].values)
            col = f'{name} | anggaran {budget}'
            preds[col] = y
            soft_ok, llm_ok = df[base_col].values == df.y_true.values, df[llm_col].values == df.y_true.values
            rows.append({'anggaran': budget, 'router': name, 'n_dirouting': int(routed.sum()),
                         'kesalahan_soft_tertangkap': int((routed & ~soft_ok).sum()),
                         'diperbaiki_LLM': int((routed & ~soft_ok & llm_ok).sum()),
                         'dirusak_LLM': int((routed & soft_ok & ~llm_ok).sum()), 'col': col})
    tab = pd.DataFrame(rows)
    frame = df[['tweet_id', 'y_true']].copy()
    for c, y in preds.items():
        frame[c] = y
    met, _ = metrics_ci(frame, list(preds), B=B, seed=seed)
    tab = tab.merge(met.rename(columns={'skema': 'col'}), on='col').drop(columns=['n_pasangan', 'n_tweet'])
    return tab.drop(columns='col')


# ── Frame evaluasi bersama R3_06/R3_07 ──
def build_eval_frame(D, frozen):
    """Gabungkan label gold (dev bila DRY_RUN, uji bila nyata) dengan semua prediksi R3_04 dan label LLM R3_05."""
    import json as _json
    P, LL = D.test_pred, D.llm
    gold = pd.read_csv(D.pairs, dtype={'tweet_id': str})
    gold['pair_id'] = gold['pair_id'].astype(str)
    df = gold[['pair_id', 'tweet_id', 'sentence_a', 'label']].rename(columns={'label': 'y_true'})
    df['aspek'] = df['pair_id'].map(lambda p: next(a for a in C.ASPECTS if p.endswith('_' + a)))
    nice = {'indobert': 'IndoBERT', 'xlm_roberta': 'XLM-RoBERTa', 'mbert': 'mBERT'}
    probs = {}
    for m, n in nice.items():
        p = pd.read_csv(P / f'{m}_test_avg.csv'); p['pair_id'] = p['pair_id'].astype(str)
        p = df[['pair_id']].merge(p, on='pair_id', how='left')
        df[n] = p['pred_label'].astype(int).values
        probs[m] = p[C.PROB_COLS].values
    ens = pd.read_csv(P / 'ensemble_test.csv'); ens['pair_id'] = ens['pair_id'].astype(str)
    ens = df[['pair_id']].merge(ens, on='pair_id', how='left')
    df['Hard Voting'], df['Weighted Soft Voting'] = ens.hard_vote_label.astype(int).values, ens.soft_vote_label.astype(int).values
    for c in ['vote_type', 'soft_vote_margin', 'pred_indobert', 'pred_xlm_roberta', 'pred_mbert']:
        df[c] = ens[c].values
    probs['ensemble'] = np.array([_json.loads(x) for x in ens.prob_ensemble_json])
    fl = pd.read_csv(P / 'flags.csv'); fl['pair_id'] = fl['pair_id'].astype(str)
    df['flagged'] = df[['pair_id']].merge(fl, on='pair_id', how='left').needs_llm.astype(bool).values
    assert (df.flagged == ((df.vote_type == 'terpecah') | (df.soft_vote_margin < C.MARGIN_THRESHOLD))).all()
    fin = pd.read_csv(LL / 'final_labels_test.csv'); fin['pair_id'] = fin['pair_id'].astype(str)
    fin = df[['pair_id']].merge(fin, on='pair_id', how='left')
    assert fin.final_label_int.notna().all(), 'label akhir hybrid belum lengkap (R3_05)'
    df['Soft Voting + LLM'] = fin.final_label_int.astype(int).values
    df['decision'] = fin.decision.values
    for name, f in [('LLM-always', 'llm_always_test.csv'), ('LLM-only', 'llm_only_test.csv')]:
        t = pd.read_csv(LL / f); t['pair_id'] = t['pair_id'].astype(str)
        t = df[['pair_id']].merge(t, on='pair_id', how='left')
        assert t.final_label.notna().all(), f'{name} belum lengkap (R3_05)'
        df[name] = t.final_label.map(C.LABEL_MAP).astype(int).values
    t17 = pd.read_csv(P / 'table17_selected_test.csv'); t17['pair_id'] = t17['pair_id'].astype(str)
    df['Tabel 17 terpilih'] = df[['pair_id']].merge(t17, on='pair_id', how='left').pred_label.astype(int).values
    for m, n in nice.items():
        for v in C.TEXT_VERSIONS_ABLATION:
            a = pd.read_csv(P / 'ablation' / f'{m}_{v}_test_avg.csv'); a['pair_id'] = a['pair_id'].astype(str)
            df[f'{n} {v.upper()}'] = df[['pair_id']].merge(a, on='pair_id', how='left').pred_label.astype(int).values
    assert df[SCHEMES6].notna().all().all()
    return df, probs
