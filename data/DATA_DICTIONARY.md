# Data dictionary

Tweet texts are **not** redistributed. Every file is keyed by `tweet_id` (X/Twitter status ID, stored as text to avoid
precision loss) and `pair_id` = `<tweet_id>_<aspek>`. See the README for rehydrating texts from IDs.

## Labels and aspects (used in every file)

| Label | Integer id | Meaning |
|---|---|---|
| `tidak_disebutkan` | 0 | aspect not mentioned in the tweet |
| `positif` | 1 | positive sentiment toward the aspect |
| `negatif` | 2 | negative sentiment toward the aspect |
| `netral` | 3 | aspect mentioned, neutral sentiment |

| `aspek` | English name | Description used as `sentence_b` / in prompts |
|---|---|---|
| `pola_konsumsi` | Consumption pattern | Tweet ini membahas pola konsumsi makanan atau minuman |
| `aktivitas_fisik` | Physical activity | Tweet ini membahas aktivitas fisik atau olahraga |
| `akses_layanan_kesehatan` | Healthcare access | Tweet ini membahas akses terhadap layanan kesehatan |
| `pengetahuan_pengobatan` | Treatment knowledge | Tweet ini membahas pengetahuan atau pengobatan penyakit |

Each tweet yields exactly four aspect pairs. Metrics: ACD = aspect-category detection (mentioned vs `tidak_disebutkan`),
ASC = sentiment classification among mentioned aspects, Overall = 4-class macro-F1.

---

## `data/gold_labels.csv` — gold standard (200 tweets, 800 pairs)

| Column | Description |
|---|---|
| `tweet_id` | X status ID |
| `pair_id` | `<tweet_id>_<aspek>` |
| `aspek` | aspect key |
| `split` | `validation` (60 tweets / 240 pairs; used for all model and threshold choices) or `test` (140 tweets / 560 pairs; used once for evaluation) |
| `label_gold` | gold label (annotator 1, reviewed) |

Label counts — validation: 164 / 21 / 17 / 38 (`tidak_disebutkan` / `positif` / `negatif` / `netral`);
test: 381 / 50 / 39 / 90. Split: `StratifiedGroupKFold` (10 folds, shuffled, seed 42) grouped by tweet; 3 folds validation, 7 folds test.
Gold tweets are excluded from the silver training set.

## `data/second_annotator.csv` — inter-annotator agreement sample (40 test tweets, 160 pairs)

| Column | Description |
|---|---|
| `tweet_id`, `pair_id`, `aspek` | as above |
| `label_annotator2` | label assigned independently by the second annotator |

Agreement with `label_gold` on the same pairs: 91.9% raw agreement, Cohen's κ = 0.849 (95% CI 0.777–0.913, cluster bootstrap over tweets).

## `data/silver_labels.csv` — silver training labels (37,747 tweets, 150,988 pairs)

| Column | Description |
|---|---|
| `tweet_id`, `pair_id`, `aspek` | as above |
| `label_silver` | lexicon-based silver label used to fine-tune the encoders |
| `lexicon_score` | sentence-level InSet lexicon score of `text_v3` (0 when the aspect is not detected) |

Rule (frozen before the test set was accessed): aspect detected by keyword match on the original text
(`resources/aspect_keywords.json`); if not detected → `tidak_disebutkan`; otherwise `score_sentence` on `text_v3` with
threshold θ = 15 (score > θ → `positif`, score < −θ → `negatif`, otherwise `netral`). θ and the scoring method were chosen on 76
sentiment-bearing validation pairs only (`outputs/theta_final_report.md`, `outputs/tables/silver_calibration/`).
Distribution: 93,203 / 4,758 / 4,812 / 48,215 (`tidak_disebutkan` / `positif` / `negatif` / `netral`).

---

## `outputs/predictions_test.csv` — encoder and ensemble predictions (560 test pairs)

| Column | Description |
|---|---|
| `pair_id`, `tweet_id`, `aspek` | as above |
| `pred_<model>` | predicted label of `indobert`, `xlm_roberta`, `mbert`: argmax of the softmax probabilities averaged over seeds 42, 123 and 7 |
| `prob_<label>_<model>` | seed-averaged probability of each label (`prob_tidak_disebutkan`, `prob_positif`, `prob_negatif`, `prob_netral`) |
| `hard_vote_label` | majority vote of the three `pred_<model>` (three-way tie → smallest label id) |
| `weighted_soft_vote_label` | argmax of the weighted average of the three probability vectors; weights = per-aspect validation 4-class macro-F1 of each model (seed-averaged), normalised per aspect (`outputs/frozen_choices.json`, `model_weights`) |
| `prob_<label>_weighted_soft_vote` | weighted ensemble probabilities |
| `soft_vote_margin` | difference between the two largest weighted ensemble probabilities |
| `vote_type` | `bulat` = unanimous (all three `pred_<model>` equal); `terpecah` = split vote |
| `flagged` | `True` if the pair is forwarded to the LLM judge: `vote_type == terpecah` **or** `soft_vote_margin < 0.15` (57 pairs) |

## `outputs/llm_arbitration.csv` — final labels of Soft Voting + LLM-as-a-Judge (560 test pairs)

| Column | Description |
|---|---|
| `pair_id`, `tweet_id`, `aspek` | as above |
| `weighted_soft_vote_label` | label before arbitration |
| `flagged` | forwarded to the LLM judge (57 pairs) |
| `decision` | `auto_accept` (not flagged; soft-vote label kept, no LLM call), `retain` or `revise` (LLM decision) |
| `final_label` | final label of the hybrid system (for flagged pairs: the parsed `final_label` of the LLM response) |
| `reasoning` | one-sentence rationale returned by the LLM (Indonesian, verbatim except as noted below); empty for `auto_accept` |
| `status` | `ok` (parsed LLM response) or `auto_accept` |
| `llm_model` | model snapshot (flagged pairs only) |
| `prompt_tokens`, `completion_tokens` | token usage reported by the API |
| `latency_s` | wall-clock latency of the API call (seconds) |
| `finish_reason` | API finish reason |
| `n_attempts` | number of attempts needed (maximum 3) |

## `outputs/llm_always.csv` — LLM-always ablation (arbitration prompt for all 560 pairs)

`pair_id`, `tweet_id`, `aspek`, `decision` (`retain`/`revise`), `final_label`, `reasoning`, `status`, `llm_model`.
For the 57 flagged pairs the prompt is identical to the arbitration prompt and the same cached response is reused.

## `outputs/llm_only.csv` — LLM-only baseline (no ensemble information; 560 pairs)

`pair_id`, `tweet_id`, `aspek`, `final_label`, `reasoning`, `status`, `llm_model`.

## `outputs/llm_only_qwen3max_sensitivity.csv` — LLM-only with the retired snapshot (sensitivity only)

Same columns as `llm_only.csv`; `llm_model` = `qwen3-max-2026-01-23`. Not used by any main result (see `prompts/PROMPTS.md` §8).

### Note on `reasoning` redaction

To comply with the no-text-redistribution policy, in **3 of the 1,737** rationales a span that quoted more than five
consecutive words of the tweet verbatim was replaced with `[…]`; the rest of each rationale is unchanged. All affected rows are
in the LLM-only files (`llm_only.csv`: 1; `llm_only_qwen3max_sensitivity.csv`: 2); `llm_arbitration.csv` and `llm_always.csv` are
unaffected. No label, decision or metric is affected. The redaction never touches the labels used in the paper.

---

## `outputs/tables/` and `outputs/figures/`

Copies (byte-identical) of the round-3 evaluation outputs. Column names are in Indonesian:
`skema` = scheme, `akurasi` = accuracy, `estimasi` = estimate, `lo`/`hi` = 95% CI bounds (cluster bootstrap over tweets,
B = 10,000, seed 42), `p_permutasi` = cluster permutation p-value, `p Holm` = Holm-adjusted p-value (15 comparisons),
`panggilan` = API calls, `latensi` = latency, `biaya_USD` = estimated cost, `gagal` = failed calls,
`terpecah`/`bulat` = split/unanimous, `ditandai` = flagged, `pasangan` = pairs.

| Folder | Content |
|---|---|
| `tables/evaluation/` | main metrics with CIs (`T_utama_enam_skema*`), significance (`T15_signifikansi`, `T_mcnemar`, `T_berpasangan_*`), per class/aspect, flagged-case analysis, transitions, revision directions, LLM baselines (`T_hybrid_vs_LLM`, `T_always_vs_only`), API usage, preprocessing ablation (`T16`), Table 17 scheme, silver audit (`T6`), LLM snapshot sensitivity (`T_sensitivitas_*`) |
| `tables/router_analysis/` | comparison of alternative routing rules |
| `tables/dev_choices/` | validation-set choices: model disagreement, preprocessing ablation, Table 17 grid |
| `tables/silver_calibration/` | θ / scoring-method calibration on validation data |
| `tables/llm/usage_summary.csv` | calls, tokens, latency and cost per LLM condition |
| `figures/evaluation/`, `figures/router_analysis/` | figures produced by the evaluation notebooks (Indonesian labels) |
| `figures/manuscript_en/` | Figures 2–4 of the article (English labels, 300 dpi PNG and PDF), drawn from the same outputs |

`outputs/frozen_choices.json` records every choice made on validation data before test access (weights, selected schemes,
data/code hashes, environment); absolute paths were made relative to the project root. `outputs/rencana_latih_ulang.md` is the
frozen protocol (Indonesian) with its change log; `outputs/theta_final_report.md` documents the silver-threshold calibration.
