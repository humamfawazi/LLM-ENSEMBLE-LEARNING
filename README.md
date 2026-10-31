# When Ensembles Disagree: Selective LLM-as-a-Judge Arbitration for ABSA of Indonesian Diabetes Tweets

Companion repository for the article *"When Ensembles Disagree: Selective LLM-as-a-Judge Arbitration for Aspect-Based
Sentiment Analysis of Indonesian Diabetes Tweets"* ([CITATION — to be added on publication](#citation)).

It contains the code, the LLM prompt templates and parsing rules, and the per-pair outputs of every model and of the LLM judge
(decisions, labels and rationales), together with tweet IDs and gold and silver labels. **Tweet texts are not redistributed**
(see [Tweet texts](#tweet-texts-and-rehydration)).

## Study in brief

* **Task.** Aspect-based sentiment analysis of Indonesian tweets about diabetes. Each tweet is paired with four aspects:
  consumption pattern, physical activity, healthcare access and treatment knowledge. Each pair gets one of four labels:
  not mentioned, positive, negative or neutral.
* **Data.** 37,747 tweets with lexicon-based silver labels (150,988 pairs) for training. 200 gold-annotated tweets are split into
  a validation set (60 tweets / 240 pairs, used for every choice) and a test set (140 tweets / 560 pairs, used once).
  A second annotator labelled 40 test tweets (κ = 0.849).
* **Silver labels.** InSet lexicon, sentence-level score, threshold θ = 15. The scoring method and θ were selected on validation
  data only and frozen before the test set was accessed.
* **Encoders.** IndoBERT (`indobenchmark/indobert-base-p1`), XLM-RoBERTa (`xlm-roberta-base`) and mBERT
  (`bert-base-multilingual-cased`), fine-tuned as sentence-pair classifiers (tweet, aspect description). Each was trained with
  3 seeds (42, 123, 7) and probabilities were averaged over the seeds.
* **Ensembles.** Hard voting and weighted soft voting. Weights are the per-aspect validation macro-F1 of each encoder.
* **Selective arbitration.** A pair is forwarded to an LLM judge (`qwen3.7-max-2026-06-08`, temperature 0) only when the three
  encoders disagree (split vote) or the soft-voting margin is below 0.15. The judge retains or revises the ensemble label.
  All other pairs are auto-accepted.
* **Baselines.** LLM-only (the LLM labels every pair without ensemble information) and LLM-always (the arbitration prompt for
  every pair).

### Main results (test set, 560 pairs / 140 tweets)

Point estimates; 95% cluster-bootstrap CIs and significance tests are in `outputs/tables/evaluation/`
(`T_utama_enam_skema_ci.csv`, `T15_signifikansi.csv`, `T_hybrid_vs_LLM.csv`).

| Scheme | ACD macro-F1 | ASC macro-F1 | Overall macro-F1 | Accuracy |
|---|---|---|---|---|
| IndoBERT | 0.8598 | 0.2871 | 0.4117 | 0.7268 |
| XLM-RoBERTa | 0.8656 | 0.2310 | 0.3750 | 0.7268 |
| mBERT | 0.8595 | 0.2614 | 0.3958 | 0.7214 |
| Hard Voting | 0.8616 | 0.2533 | 0.3888 | 0.7250 |
| Weighted Soft Voting | 0.8620 | 0.2624 | 0.3949 | 0.7232 |
| Soft Voting + LLM-as-a-Judge (57 pairs sent to the LLM) | 0.8969 | 0.3521 | 0.4779 | 0.7750 |
| LLM-only (560 pairs) | 0.8848 | 0.6318 | 0.6506 | 0.8000 |
| LLM-always (560 pairs) | 0.9191 | 0.5673 | 0.6419 | 0.8286 |

## Repository structure

```
README.md                 this file
LICENSE                   MIT (code)
requirements.txt          library versions used to produce the outputs
prompts/PROMPTS.md        system prompt, arbitration and LLM-only templates (Indonesian, verbatim + English), aspect
                          definitions, labels, JSON schema, parsing, retry/fallback, parameters, synthetic example
code/
  R3_00 … R3_07 *.ipynb   pipeline notebooks (outputs cleared), run in order
  R3_02_run_all.py        runs the 9 training runs of R3_02 sequentially (jalankan_R3_02_semua.bat on Windows)
  config.py               single configuration file (frozen values)
  r3_lib.py, r3_eval.py   shared library and evaluation functions
  revision/revision_utils.py   statistics (cluster bootstrap, permutation tests, McNemar, Holm)
  legacy_code/            verbatim copies of the original notebook cells reused by the pipeline (+ sha256 manifest)
  preprocessing/text_versions.py   text normalisation V1–V6 (verbatim from the original preprocessing notebook)
  calibration/            validation-only calibration of the silver-label scoring method and θ
  README_RUN_ORDER.md     detailed run notes (Indonesian)
resources/aspect_keywords.json   aspect keyword lists for aspect detection in silver labelling
data/
  gold_labels.csv         gold labels, 800 pairs, with validation/test split
  second_annotator.csv    second-annotator labels, 160 pairs (40 test tweets)
  silver_labels.csv       silver training labels (θ = 15), 150,988 pairs, with lexicon scores
  DATA_DICTIONARY.md      column-by-column description of every data and output file
outputs/
  predictions_test.csv    per-pair predictions and seed-averaged probabilities of the three encoders, hard vote,
                          weighted soft vote, margin, vote type, routing flag
  llm_arbitration.csv     final hybrid labels: LLM decision, label, rationale, tokens, latency (560 pairs; 57 sent to the LLM)
  llm_always.csv, llm_only.csv          LLM baselines (560 pairs each)
  llm_only_qwen3max_sensitivity.csv     LLM-only with the retired snapshot (sensitivity analysis only)
  tables/                 all evaluation tables (CSV) of this study
  figures/                evaluation figures; figures/manuscript_en/ = Figures 2–4 of the article
  frozen_choices.json     every validation-based choice, recorded before test access (weights, schemes, hashes, environment)
  R3_01_manifest.json     silver-label run record (method, θ, distribution, input/output hashes)
  rencana_latih_ulang.md  frozen analysis protocol with change log (Indonesian)
  theta_final_report.md   silver-threshold calibration report (Indonesian)
```

All identifiers join on `pair_id` (`<tweet_id>_<aspect>`). Labels are stored with their Indonesian names
(`tidak_disebutkan`, `positif`, `negatif`, `netral`); see `data/DATA_DICTIONARY.md`.

## Re-analysing the released outputs (no tweet text needed)

Every metric can be recomputed from the released labels and outputs. For example:

```python
import pandas as pd
gold = pd.read_csv('data/gold_labels.csv', dtype=str).query("split == 'test'")
hyb = pd.read_csv('outputs/llm_arbitration.csv', dtype=str)
df = gold.merge(hyb[['pair_id', 'final_label']], on='pair_id')
print((df.label_gold == df.final_label).mean())          # accuracy of Soft Voting + LLM-as-a-Judge: 0.775
```

## Tweet texts and rehydration

Under the X (Twitter) developer terms only tweet IDs are shared. To obtain the texts:

1. Collect the IDs: `tweet_id` in `data/silver_labels.csv` (37,747 training tweets) and `data/gold_labels.csv` (200 gold tweets).
2. Retrieve the tweets with the X API v2 tweet lookup endpoint (`GET /2/tweets?ids=…`, up to 100 IDs per request; requires an
   X developer account), or with a hydration tool such as `twarc2 hydrate ids.txt tweets.jsonl`. Deleted, suspended or protected
   tweets cannot be retrieved, so a rehydrated corpus may be smaller than the original and results may differ slightly.
3. Rebuild the text versions with the original normalisation functions (`code/preprocessing/text_versions.py`;
   `text_v1` = URLs and @mentions removed, whitespace collapsed; `text_v1` is the model input and the text shown to the LLM).
4. Write the input files expected by `code/config.py`. With `tweets.csv` holding `tweet_id,text`:

```python
import sys, pandas as pd
sys.path.insert(0, 'code/preprocessing'); import text_versions as tv
from pathlib import Path
ASPECT_DESC = {'pola_konsumsi': 'Tweet ini membahas pola konsumsi makanan atau minuman',
               'aktivitas_fisik': 'Tweet ini membahas aktivitas fisik atau olahraga',
               'akses_layanan_kesehatan': 'Tweet ini membahas akses terhadap layanan kesehatan',
               'pengetahuan_pengobatan': 'Tweet ini membahas pengetahuan atau pengobatan penyakit'}
LABEL = {'tidak_disebutkan': 0, 'positif': 1, 'negatif': 2, 'netral': 3}
for d in ['data/processed', 'data/gold', 'data/pairs', 'data_round3']:
    Path(d).mkdir(parents=True, exist_ok=True)

tw = pd.read_csv('tweets.csv', dtype={'tweet_id': str})
for i, f in enumerate([tv.preprocess_v1, tv.preprocess_v2, tv.preprocess_v3, tv.preprocess_v4, tv.preprocess_v5, tv.preprocess_v6], 1):
    tw[f'text_v{i}'] = tw.text.map(f)
tw['char_length'], tw['word_count'] = tw.text.str.len(), tw.text.str.split().str.len()
tw[['tweet_id', 'text', 'char_length', 'word_count'] + [f'text_v{i}' for i in range(1, 7)]].to_csv(
    'data/processed/tweets_preprocessed.csv', index=False, encoding='utf-8-sig')
v1 = tw.set_index('tweet_id').text_v1

def pairs(df, label_col):
    df = df[df.tweet_id.isin(v1.index)]
    return pd.DataFrame({'pair_id': df.pair_id, 'tweet_id': df.tweet_id, 'sentence_a': df.tweet_id.map(v1),
                         'sentence_b': df.aspek.map(ASPECT_DESC), 'label': df[label_col].map(LABEL)})

gold = pd.read_csv('data/gold_labels.csv', dtype=str)
gold[gold.tweet_id.isin(v1.index)].assign(text=lambda d: d.tweet_id.map(v1))[['tweet_id', 'text', 'aspek', 'label_gold']].to_csv(
    'data/gold/gold_labeled.csv', index=False)
gold[['tweet_id']].drop_duplicates().to_csv('data/gold/gold_tweet_ids.csv', index=False)
for split, name in [('validation', 'dev_pairs'), ('test', 'test_pairs')]:
    pairs(gold[gold.split == split], 'label_gold').to_csv(f'data/pairs/{name}.csv', index=False)

silver = pd.read_csv('data/silver_labels.csv', dtype={'tweet_id': str})   # released θ = 15 silver labels
silver = silver[silver.tweet_id.isin(v1.index)]
silver.assign(text=silver.tweet_id.map(tw.set_index('tweet_id').text))[
    ['tweet_id', 'text', 'aspek', 'label_silver', 'lexicon_score']].to_csv('data_round3/silver_labeled_s15.csv', index=False)
pairs(silver, 'label_silver').to_csv('data_round3/train_pairs.csv', index=False)

import hashlib, json                                     # R3_03 checks training-file hashes against this manifest
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump({'outputs': {f: sha(f'data_round3/{f}') for f in ['silver_labeled_s15.csv', 'train_pairs.csv']}},
          open('data_round3/R3_01_manifest.json', 'w'), indent=2)
```

## Reproducing the pipeline

1. Python 3.12 and `pip install -r requirements.txt` (PyTorch 2.6.0 with CUDA 12.4; see the comment in `requirements.txt`).
2. Download the InSet lexicon (Koto & Rahmaningtyas, 2017; https://github.com/fajri91/InSet) and place `positive.tsv` and
   `negative.tsv` in `resources/inset_lexicon/`.
3. Rebuild the input files from rehydrated tweets (previous section).
4. Environment variables for the LLM step: `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL` (your OpenAI-compatible DashScope
   endpoint). Optional: `R3_BASE_DIR` (project root, default = repository root) and `R3_OUTPUT_ROOT` (default `outputs_round3/`).
5. Run the notebooks in `code/` in order (all outputs go to `outputs_round3/` and `data_round3/`):

| # | Notebook | Purpose | Notes |
|---|---|---|---|
| — | `calibration/calibrate_method_theta_validation.ipynb` | choose silver scoring method and θ on validation pairs | documentation of the choice. It executes cells of the original labelling notebook (`notebooks/02_Silverrr_Labeling_dan_SentencePair.ipynb`, not distributed; the same cells are in `legacy_code/nb02_*.py`) and expects the original folder layout. |
| 0 | `R3_00_config.ipynb` | checks inputs and the `legacy_code/` copies | comparison with the original notebooks is skipped when they are absent |
| 1 | `R3_01_silver_labels.ipynb` | silver labels θ = 15 and training pairs | asserts the original corpus size and compares with the original θ = 3 labels (not distributed). On a rehydrated corpus, skip it: the snippet above writes its three outputs from `data/silver_labels.csv`. The manifest of the original run is in `outputs/R3_01_manifest.json`. |
| 2 | `R3_02_train_one_run.ipynb` × 9 | fine-tune 3 encoders × 3 seeds | or `python R3_02_run_all.py`; about 57–62 h on an RTX 4050 Laptop (6 GB) |
| 3 | `R3_03_dev_choices_and_freeze.ipynb` | validation-only choices; writes `frozen_choices.json` | |
| 4 | `R3_04_test_inference.ipynb` | test predictions, ensembles, routing flags | set `DRY_RUN = False`; runs once |
| 5 | `R3_05_llm_calls.ipynb` | arbitration, LLM-always, LLM-only | `RUN_API=False` first (cost estimate), then `RUN_API=True`; responses are cached and API errors stop the run without fallback |
| 6 | `R3_06_evaluation.ipynb` | all tables and figures | set `DRY_RUN = False`; runs once |
| 7 | `R3_07_router_analysis.ipynb` | alternative routing rules | |

Notebooks R3_04–R3_07 default to `DRY_RUN = True`. That mode runs the whole flow on validation data (with mock LLM labels) to test
the code. Its numbers are not results.

**The LLM snapshot.** The pre-registered snapshot `qwen3-max-2026-01-23` was retired by the provider before the test-set LLM
calls of this round. All LLM results use `qwen3.7-max-2026-06-08` (see `prompts/PROMPTS.md` §8). Snapshots can be retired at any
time, so exact LLM outputs may not be reproducible later. The released rationales and labels are the record of this run.

### Differences between this release and the code that produced the outputs

The pipeline notebooks are released with their outputs cleared; their code cells are unchanged. The code hashes of R3_04–R3_07
equal those recorded in `outputs/frozen_choices.json`. Only these files were edited for publication:

* `code/config.py`: the default project root is the repository root (originally two folders above the code), and
  `revision_utils.py` is read from `code/revision/`. All frozen values are unchanged.
* `code/r3_lib.py`: `nb06_base_url()` reads the endpoint from `DASHSCOPE_BASE_URL`. Originally it read the URL from the original
  LLM notebook, which is not distributed.
* `code/R3_02_run_all.py`, `code/jalankan_R3_02_semua.bat`, `code/README_RUN_ORDER.md`: local absolute paths replaced by `python`
  and `<REPO>`.
* `outputs/frozen_choices.json`: absolute paths made relative to the project root.

Because `config.py` and `r3_lib.py` changed, `verify_frozen()` reports these two files as modified against the released
`frozen_choices.json`. A fresh run creates its own `frozen_choices.json` in R3_03.

## Licence

* **Code** (`code/`, `prompts/`): MIT License (`LICENSE`).
* **Annotations and outputs** (`data/`, `outputs/`): Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Tweets**: content belongs to its authors and is subject to the X developer terms; only IDs are shared.
* **InSet lexicon**: not redistributed; obtain it from its authors' repository and follow its terms.

## Citation

```bibtex
@article{CITATION_KEY,
  title   = {When Ensembles Disagree: Selective LLM-as-a-Judge Arbitration for Aspect-Based Sentiment Analysis of Indonesian Diabetes Tweets},
  author  = {[HUMAM NASYWA FAWAZI], [FEDDY SETIO PRIBADI]},
  journal = {[Journal of Technology Informatics and Engineering (JTIE)]},
  year    = {[2026]},
  doi     = {[DOI]}
}
```
