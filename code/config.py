"""Konfigurasi tunggal latih ulang ronde 3 (nilai beku dari revisi_round3/rencana_latih_ulang.md).

Semua notebook R3_* meng-import file ini. JANGAN mengubah nilai beku setelah R3_03 menulis frozen_choices.json.

Mengganti lokasi (laptop / Kaggle / Colab) — cukup salah satu:
  * set environment variable R3_BASE_DIR ke folder proyek (yang berisi data/, revisi_round3/, …), atau
  * ubah BASE_DIR_DEFAULT di bawah.
Folder output dapat dipindah dengan R3_OUTPUT_ROOT (mis. ke Google Drive di Colab agar resume bertahan).
"""
from __future__ import annotations

import os
from pathlib import Path

# Sama dengan sel konfigurasi NB03: harus diset sebelum torch dimuat (config di-import sebelum torch di semua notebook R3)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ─────────────────────────────────────────────────────────────
# 1. Lokasi
# ─────────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent                       # code/
BASE_DIR_DEFAULT = CODE_DIR.parent                               # akar repositori (rilis: code/ langsung di bawah akar)
BASE_DIR = Path(os.environ.get('R3_BASE_DIR', BASE_DIR_DEFAULT)).resolve()

# Input lama (hanya dibaca)
DATA_OLD = BASE_DIR / 'data'
TWEETS_PREPROCESSED = DATA_OLD / 'processed' / 'tweets_preprocessed.csv'
GOLD_LABELED = DATA_OLD / 'gold' / 'gold_labeled.csv'
GOLD_TWEET_IDS = DATA_OLD / 'gold' / 'gold_tweet_ids.csv'
DEV_PAIRS = DATA_OLD / 'pairs' / 'dev_pairs.csv'
TEST_PAIRS = DATA_OLD / 'pairs' / 'test_pairs.csv'               # hanya R3_04 (tanpa label) dan R3_06 (label)
SILVER_OLD = DATA_OLD / 'silver' / 'silver_labeled.csv'
TRAIN_PAIRS_OLD = DATA_OLD / 'pairs' / 'train_pairs.csv'
RESOURCES = BASE_DIR / 'resources'
OLD_NOTEBOOKS = BASE_DIR / 'notebooks'
REVISION_DIR = CODE_DIR / 'revision'                           # revision_utils.py (statistik)
LLM_CACHE_OLD = REVISION_DIR / 'outputs' / 'llm_cache' / 'responses.jsonl'
LEGACY_DIR = CODE_DIR / 'legacy_code'

# Output baru (satu-satunya tempat menulis)
DATA_R3 = BASE_DIR / 'data_round3'
OUT_R3 = Path(os.environ.get('R3_OUTPUT_ROOT', BASE_DIR / 'outputs_round3')).resolve()
SILVER_R3 = DATA_R3 / 'silver_labeled_s15.csv'
TRAIN_PAIRS_R3 = DATA_R3 / 'train_pairs.csv'
RUNS_DIR = OUT_R3 / 'runs'                                      # satu subfolder per (model, seed)
PRED_DIR = OUT_R3 / 'predictions'
DEV_DIR = OUT_R3 / 'dev_choices'
FROZEN_JSON = OUT_R3 / 'frozen_choices.json'
TEST_DIR = OUT_R3 / 'test_predictions'                          # R3_04 (mode nyata)
DRY_DIR = OUT_R3 / 'dry_run'                                    # R3_04–R3_07 mode DRY_RUN (data dev)
LLM_DIR = OUT_R3 / 'llm'
LLM_CACHE_NEW = LLM_DIR / 'llm_cache' / 'responses.jsonl'
LLM_ARCHIVE_DIR = OUT_R3 / 'llm_archive_qwen3-max-2026-01-23'   # salinan cache qwen3-max lama + keluaran R3_05 sebelum ganti model
LLM_CACHE_ARCHIVE = LLM_ARCHIVE_DIR / 'responses.jsonl'         # hanya untuk analisis sensitivitas; tidak dipakai kondisi utama
EVAL_DIR = OUT_R3 / 'evaluation'
ROUTER_DIR = OUT_R3 / 'router_analysis'

# ─────────────────────────────────────────────────────────────
# 2. Nilai beku (rencana_latih_ulang.md)
# ─────────────────────────────────────────────────────────────
# D1–D3 label silver
SILVER_METHOD = 'score_sentence'
SILVER_THETA = 15.0
TEXT_VERSION = 'text_v1'                                         # sentence_a
EXPECTED_SILVER_DIST = {'tidak_disebutkan': 93203, 'positif': 4758, 'negatif': 4812, 'netral': 48215}
EXPECTED_N_CHANGED = 29284
EXPECTED_N_PAIRS = 150988

# D4–D5 pelatihan (identik dengan notebooks/03_FineTuning_3_Model.ipynb)
MODELS = {
    'indobert':    'indobenchmark/indobert-base-p1',
    'xlm_roberta': 'xlm-roberta-base',
    'mbert':       'bert-base-multilingual-cased',
}
SEEDS = [42, 123, 7]
LEARNING_RATE = 2e-5
BATCH_SIZE = 4                     # batch efektif 4
GRADIENT_ACCUMULATION_ACTIVE = False   # NB03 mendefinisikan GRADIENT_ACCUMULATION_STEPS=4 tetapi tidak menerapkannya
FP16_ACTIVE = False                    # NB03 mendefinisikan USE_FP16=True tetapi tidak menerapkannya → FP32
NUM_EPOCHS = 4
MAX_LENGTH = 80
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
NUM_LABELS = 4
EARLY_STOPPING = False             # checkpoint = epoch dengan macro-F1 4 kelas dev tertinggi
NUM_WORKERS = 0

# D6–D9 ensemble & pilihan dev
MARGIN_THRESHOLD = 0.15
TEXT_VERSIONS_ABLATION = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']

# D11 LLM — qwen3-max-2026-01-23 ditarik penyedia (404 model_not_found, revisi_round3/diagnosis_llm.md);
# diganti 2026-10-30 sebelum ada hasil LLM pada data uji (Log perubahan rencana_latih_ulang.md, ganti_model_llm.md)
LLM_MODEL = 'qwen3.7-max-2026-06-08'           # satu model untuk arbitrase, LLM-always, dan LLM-only
LLM_MODEL_OLD = 'qwen3-max-2026-01-23'         # hanya analisis sensitivitas LLM-only (cache arsip, tanpa API)
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 256
LLM_EXTRA_BODY = {'enable_thinking': False}
LLM_MAX_RETRIES = 3                            # sama dengan NB06 (MAX_RETRIES)
LLM_RETRY_DELAY_SEC = 5                        # sama dengan NB06 (RETRY_DELAY_SEC)
LLM_API_KEY_ENV = 'DASHSCOPE_API_KEY'          # API key HANYA dari environment variable
PRICE_INPUT_PER_1M = 2.50                      # USD, qwen3.7-max Singapore (agregator, diakses 2026-10-30; cek konsol) — hanya estimasi
PRICE_OUTPUT_PER_1M = 7.50

# D12 statistik evaluasi
STAT_SEED = 42
B_BOOT = 10_000
B_PERM = 10_000
ALPHA = 0.05

# ─────────────────────────────────────────────────────────────
# 3. Label dan aspek
# ─────────────────────────────────────────────────────────────
ASPECTS = ['pola_konsumsi', 'aktivitas_fisik', 'akses_layanan_kesehatan', 'pengetahuan_pengobatan']
LABEL_MAP = {'tidak_disebutkan': 0, 'positif': 1, 'negatif': 2, 'netral': 3}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
PROB_COLS = ['prob_tidak_disebutkan', 'prob_positif', 'prob_negatif', 'prob_netral']


def run_dir(model_name: str, seed: int) -> Path:
    return RUNS_DIR / f'{model_name}_seed{seed}'


def frozen_config_dict() -> dict:
    """Nilai beku yang dicatat di run_info.json dan frozen_choices.json."""
    return {
        'SILVER_METHOD': SILVER_METHOD, 'SILVER_THETA': SILVER_THETA, 'TEXT_VERSION': TEXT_VERSION,
        'MODELS': MODELS, 'SEEDS': SEEDS, 'LEARNING_RATE': LEARNING_RATE, 'BATCH_SIZE': BATCH_SIZE,
        'GRADIENT_ACCUMULATION_ACTIVE': GRADIENT_ACCUMULATION_ACTIVE, 'FP16_ACTIVE': FP16_ACTIVE,
        'NUM_EPOCHS': NUM_EPOCHS, 'MAX_LENGTH': MAX_LENGTH, 'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_RATIO': WARMUP_RATIO, 'GRAD_CLIP_NORM': GRAD_CLIP_NORM, 'NUM_LABELS': NUM_LABELS,
        'EARLY_STOPPING': EARLY_STOPPING, 'MARGIN_THRESHOLD': MARGIN_THRESHOLD, 'LLM_MODEL': LLM_MODEL,
        'LLM_TEMPERATURE': LLM_TEMPERATURE, 'LLM_MAX_TOKENS': LLM_MAX_TOKENS, 'LLM_EXTRA_BODY': LLM_EXTRA_BODY,
    }


def assert_writable(path) -> Path:
    """Penjaga: hanya boleh menulis di data_round3/ atau OUT_R3 (outputs_round3/ atau R3_OUTPUT_ROOT)."""
    p = Path(path).resolve()
    allowed = [DATA_R3.resolve(), OUT_R3.resolve()]
    if not any(p == a or a in p.parents for a in allowed):
        raise PermissionError(f'Menolak menulis di luar folder ronde 3: {p}')
    return p
