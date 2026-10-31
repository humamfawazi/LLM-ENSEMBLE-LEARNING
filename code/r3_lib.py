"""Pustaka bersama notebook R3_*.

Prinsip: logika lama TIDAK ditulis ulang. Sel notebook lama disalin verbatim ke legacy_code/ (lihat manifest.json) dan
dieksekusi dengan exec_legacy(). Kode baru di sini hanya untuk: memuat/verifikasi kode lama, penjaga akses data uji,
loop pelatihan yang bisa di-resume (urutan operasi per langkah identik dengan train_epoch NB03), rekonstruksi skema
Tabel 17 dari kuncinya, dan klien cache LLM.
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config as C

MANIFEST_PATH = C.LEGACY_DIR / 'manifest.json'


# ─────────────────────────────────────────────────────────────
# Hash & util
# ─────────────────────────────────────────────────────────────
def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def notebook_code_hash(nb_path) -> str:
    """Hash isi sel KODE saja (tanpa output) — dipakai untuk membekukan kode evaluasi."""
    nb = json.loads(Path(nb_path).read_text(encoding='utf-8'))
    return sha256_text(json.dumps([''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'], ensure_ascii=False))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, obj) -> None:
    p = C.assert_writable(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp, p)


def environment_info() -> dict:
    import torch
    info = {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}
    try:
        import transformers, sklearn
        info.update({'transformers': transformers.__version__, 'sklearn': sklearn.__version__})
    except Exception:
        pass
    if torch.cuda.is_available():
        info.update({'gpu': torch.cuda.get_device_name(0),
                     'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                     'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()})
    return info


def hf_offline_if_cached(model_id: str) -> bool:
    """Bila model dasar sudah lengkap di cache Hugging Face lokal, paksa mode offline (HF_HUB_OFFLINE=1).
    Koneksi yang menggantung (bukan putus total) membuat from_pretrained gagal dengan ConnectTimeout alih-alih memakai cache.
    Harus dipanggil SEBELUM transformers/huggingface_hub di-import. Bobot yang dimuat sama persis (file cache yang sama)."""
    if os.path.isdir(model_id):
        return False                                                 # path lokal: tidak pernah memakai jaringan
    hub = Path(os.environ.get('HF_HUB_CACHE') or Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub')
    snaps = hub / f"models--{model_id.replace('/', '--')}" / 'snapshots'
    cached = snaps.is_dir() and any((s / 'config.json').exists() and any((s / f).exists() for f in ('model.safetensors', 'pytorch_model.bin'))
                                    for s in snaps.iterdir())
    if cached:
        if 'huggingface_hub' in sys.modules and os.environ.get('HF_HUB_OFFLINE') != '1':
            print('⚠️  huggingface_hub sudah ter-import — mode offline mungkin tidak berlaku; Restart kernel lalu Run All')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return cached


# ─────────────────────────────────────────────────────────────
# Kode lama (legacy_code/)
# ─────────────────────────────────────────────────────────────
def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))


def legacy_source(name: str) -> str:
    man = load_manifest()[name]
    src = (C.LEGACY_DIR / f'{name}.py').read_text(encoding='utf-8')
    if sha256_text(src) != man['sha256']:
        raise RuntimeError(f'legacy_code/{name}.py berubah dari salinan verbatim (sha256 tidak cocok dengan manifest)')
    return src


def exec_legacy(names, ns: dict) -> dict:
    for n in ([names] if isinstance(names, str) else names):
        exec(compile(legacy_source(n), f'<legacy:{n}>', 'exec'), ns)
    return ns


def verify_legacy_against_originals() -> pd.DataFrame:
    """Bandingkan tiap salinan legacy dengan sel di notebook aslinya (bila notebook asli tersedia)."""
    import ast
    rows = []
    for name, m in load_manifest().items():
        src_nb = C.BASE_DIR / m['source_notebook']
        status = 'notebook asli tidak tersedia (dilewati)'
        if src_nb.exists():
            nb = json.loads(src_nb.read_text(encoding='utf-8'))
            cell = ''.join(nb['cells'][m['cell_index']]['source'])
            if m.get('extract', '').startswith('ast literal'):
                node = next(x for x in ast.walk(ast.parse(cell)) if isinstance(x, ast.Assign)
                            and getattr(x.targets[0], 'id', '') == 'BASELINE_SRC')
                cell = ast.literal_eval(node.value)
            status = 'IDENTIK' if sha256_text(cell) == m['sha256'] == sha256_text(legacy_source(name)) else 'BERBEDA'
        rows.append({'legacy': name, 'sumber': m['source_notebook'], 'sel': m['cell_index'], 'status': status})
    return pd.DataFrame(rows)


def base_namespace(**extra) -> dict:
    """Import yang dipakai sel-sel notebook lama (setara sel setup NB02/NB03/NB05/NB05B/NB06)."""
    import warnings
    from collections import Counter
    from itertools import combinations
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from sklearn.metrics import classification_report, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    ns = dict(os=os, re=re, json=json, random=random, time=time, warnings=warnings, np=np, pd=pd, torch=torch, nn=nn, F=F,
              AdamW=AdamW, Dataset=Dataset, DataLoader=DataLoader, tqdm=tqdm, f1_score=f1_score,
              classification_report=classification_report, compute_class_weight=compute_class_weight,
              AutoTokenizer=AutoTokenizer, AutoModelForSequenceClassification=AutoModelForSequenceClassification,
              get_linear_schedule_with_warmup=get_linear_schedule_with_warmup, Counter=Counter, combinations=combinations,
              plt=plt, sns=sns, display=lambda *a, **k: None,
              ASPECTS=list(C.ASPECTS), NUM_LABELS=C.NUM_LABELS, PROB_COLS=list(C.PROB_COLS),
              LABEL_MAP_INV=dict(C.LABEL_MAP_INV), MODELS=list(C.MODELS))
    ns.update(extra)
    return ns


# ─────────────────────────────────────────────────────────────
# Penjaga data uji
# ─────────────────────────────────────────────────────────────
_TEST_PATTERNS = ('test_pairs', 'test_predictions', '_test.csv', '_test_avg.csv', 'ensemble_test', 'final_labels_test')
_GUARD = {'on': False}


def forbid_test_access(extra_patterns=(), allow_dirs=()) -> None:
    """Sesudah dipanggil, membaca/menulis file yang namanya mengandung pola data uji akan menimbulkan error.
    allow_dirs: folder yang dikecualikan (mis. folder DRY_RUN yang berisi file bernama *_test.csv dari data dev)."""
    pats = tuple(_TEST_PATTERNS) + tuple(extra_patterns)
    allowed = [str(Path(a).resolve()).replace('\\', '/').lower() for a in allow_dirs]
    if _GUARD['on']:
        return
    _orig_read_csv, _orig_open = pd.read_csv, builtins.open

    def _check(p):
        s = str(Path(p).resolve()).replace('\\', '/').lower()
        if any(s.startswith(a + '/') for a in allowed):
            return
        if any(x in s for x in pats):
            raise PermissionError(f'Akses data uji DILARANG di notebook ini: {p}')

    def read_csv(path, *a, **k):
        if isinstance(path, (str, os.PathLike)):
            _check(path)
        return _orig_read_csv(path, *a, **k)

    def _open(file, *a, **k):
        if isinstance(file, (str, os.PathLike)):
            _check(file)
        return _orig_open(file, *a, **k)

    pd.read_csv = read_csv
    builtins.open = _open
    _GUARD['on'] = True
    print('Penjaga aktif: file data uji tidak dapat dibaca di notebook ini.')


# ─────────────────────────────────────────────────────────────
# RNG
# ─────────────────────────────────────────────────────────────
def capture_rng() -> dict:
    import torch
    s = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        s['cuda'] = torch.cuda.get_rng_state_all()
    return s


def restore_rng(s: dict) -> None:
    import torch
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    if 'cuda' in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])


# ─────────────────────────────────────────────────────────────
# Pelatihan satu run (dapat di-resume)
# ─────────────────────────────────────────────────────────────
class SimulatedInterruption(RuntimeError):
    """Hanya untuk uji resume: menghentikan run tanpa menyimpan (meniru sesi terputus)."""


def _train_fingerprint(model_name, seed, model_id, n_train, num_epochs, train_sha, dev_sha) -> dict:
    return {'model_name': model_name, 'seed': seed, 'model_id': model_id, 'n_train': n_train, 'num_epochs': num_epochs,
            'train_sha256': train_sha, 'dev_sha256': dev_sha, **{k: v for k, v in C.frozen_config_dict().items()
                                                                 if k not in ('LLM_MODEL', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS', 'LLM_EXTRA_BODY')}}


def train_one_run(model_name: str, seed: int, *, model_id: str, df_train: pd.DataFrame, df_dev: pd.DataFrame,
                  out_dir: Path, num_epochs: int = C.NUM_EPOCHS, ckpt_every_steps: int = 5000,
                  stop_after_steps: int | None = None, keep_resume_state: bool = False,
                  train_sha: str = '', dev_sha: str = '', smoke: dict | None = None, log=print) -> dict:
    """Satu run (model, seed) dengan konfigurasi NB03. Urutan operasi identik dengan sel loop utama NB03:
    set_seed → tokenizer → model → dataset/loader → AdamW → linear warmup → CE berbobot → per epoch: train_epoch, evaluate(dev),
    simpan checkpoint terbaik bila dev_f1 > best. Tidak ada inferensi uji. Resume state disimpan tiap ckpt_every_steps dan
    di akhir epoch; status RNG dipulihkan sehingga hasilnya sama dengan run tanpa jeda."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    out_dir = C.assert_writable(out_dir)
    best_dir, resume_dir = out_dir / 'best', out_dir / 'resume'
    state_path, log_path, info_path = resume_dir / 'state.pt', out_dir / 'training_log.json', out_dir / 'run_info.json'
    dev_pred_path = out_dir / f'{model_name}_seed{seed}_dev.csv'
    for d in (best_dir, resume_dir):
        d.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'DONE').exists():
        log(f'[{model_name}_seed{seed}] sudah selesai (file DONE ada) — dilewati.')
        return json.loads(info_path.read_text(encoding='utf-8'))

    ns = exec_legacy(['nb03_dataset', 'nb03_train_functions'], base_namespace())
    SentencePairDataset, set_seed = ns['SentencePairDataset'], ns['set_seed']
    compute_class_weights, evaluate, save_predictions = ns['compute_class_weights'], ns['evaluate'], ns['save_predictions']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fingerprint = _train_fingerprint(model_name, seed, model_id, len(df_train), num_epochs, train_sha, dev_sha)
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {
        'fingerprint': fingerprint, 'environment': environment_info(), 'smoke': smoke or {}, 'sessions': [], 'epochs': []}
    if info['fingerprint'] != fingerprint:
        raise RuntimeError('Konfigurasi berbeda dari run yang tersimpan di folder ini — gunakan folder lain.')
    session = {'start': now_utc(), 'device': str(DEVICE), 'environment': environment_info()}
    info['sessions'].append(session)

    # ── Bobot kelas dari label latih (sekali, sama untuk semua run) ──
    class_weights = compute_class_weights(df_train['label'].values, num_classes=C.NUM_LABELS)
    info['class_weights'] = [round(float(x), 6) for x in class_weights]

    state = torch.load(state_path, map_location='cpu', weights_only=False) if state_path.exists() else None
    if state is not None and state['fingerprint'] != fingerprint:
        raise RuntimeError('resume/state.pt berasal dari konfigurasi lain.')

    # ── Persis urutan NB03 (LEVEL-3B) ──
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=C.NUM_LABELS, ignore_mismatched_sizes=True).to(DEVICE)
    train_dataset = SentencePairDataset(df_train, tokenizer, C.MAX_LENGTH)
    dev_dataset = SentencePairDataset(df_dev, tokenizer, C.MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=C.NUM_WORKERS)
    dev_loader = DataLoader(dev_dataset, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)
    optimizer = AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(C.WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    info.update({'steps_per_epoch': len(train_loader), 'total_steps': total_steps, 'warmup_steps': warmup_steps})

    run_log, best_dev_f1, start_epoch, k0 = {'epochs': []}, 0.0, 1, 0
    partial = None
    if state is not None:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        run_log, best_dev_f1 = state['run_log'], state['best_dev_f1']
        start_epoch, k0 = state['epoch'], state['step_in_epoch']
        partial = state.get('partial')
        if k0 == 0:
            restore_rng(state['rng'])
        log(f'RESUME dari epoch {start_epoch}, langkah {k0}/{len(train_loader)} (best_dev_f1={best_dev_f1:.4f})')

    def save_state(epoch, step_in_epoch, epoch_start_rng, acc):
        # Tensor disimpan langsung dari GPU (torch.save menyalin per storage, bukan seluruh model sekaligus ke RAM);
        # cache dibersihkan dulu agar RAM/VRAM longgar di laptop.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tmp = state_path.with_suffix('.tmp')
        torch.save({'fingerprint': fingerprint, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng': capture_rng(), 'epoch_start_rng': epoch_start_rng,
                    'epoch': epoch, 'step_in_epoch': step_in_epoch, 'partial': acc, 'run_log': run_log,
                    'best_dev_f1': best_dev_f1, 'saved_at': now_utc()}, tmp)
        os.replace(tmp, state_path)

    steps_this_session = 0
    for epoch in range(start_epoch, num_epochs + 1):
        log(f'    Epoch {epoch}/{num_epochs}')
        t0 = time.time()
        if epoch == start_epoch and k0 > 0:                       # resume di tengah epoch
            epoch_start_rng = state['epoch_start_rng']
            restore_rng(epoch_start_rng)
            it = iter(train_loader)
            for _ in range(k0):
                it._next_index()                                     # lewati indeks batch yang sudah dilatih (tanpa memuat data)
            restore_rng(state['rng'])
            total_loss, all_preds, all_labels = partial['total_loss'], list(partial['preds']), list(partial['labels'])
            step = k0
        else:
            epoch_start_rng = capture_rng()
            it = iter(train_loader)
            total_loss, all_preds, all_labels, step = 0.0, [], [], 0

        # ── Badan train_epoch NB03 (urutan operasi identik) ──
        model.train()
        for batch in tqdm(it, desc='  Training', leave=False, total=len(train_loader), initial=step):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if 'token_type_ids' in batch:
                kwargs['token_type_ids'] = batch['token_type_ids'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(**kwargs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=C.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            step += 1
            steps_this_session += 1
            if stop_after_steps is not None and steps_this_session >= stop_after_steps:
                raise SimulatedInterruption(f'berhenti buatan setelah {steps_this_session} langkah sesi ini')
            if ckpt_every_steps and step % ckpt_every_steps == 0 and step < len(train_loader):
                save_state(epoch, step, epoch_start_rng,
                           {'total_loss': total_loss, 'preds': np.array(all_preds), 'labels': np.array(all_labels)})
        train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        train_seconds = time.time() - t0

        dev_loss, dev_f1, dev_preds, dev_probs, dev_ids, dev_labels = evaluate(model, dev_loader, criterion, DEVICE)
        run_log['epochs'].append({'epoch': epoch, 'train_loss': round(train_loss, 4), 'train_f1': round(train_f1, 4),
                                  'dev_loss': round(dev_loss, 4), 'dev_f1': round(dev_f1, 4)})
        log(f'    train_loss={train_loss:.4f} train_f1={train_f1:.4f} | dev_loss={dev_loss:.4f} dev_f1={dev_f1:.4f}')
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_predictions(dev_ids, dev_probs, dev_preds, dev_labels, str(dev_pred_path))
            run_log['best_epoch'] = epoch
            log(f'    Checkpoint terbaik disimpan (dev_f1={dev_f1:.4f})')
        run_log['best_dev_f1'] = round(best_dev_f1, 4)
        C.assert_writable(log_path).write_text(json.dumps(run_log, indent=2), encoding='utf-8')
        info['epochs'] = [e for e in info['epochs'] if e['epoch'] != epoch] + [
            {'epoch': epoch, 'train_seconds': round(train_seconds, 1), 'steps': len(train_loader),
             'steps_per_second_session': round((step - (k0 if epoch == start_epoch else 0)) / max(train_seconds, 1e-9), 3),
             'dev_f1': round(dev_f1, 4), 'finished_at': now_utc()}]
        write_json(info_path, info)                                 # sebelum state: bila putus di antaranya, epoch diulang
        save_state(epoch + 1, 0, None, None)                        # batas epoch: lanjut ke epoch berikutnya

    if not dev_pred_path.exists():                                  # sama dengan NB03: pastikan prediksi dev ada
        bm = AutoModelForSequenceClassification.from_pretrained(best_dir, num_labels=C.NUM_LABELS,
                                                                ignore_mismatched_sizes=True).to(DEVICE)
        _, _, dp, dpr, di, dl = evaluate(bm, dev_loader, criterion, DEVICE)
        save_predictions(di, dpr, dp, dl, str(dev_pred_path))

    weights_file = next((best_dir / f for f in ('model.safetensors', 'pytorch_model.bin') if (best_dir / f).exists()), None)
    info.update({'status': 'complete', 'best_epoch': run_log.get('best_epoch'), 'best_dev_f1': run_log.get('best_dev_f1'),
                 'best_model_sha256': sha256_file(weights_file) if weights_file else None,
                 'dev_predictions_sha256': sha256_file(dev_pred_path), 'finished_at': now_utc(),
                 'total_train_seconds': round(sum(e['train_seconds'] for e in info['epochs']), 1)})
    session['end'] = now_utc()
    write_json(info_path, info)
    if not keep_resume_state and state_path.exists():
        state_path.unlink()
    (out_dir / 'DONE').write_text(now_utc(), encoding='utf-8')
    return info


# ─────────────────────────────────────────────────────────────
# Tabel 17: rekonstruksi prediksi satu skema 05B dari kuncinya (tanpa metrik)
# ─────────────────────────────────────────────────────────────
def scheme_complexity(key: str):
    fam = int(key[:2])
    n_models = 1 if fam == 1 else (2 if fam in (4, 5, 6, 8, 13) else 3)
    extra = 1 if fam >= 7 else 0
    return (n_models, extra)


def select_scheme(all_results: dict) -> str:
    """Rencana §6: ASC F1 dev tertinggi → Overall F1 dev → paling sederhana (lebih sedikit model, tanpa power/T) → urutan kunci."""
    return sorted(all_results, key=lambda k: (-round(all_results[k]['asc'], 12), -round(all_results[k]['overall'], 12),
                                              scheme_complexity(k), k))[0]


def scheme_predict(key: str, h: dict, probs_all: dict, model_weights_dev: dict, aspek, best_T: float) -> np.ndarray:
    """Prediksi skema `key` memakai fungsi pembantu 05B (h = namespace hasil exec legacy nb05b_helpers)."""
    available = list(probs_all)
    preds_all = {m: probs_all[m].argmax(axis=1) for m in available}
    hv, suv, wsv, temp, pw = (h['hard_vote'], h['soft_vote_uniform'], h['weighted_soft_vote_aspect'],
                             h['apply_temperature'], h['make_power_weights'])
    best2 = [m for m in available if m != 'mbert']

    def pair(rest):
        for m1 in available:
            if rest.startswith(m1 + '_') and rest[len(m1) + 1:] in available:
                return m1, rest[len(m1) + 1:]
        raise KeyError(key)

    def nombert_weights(power):
        return pw({asp: {m: v for m, v in w.items() if m in best2} for asp, w in model_weights_dev.items()}, power=power)

    if key.startswith('01_Individual_'):
        return preds_all[key[len('01_Individual_'):]]
    if key == '02_HardVote_NB05_baseline':
        return hv([preds_all[m] for m in available])
    if key == '03_SoftVote_linear_NB05_baseline':
        return wsv(probs_all, model_weights_dev, aspek)
    if key.startswith('04_HardVote_'):
        m1, m2 = pair(key[len('04_HardVote_'):]); return hv([preds_all[m1], preds_all[m2]])
    if key.startswith('05_SoftVote_uniform_'):
        m1, m2 = pair(key[len('05_SoftVote_uniform_'):]); return suv([probs_all[m1], probs_all[m2]])
    if key.startswith('06_SoftVote_weighted_'):
        m1, m2 = pair(key[len('06_SoftVote_weighted_'):]); return wsv({m1: probs_all[m1], m2: probs_all[m2]}, model_weights_dev, aspek)
    m = re.fullmatch(r'07_SoftVote_power([\d.]+)', key)
    if m:
        return wsv(probs_all, pw(model_weights_dev, power=float(m.group(1))), aspek)
    m = re.fullmatch(r'08_SoftVote_power([\d.]+)_nombert', key)
    if m:
        return wsv({x: probs_all[x] for x in best2}, nombert_weights(float(m.group(1))), aspek)
    m = re.fullmatch(r'(09_HardVote|10_SoftVote_uniform|11_SoftVote_weighted)_T([\d.]+)', key)
    if m:
        scaled = temp(probs_all, T=float(m.group(2)))
        if m.group(1).startswith('09'):
            return hv([p.argmax(axis=1) for p in scaled.values()])
        if m.group(1).startswith('10'):
            return suv(list(scaled.values()))
        return wsv(scaled, model_weights_dev, aspek)
    m = re.fullmatch(r'(12|13)_SoftVote_p([\d.]+)_T([\d.]+)(_nombert)?', key)
    if m:
        scaled = temp(probs_all, T=float(m.group(3)))
        if m.group(4):
            return wsv({x: scaled[x] for x in best2}, nombert_weights(float(m.group(2))), aspek)
        return wsv(scaled, pw(model_weights_dev, power=float(m.group(2))), aspek)
    raise KeyError(f'Kunci skema tidak dikenal: {key}')


# ─────────────────────────────────────────────────────────────
# LLM: klien cache (kunci & format sama dengan revision/04 CachedLoggingClient)
# ─────────────────────────────────────────────────────────────
def llm_cache_key(model, messages, temperature, max_tokens, extra_body) -> str:
    blob = json.dumps({'model': model, 'messages': messages, 'temperature': temperature,
                       'max_tokens': max_tokens, 'extra_body': extra_body}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


class CachedLLMClient:
    """Meniru client.chat.completions.create untuk dipakai call_llm NB06 apa adanya.
    Membaca cache lama (read-only) + cache baru; respons API baru hanya ditulis ke cache baru (append, bisa dilanjutkan)."""

    def __init__(self, read_only_caches, new_cache: Path, attempts_path: Path, live_client=None):
        self.new_cache, self.attempts_path, self.live = C.assert_writable(new_cache), C.assert_writable(attempts_path), live_client
        self.new_cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache, self.source = {}, {}
        for p in list(read_only_caches) + [self.new_cache]:
            if Path(p).exists():
                for line in Path(p).read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        r = json.loads(line)
                        self.cache.setdefault(r['key'], []).append(r)
                        self.source.setdefault(r['key'], str(p))
        self.counter, self.current_tag, self.missing = {}, None, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def has(self, key) -> bool:
        return key in self.cache

    def _log(self, rec):
        with open(self.attempts_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def _create(self, model, messages, temperature, max_tokens, extra_body=None, **_):
        key = llm_cache_key(model, messages, temperature, max_tokens, extra_body)
        n = self.counter.get(key, 0); self.counter[key] = n + 1
        cached = self.cache.get(key, [])
        if n < len(cached):
            r = cached[n]
            self._log({'key': key, 'tag': self.current_tag, 'attempt': n + 1, 'from_cache': True, 'ok': True,
                       'cache_source': self.source.get(key), 'prompt_tokens': r.get('prompt_tokens'),
                       'completion_tokens': r.get('completion_tokens'), 'latency_s': r.get('latency_s')})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r['raw_text']))])
        if self.live is None:
            self.missing.append(key)
            raise RuntimeError('Tidak ada di cache dan RUN_API=False — tidak memanggil API')
        t0 = time.perf_counter()
        try:
            resp = self.live.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                     max_tokens=max_tokens, extra_body=extra_body)
        except Exception as e:
            self._log({'key': key, 'tag': self.current_tag, 'attempt': n + 1, 'from_cache': False, 'ok': False,
                       'error': repr(e)[:500], 'latency_s': time.perf_counter() - t0})
            raise
        lat = time.perf_counter() - t0
        u = getattr(resp, 'usage', None)
        rec = {'key': key, 'tag': self.current_tag, 'model': model, 'response_model': getattr(resp, 'model', None),
               'response_id': getattr(resp, 'id', None), 'created_utc': now_utc(), 'messages': messages,
               'params': {'temperature': temperature, 'max_tokens': max_tokens, 'extra_body': extra_body},
               'raw_text': resp.choices[0].message.content, 'finish_reason': resp.choices[0].finish_reason,
               'prompt_tokens': getattr(u, 'prompt_tokens', None), 'completion_tokens': getattr(u, 'completion_tokens', None),
               'total_tokens': getattr(u, 'total_tokens', None), 'latency_s': lat}
        with open(self.new_cache, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.cache.setdefault(key, []).append(rec)
        self.source.setdefault(key, str(self.new_cache))
        self._log({'key': key, 'tag': self.current_tag, 'attempt': n + 1, 'from_cache': False, 'ok': True,
                   'prompt_tokens': rec['prompt_tokens'], 'completion_tokens': rec['completion_tokens'],
                   'latency_s': lat, 'response_model': rec['response_model']})
        return resp


# ─────────────────────────────────────────────────────────────
# Pembekuan (frozen_choices.json) — dipakai R3_03 (tulis) dan R3_04–R3_07 (cek)
# ─────────────────────────────────────────────────────────────
FROZEN_CODE_NOTEBOOKS = ['R3_04_test_inference.ipynb', 'R3_05_llm_calls.ipynb', 'R3_06_evaluation.ipynb',
                         'R3_07_router_analysis.ipynb']


def code_fingerprints() -> dict:
    """Hash kode yang menentukan hasil uji: notebook R3_04–R3_07 (sel kode saja), modul, salinan legacy."""
    out = {nb: notebook_code_hash(C.CODE_DIR / nb) for nb in FROZEN_CODE_NOTEBOOKS if (C.CODE_DIR / nb).exists()}
    for f in ['config.py', 'r3_lib.py', 'r3_eval.py', 'legacy_code/manifest.json']:
        out[f] = sha256_file(C.CODE_DIR / f)
    rev = C.REVISION_DIR / 'revision_utils.py'
    if rev.exists():
        out['revision/revision_utils.py'] = sha256_file(rev)
    return out


def data_fingerprints() -> dict:
    return {p.name: sha256_file(p) for p in [C.TRAIN_PAIRS_R3, C.SILVER_R3, C.DEV_PAIRS, C.TWEETS_PREPROCESSED]}


def verify_frozen(strict: bool, check_models: bool = True) -> dict:
    """Bandingkan keadaan sekarang dengan frozen_choices.json. strict=True (mode nyata): ketidakcocokan → error."""
    if not C.FROZEN_JSON.exists():
        raise FileNotFoundError(f'{C.FROZEN_JSON} tidak ada — jalankan R3_03 dulu')
    fr = json.loads(C.FROZEN_JSON.read_text(encoding='utf-8'))
    problems = []
    if strict and fr.get('smoke'):
        problems.append('frozen_choices.json berasal dari smoke test — tidak boleh dipakai untuk data uji')
    for k, v in fr['code_sha256'].items():
        now = code_fingerprints().get(k)
        if now != v:
            problems.append(f'kode berubah sejak dibekukan: {k}')
    for k, v in fr['data_sha256'].items():
        if data_fingerprints().get(k) != v:
            problems.append(f'data berubah sejak dibekukan: {k}')
    if check_models:
        for key, r in fr['runs'].items():
            p = Path(r['best_model_path'])
            if not p.exists() or sha256_file(p) != r['best_model_sha256']:
                problems.append(f'checkpoint tidak cocok: {key}')
    for msg in problems:
        print(('✗ ' if strict else '⚠️  ') + msg)
    if problems and strict:
        raise RuntimeError(f'{len(problems)} ketidakcocokan dengan frozen_choices.json — berhenti')
    if not problems:
        print('✓ Kode, data, dan checkpoint cocok dengan frozen_choices.json')
    return fr


# ─────────────────────────────────────────────────────────────
# NB06: prompt sistem, parameter, base URL (dibaca dari kode asli, bukan diketik ulang)
# ─────────────────────────────────────────────────────────────
def nb06_call_spec() -> dict:
    import ast
    src = legacy_source('nb06_call_llm')
    call = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'create'
                and any(k.arg == 'extra_body' for k in n.keywords))
    kw = {k.arg: k.value for k in call.keywords}
    spec = {'system_prompt': ast.literal_eval(kw['messages'].elts[0].values[1]),
            'temperature': ast.literal_eval(kw['temperature']), 'max_tokens': ast.literal_eval(kw['max_tokens']),
            'extra_body': ast.literal_eval(kw['extra_body'])}
    assert (spec['temperature'], spec['max_tokens'], spec['extra_body']) == (C.LLM_TEMPERATURE, C.LLM_MAX_TOKENS, C.LLM_EXTRA_BODY)
    return spec


def nb06_base_url() -> str:
    """Base URL endpoint DashScope (OpenAI-compatible) dari environment variable DASHSCOPE_BASE_URL.
    Rilis publik: endpoint workspace tidak disertakan; asal NB06 membaca literal URL dari notebook 06 asli."""
    url = os.environ.get('DASHSCOPE_BASE_URL', '').strip()
    if not url:
        raise RuntimeError('Set environment variable DASHSCOPE_BASE_URL ke endpoint OpenAI-compatible DashScope Anda, '
                           'mis. https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
    return url


def mode_dirs(dry_run: bool) -> SimpleNamespace:
    """Folder input/output R3_04–R3_07 sesuai mode."""
    root = C.DRY_DIR if dry_run else C.OUT_R3
    return SimpleNamespace(test_pred=(C.DRY_DIR / 'test_predictions') if dry_run else C.TEST_DIR,
                           llm=(C.DRY_DIR / 'llm') if dry_run else C.LLM_DIR,
                           eval=(C.DRY_DIR / 'evaluation') if dry_run else C.EVAL_DIR,
                           router=(C.DRY_DIR / 'router_analysis') if dry_run else C.ROUTER_DIR,
                           pairs=C.DEV_PAIRS if dry_run else C.TEST_PAIRS, root=root)
