"""Jalankan R3_02_train_one_run.ipynb berurutan untuk semua run (default 9: IndoBERT → XLM-R → mBERT, seed 42/123/7).

Cara pakai (dari Command Prompt / PowerShell biasa, BUKAN dari dalam notebook, agar tetap jalan bila editor ditutup):

    cd /d <REPO>\\code
    set PY=python
    %PY% R3_02_run_all.py                           # semua 9 run
    %PY% R3_02_run_all.py indobert xlm_roberta      # hanya model tertentu (urutan sesuai argumen)

* Notebook R3_02 dieksekusi apa adanya (kode identik dengan menjalankan manual), satu kernel baru per run → RAM/VRAM bersih.
* Run yang sudah selesai (file DONE) dilewati; run yang terputus dilanjutkan dari resume state. Jadi bila skrip ini berhenti
  (crash, listrik, restart), cukup jalankan perintah yang sama lagi.
* Bila satu run gagal, dicoba ulang sampai MAX_RETRY kali (melanjutkan dari resume state); bila tetap gagal, skrip berhenti.
* Berhenti dengan rapi setelah run yang sedang berjalan: buat file kosong `outputs_round3/STOP` (hapus lagi untuk lanjut).
* Selama berjalan, Windows dicegah masuk sleep (sleep otomatis saja; menutup layar laptop / Windows Update tetap bisa menghentikan).
* Log: `outputs_round3/logs/run_all.log` dan notebook hasil eksekusi per run `outputs_round3/logs/R3_02_<model>_seed<seed>.ipynb`.
"""
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import importlib.util

# Kernel notebook memakai Python yang sama dengan skrip ini → harus Python yang berisi torch/transformers (bukan .venv lain)
_missing = [m for m in ('nbformat', 'nbclient', 'ipykernel', 'torch', 'transformers') if importlib.util.find_spec(m) is None]
if _missing:
    sys.exit(f'Python ini ({sys.executable}) tidak punya: {", ".join(_missing)}.\n'
             'Jalankan dengan Python yang dipakai untuk pelatihan, mis.:\n'
             r'  python R3_02_run_all.py')

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8', errors='replace')     # konsol/redirect Windows: hindari UnicodeEncodeError
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import config as C  # noqa: E402

MAX_RETRY = 2                       # percobaan ulang per run setelah gagal (masing-masing melanjutkan resume state)
MIN_FREE_DISK_GB = 6                # ruang bebas minimum di drive output sebelum memulai run
NOTEBOOK = CODE_DIR / 'R3_02_train_one_run.ipynb'
SMOKE = os.environ.get('R3_SMOKE_TEST') == 'True'          # hanya untuk uji pengembang
RUNS_ROOT = C.OUT_R3 / 'smoke_runs' if SMOKE else C.RUNS_DIR
LOG_DIR = C.assert_writable(C.OUT_R3 / 'logs')
STOP_FILE = C.OUT_R3 / 'STOP'


def log(msg):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print(line, flush=True)
    with open(LOG_DIR / 'run_all.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def keep_awake(on: bool):
    """Cegah sleep otomatis Windows selama skrip berjalan (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def collapse_cr(text: str) -> str:
    """Ringkas keluaran progress bar (\\r) agar notebook hasil eksekusi tetap kecil: per baris, simpan teks setelah \\r terakhir."""
    return '\n'.join(seg.split('\r')[-1] for seg in text.split('\n'))


class LiveClient(NotebookClient):
    """NotebookClient yang meneruskan keluaran sel ke konsol secara langsung (progress bar ikut tampil)."""
    last_progress_log = 0.0

    def output(self, outs, msg, display_id, cell_index):
        if msg['msg_type'] == 'stream':
            text = msg['content']['text']
            sys.stdout.write(text); sys.stdout.flush()
            if '\r' not in text:                                     # baris print biasa → juga ke log
                with open(LOG_DIR / 'run_all.log', 'a', encoding='utf-8') as f:
                    f.write(text if text.endswith('\n') else text + '\n')
            elif time.time() - self.last_progress_log > 600:         # progress bar → ke log tiap 10 menit
                self.last_progress_log = time.time()
                with open(LOG_DIR / 'run_all.log', 'a', encoding='utf-8') as f:
                    f.write('    progress: ' + text.split('\r')[-1].strip() + '\n')
        elif msg['msg_type'] == 'error':
            sys.stdout.write('\n' + '\n'.join(re.sub(r'\x1b\[[0-9;]*m', '', t) for t in msg['content']['traceback']) + '\n')
        return super().output(outs, msg, display_id, cell_index)


def run_one(model, seed):
    os.environ.update({'R3_MODEL_NAME': model, 'R3_SEED': str(seed), 'R3_CODE_DIR': str(CODE_DIR)})
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = LiveClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': str(CODE_DIR)}})
    try:
        client.execute()
        return True, ''
    except Exception as e:                                           # CellExecutionError, DeadKernelError, dll.
        detail = re.sub(r'\x1b\[[0-9;]*m', '', str(e))[-1500:]
        return False, f'{type(e).__name__}: {detail}'
    finally:
        for c in nb.cells:
            for o in c.get('outputs', []):
                if o.get('output_type') == 'stream':
                    o['text'] = collapse_cr(o['text'])
        nbformat.write(nb, LOG_DIR / f'R3_02_{model}_seed{seed}.ipynb')


def main():
    models = sys.argv[1:] or list(C.MODELS)
    bad = [m for m in models if m not in C.MODELS]
    assert not bad, f'Model tidak dikenal: {bad} (pilihan: {list(C.MODELS)})'
    runs = [(m, s) for m in models for s in C.SEEDS]
    log(f'=== Mulai: {len(runs)} run {"(SMOKE TEST) " if SMOKE else ""}| output {C.OUT_R3}')
    keep_awake(True)
    try:
        for i, (m, s) in enumerate(runs, 1):
            tag = f'{m}_seed{s}'
            if (RUNS_ROOT / tag / 'DONE').exists():
                log(f'[{i}/{len(runs)}] {tag}: sudah selesai — dilewati')
                continue
            if STOP_FILE.exists():
                log(f'File STOP ditemukan ({STOP_FILE}) — berhenti sebelum {tag}. Hapus file itu lalu jalankan ulang untuk lanjut.')
                return 0
            C.OUT_R3.mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(C.OUT_R3).free / 2**30
            if free_gb < MIN_FREE_DISK_GB:
                log(f'Ruang disk bebas hanya {free_gb:.1f} GB (< {MIN_FREE_DISK_GB} GB) — berhenti sebelum {tag}.')
                return 1
            for attempt in range(1, MAX_RETRY + 2):
                log(f'[{i}/{len(runs)}] {tag}: mulai (percobaan {attempt}) | disk bebas {free_gb:.1f} GB')
                t0 = time.time()
                ok, err = run_one(m, s)
                if ok and (RUNS_ROOT / tag / 'DONE').exists():
                    log(f'[{i}/{len(runs)}] {tag}: SELESAI dalam {(time.time() - t0) / 3600:.2f} jam')
                    break
                log(f'[{i}/{len(runs)}] {tag}: GAGAL setelah {(time.time() - t0) / 60:.1f} menit — {err or "DONE tidak ada"}')
                if attempt <= MAX_RETRY:
                    log('    mencoba lagi dalam 60 detik (melanjutkan dari resume state)...')
                    time.sleep(60)
            else:
                log(f'{tag} tetap gagal setelah {MAX_RETRY + 1} percobaan — berhenti. Periksa {LOG_DIR / f"R3_02_{tag}.ipynb"}.')
                return 1
        log('=== Semua run selesai. Lanjut ke R3_03_dev_choices_and_freeze.ipynb')
        return 0
    finally:
        keep_awake(False)


if __name__ == '__main__':
    sys.exit(main())
