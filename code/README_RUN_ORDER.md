# README — Urutan menjalankan notebook ronde 3 (latih ulang & evaluasi ulang)

Semua notebook mengikuti `revisi_round3/rencana_latih_ulang.md` (rencana beku D1–D12). Notebook dijalankan **manual oleh peneliti**;
tidak ada notebook yang mengubah file lama (semua keluaran ke `data_round3/` dan `outputs_round3/`).
Kode lama dipakai ulang dari salinan verbatim di `legacy_code/` (diverifikasi sha256 terhadap notebook asli di R3_00).

## Ringkasan urutan

| # | Notebook | Dijalankan | Waktu (laptop RTX 4050 6 GB) | Masukan | Keluaran |
|---|---|---|---|---|---|
| 0 | `R3_00_config.ipynb` | sekali, kapan saja | < 1 menit | `config.py`, notebook lama | cek input + verifikasi `legacy_code/` (27 IDENTIK) |
| 1 | `R3_01_silver_labels.ipynb` | sekali | ± 6–8 menit (CPU) | `data/processed/tweets_preprocessed.csv`, `resources/`, `data/gold/gold_tweet_ids.csv`, `data/pairs/train_pairs.csv` | `data_round3/silver_labeled_s15.csv`, `data_round3/train_pairs.csv`, `data_round3/R3_01_manifest.json` |
| 2 | `R3_02_train_one_run.ipynb` | **9 kali** (3 model × seed 42/123/7) | lihat tabel waktu di bawah | `data_round3/train_pairs.csv`, `data/pairs/dev_pairs.csv` | `outputs_round3/runs/<model>_seed<seed>/` |
| 3 | `R3_03_dev_choices_and_freeze.ipynb` | sekali, setelah 9 run selesai | ± 15–25 menit (inferensi ablasi dev) | 9 folder `runs/`, dev | `outputs_round3/dev_choices/`, **`outputs_round3/frozen_choices.json`** |
| 4 | `R3_04_test_inference.ipynb` | **sekali** (`DRY_RUN=False`) | ± 10–20 menit | checkpoint, `test_pairs.csv` (tanpa label) | `outputs_round3/test_predictions/` |
| 5 | `R3_05_llm_calls.ipynb` | `RUN_API=False` dulu (estimasi), lalu `RUN_API=True` | estimasi < 1 menit; panggilan: lihat tabel estimasi di notebook (± 1,5–2 dtk/panggilan) | prediksi R3_04, cache arsip qwen3-max (hanya sensitivitas) | `outputs_round3/llm/` |
| 6 | `R3_06_evaluation.ipynb` | **sekali** (`DRY_RUN=False`) | ± 10–20 menit (bootstrap/permutasi B = 10.000) | semua di atas + label uji | `outputs_round3/evaluation/`, `outputs_round3/perbandingan_lama_vs_baru.md` |
| 7 | `R3_07_router_analysis.ipynb` | sekali, setelah R3_06 | ± 3–5 menit | sama dengan R3_06 | `outputs_round3/router_analysis/` |

Setiap notebook R3_04–R3_07 punya `DRY_RUN = True` sebagai **default**: seluruh alur dijalankan dengan **data dev** sebagai pengganti data
uji ke `outputs_round3/dry_run/` (penjaga aktif: membuka file uji akan error). Jalankan R3_04 → R3_07 dalam mode DRY_RUN dulu untuk
memastikan kode berjalan, **baru** ganti ke `DRY_RUN = False`. Mode dry R3_05 tidak memanggil API dan tidak membaca cache lama (isinya data uji); label LLM diisi label
buatan bertanda *mock* (revise ke label IndoBERT bila berbeda dari soft vote) supaya jalur retain/revise teruji;
angka dry **bukan hasil** dan tidak boleh dilaporkan.

## R3_02 — sembilan run

**Otomatis (disarankan di laptop):** klik dua kali `jalankan_R3_02_semua.bat` di folder ini (memakai Python pelatihan secara
otomatis; jangan tutup jendelanya). Atau dari Command Prompt biasa (bukan dari dalam editor, agar tetap jalan bila editor ditutup):

```
cd /d <REPO>\code
python R3_02_run_all.py
```

Pakai path Python lengkap di atas (Python yang berisi torch/transformers). Jangan pakai tombol *Run* editor bila editor memilih
venv lain tanpa transformers — skrip akan menolak dengan pesan yang jelas.

Skrip ini mengeksekusi `R3_02_train_one_run.ipynb` apa adanya untuk IndoBERT (seed 42, 123, 7) → XLM-R → mBERT, satu kernel baru
per run. Run yang sudah `DONE` dilewati; run yang terputus dilanjutkan; gagal → dicoba ulang 2× lalu berhenti. Bila skrip
berhenti karena apa pun, jalankan perintah yang sama lagi. Berhenti rapi setelah run berjalan: buat file `outputs_round3/STOP`.
Hanya model tertentu: `python R3_02_run_all.py indobert xlm_roberta`. Log: `outputs_round3/logs/run_all.log` (+ notebook hasil
eksekusi per run di folder yang sama). Selama berjalan, sleep otomatis Windows dicegah.

**Manual:** Isi `MODEL_NAME` dan `SEED` di sel parameter → *Run All*. Bila terputus, *Run All* lagi: run dilanjutkan dari resume state
(laptop: akhir epoch terakhir; Kaggle/Colab dengan `CKPT_EVERY_STEPS=5000`: tiap 5.000 langkah). Hasil identik dengan run tanpa
interupsi (diuji: selisih bobot maks 0,0). Run yang sudah selesai (`DONE`) tidak dilatih ulang.

### Perkiraan waktu per run (4 epoch × 37.747 langkah, batch 4, FP32)

Diukur di laptop ini (RTX 4050 Laptop 6 GB, 500 langkah steady-state setelah 100 langkah pemanasan, 2026-10-26):

| Model | Langkah/detik | Per epoch | Per run (4 epoch) | 3 seed | VRAM puncak |
|---|---|---|---|---|---|
| IndoBERT (`indobenchmark/indobert-base-p1`) | 9,7 | ± 65 menit | ± 4,5–5 jam | ± 14–15 jam | 2,3 GB |
| mBERT (`bert-base-multilingual-cased`) | 7,5 | ± 84 menit | ± 6–6,5 jam | ± 18–20 jam | 3,3 GB |
| XLM-R (`xlm-roberta-base`) | 5,3 | ± 118 menit | ± 8–9 jam | ± 25–27 jam | 5,2 GB |
| **Total 9 run** | | | | **± 57–62 jam** | |

Rentang sudah termasuk evaluasi dev per epoch, penyimpanan checkpoint, dan cadangan ± 5–10% untuk *thermal throttling*. Kecepatan di
Kaggle/Colab (T4/P100) belum diukur; perkiraan kasar: setara orde besarnya dengan laptop ini — lihat `epochs[].steps_per_second_session` di
`run_info.json` setelah epoch pertama untuk perkiraan sebenarnya.

Run boleh dibagi ke beberapa mesin (laptop, Kaggle, Colab) — konfigurasi identik; perangkat keras/versi pustaka tercatat di
`run_info.json` dan dilaporkan.

### Laptop (RAM 8 GB): persiapan

* **Page file ≥ 16 GB** (sudah diset 16.000–24.000 MB). Kekurangan memori virtual sebelumnya menyebabkan *access violation* saat
  menyimpan checkpoint.
* Tutup aplikasi berat (browser, dll.) — sisakan RAM bebas ≥ 3 GB. Satu run memakai ± 4 GB RAM dan ± 3–5 GB VRAM.
* **Ruang disk**: checkpoint terbaik ± 0,5 GB (IndoBERT, mBERT ± 0,7 GB) dan ± 1,1 GB (XLM-R) per run → total ± 7 GB di drive
  `outputs_round3` (D:). `resume/state.pt` sementara 1,3–3,3 GB per run (dihapus otomatis setelah selesai bila
  `KEEP_RESUME_STATE=False`). Drive C: hanya ± 8–12 GB bebas dan dipakai page file — **jangan** mengarahkan `R3_OUTPUT_ROOT` ke C:.
* Ketiga model dasar (`indobenchmark/indobert-base-p1`, `xlm-roberta-base`, `bert-base-multilingual-cased`) sudah ada lengkap
  di cache Hugging Face laptop (`%USERPROFILE%\.cache\huggingface\hub`, diunduh saat pelatihan NB03); tidak perlu internet.
  R3_02 otomatis memakai mode offline bila model ada di cache, jadi internet putus atau menggantung tidak berpengaruh
  (diuji dengan endpoint yang tidak merespons).
* Matikan *sleep*/hibernasi saat dicolok (Settings → System → Power), colok charger, dan tunda Windows Update selama run
  berjalan — ini yang bisa menghentikan run, bukan internet.
* Pendinginan: GPU 43–47 °C saat uji; tetap pakai kipas pendingin untuk run berjam-jam.

### Kaggle / Colab

1. Zip dan unggah: `revisi_round3/notebooks_r3/` (`config.py`, `r3_lib.py`, `r3_eval.py`, `legacy_code/`, notebook),
   `revisi_round3/rencana_latih_ulang.md`, `data_round3/train_pairs.csv`, `data_round3/R3_01_manifest.json`,
   `data/pairs/dev_pairs.csv` — **dengan struktur folder yang sama**. Jangan ikut sertakan `data/pairs/test_pairs.csv`.
2. Di sel parameter, sebelum import: `os.environ['R3_BASE_DIR']` = folder bundel, `os.environ['R3_CODE_DIR']` =
   `<bundel>/revisi_round3/notebooks_r3`, `os.environ['R3_OUTPUT_ROOT']` = Kaggle `/kaggle/working/outputs_round3`, Colab
   `/content/drive/MyDrive/outputs_round3` (Google Drive, agar resume bertahan). Set `CKPT_EVERY_STEPS = 5000`.
3. GPU + internet aktif.
4. **Bawa pulang** folder `outputs_round3/runs/<model>_seed<seed>/` **tanpa** `resume/`: `best/`, `<model>_seed<seed>_dev.csv`,
   `training_log.json`, `run_info.json`, `DONE`. Letakkan di `outputs_round3/runs/` laptop. R3_03 memeriksa sha256 berkas latih/dev
   yang tercatat di `run_info.json` — harus sama dengan `data_round3/train_pairs.csv` laptop.

## Checklist SEBELUM R3_04 mode nyata (menyentuh data uji)

- [ ] R3_00: semua input ✓ dan 27 sel `legacy_code/` IDENTIK.
- [ ] R3_01 selesai; `R3_01_manifest.json` ada; distribusi silver = 93.203 / 4.758 / 4.812 / 48.215; 29.284 label berubah.
- [ ] 9 folder `outputs_round3/runs/<model>_seed<seed>/` lengkap dengan `DONE`, `best/`, prediksi dev, `run_info.json`.
- [ ] R3_03 selesai **tanpa** `ALLOW_SMOKE_RUNS`; `frozen_choices.json` ada dan `"smoke": false`.
- [ ] Bobot model, model individual terbaik, skema Tabel 17, `best_T`, referensi dev sudah dicatat (dari dev saja).
- [ ] R3_04 → R3_07 sudah lolos dalam mode `DRY_RUN = True` **setelah** R3_03 (memakai checkpoint sesungguhnya).
- [ ] Notebook R3_04–R3_07 **tidak diubah** setelah R3_03 (hash kode sel dicek; bila diubah, R3_04 berhenti). Bila perlu
      memperbaiki kode, ubah dulu, lalu jalankan ulang R3_03 dengan `FORCE_REFREEZE=True` **sebelum** data uji disentuh dan catat
      di Log perubahan `rencana_latih_ulang.md`.
- [ ] `DASHSCOPE_API_KEY` diset sebagai environment variable (bukan di notebook) sebelum R3_05 `RUN_API=True`.
- [ ] Ruang disk cukup; tidak ada folder `outputs_round3/test_predictions/` dari percobaan sebelumnya.

Setelah itu: R3_04 (`DRY_RUN=False`) → R3_05 (`DRY_RUN=False`, `RUN_API=False` untuk melihat estimasi → `RUN_API=True`) →
R3_06 (`DRY_RUN=False`) → R3_07 (`DRY_RUN=False`). R3_04 dan R3_06 menolak dijalankan dua kali (`DONE.json`).

## Catatan

* **Penomoran tabel**: R3_06 menulis `pemetaan_tabel_naskah.md` (pemetaan isi → nomor tabel/gambar naskah). Nomor ini
  perkiraan — **konfirmasi** dengan naskah sebelum menyalin angka.
* R3_05 (sejak 2026-10-30, lihat `../ganti_model_llm.md`): model `qwen3.7-max-2026-06-08` untuk arbitrase, LLM-always, dan
  LLM-only; ketiganya dipanggil ulang (± 1.120 panggilan unik), cache qwen3-max lama tidak dipakai untuk kondisi utama. Arbitrase dan
  LLM-always berbagi kunci cache untuk prompt identik. Bila panggilan gagal karena galat API (kuota/saldo, koneksi), notebook berhenti
  tanpa menulis label; jalankan ulang (kernel boleh tetap) untuk melanjutkan dari cache — **jangan** mengganti model.
  `MAX_NEW_CALLS` membatasi jumlah panggilan baru per sesi. LLM-only qwen3-max untuk analisis sensitivitas dibaca dari
  `outputs_round3/llm_archive_qwen3-max-2026-01-23/` tanpa API.
* Variabel lingkungan yang dikenali: `R3_BASE_DIR`, `R3_CODE_DIR`, `R3_OUTPUT_ROOT`, `R3_DRY_RUN`, `R3_RUN_API`, `R3_MODEL_NAME`,
  `R3_SEED`, `R3_CKPT_EVERY_STEPS` (sisanya hanya untuk uji pengembang).
