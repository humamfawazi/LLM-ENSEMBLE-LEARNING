# Rencana Latih Ulang (dibekukan sebelum data uji disentuh)

**Tanggal pembekuan:** 2026-10-26
**Dasar:** `theta_report.md`, `leakage_audit.md`, `theta_final_report.md` (`calibrate_method_theta_validation.ipynb`).
**Status data uji:** sejak audit ini, belum ada perhitungan pada data uji. Kalibrasi tidak memuat `test_pairs.csv`; baris gold uji di `gold_labeled.csv` ikut terbaca lalu langsung disaring tanpa perhitungan.

Semua keputusan di bawah ditetapkan **hanya** dari data latih dan validasi. Setelah rencana ini dibekukan, data uji hanya dipakai **satu kali**
pada Langkah 8. Setiap penyimpangan dari rencana dicatat di bagian "Log perubahan" beserta alasannya, dan diungkapkan di naskah.

---

## 0. Ringkasan keputusan yang dibekukan

| # | Keputusan | Nilai beku | Dipilih dengan | Sumber |
|---|---|---|---|---|
| D1 | Metode skor silver | **`score_sentence`** (leksikon InSet + negasi 2 token, seluruh kalimat) | 76 pasangan gold validasi | `theta_final_report.md` |
| D2 | Ambang θ silver | **θ = 15,0** (kandidat 0,5–15,0; macro-F1 3 kelas; seri → `score_windowed` lalu θ kecil) | 76 pasangan gold validasi | `final_silver_config.json` |
| D3 | Teks & deteksi aspek silver | Loop pelabelan asli: deteksi `detect_aspect` pada `text`, skor pada `text_v3` | Tetap (sama dengan pipeline lama) | sel loop `02_Silverrr…` (sha256 `55acd499…`) |
| D4 | Konfigurasi pelatihan | Sama persis dengan run lama (lihat §2) | Tidak dituning | `03_FineTuning_3_Model.ipynb` |
| D5 | Pemilihan checkpoint | Epoch dengan macro-F1 4 kelas **dev** tertinggi; tanpa early stopping | Dev | §2 |
| D6 | Bobot weighted soft voting | Macro-F1 4 kelas **dev** per aspek per model | Dev | §3 |
| D7 | Pemicu LLM | Vote terpecah **atau** margin soft vote < 0,15 | Tidak dituning (spesifikasi 12 Sep) | §4 |
| D8 | Model individu terbaik | Overall macro-F1 **dev** tertinggi; seri → ASC F1 dev | Dev | §5 |
| D9 | Konfigurasi ablasi ensemble (Tabel 17) | Grid 69 skema NB05B; dipilih dengan **ASC F1 dev** | Dev | §6 |
| D10 | Ablasi preprocessing (Tabel 16) | Inferensi V1–V6 dari model baru, deskriptif (tidak memilih apa pun) | — | §7 |
| D11 | LLM judge | `qwen3-max-2026-01-23`, prompt dan parameter NB6 identik | Tidak dituning | §4 |
| D12 | Evaluasi uji | Satu kali, protokol §8 | — | §8 |

---

## 1. Label silver baru (D1–D3)

* **Prosedur:** jalankan **sel loop pelabelan asli** dengan `score_sentence` dan θ = 15,0 pada 37.747 tweet non-gold, dengan cara yang sama seperti
  `run_labeling_loop` di `calibrate_method_theta_validation.ipynb`. Pengecualian gold memakai `data/gold/gold_tweet_ids.csv`.
* **Hasil yang diharapkan** (sudah dihitung, untuk verifikasi):

| label | saat ini (`score_windowed`, θ = 3,0) | baru (`score_sentence`, θ = 15,0) |
|---|---|---|
| tidak_disebutkan | 93.203 | 93.203 |
| positif | 19.910 | 4.758 |
| negatif | 15.816 | 4.812 |
| netral | 22.059 | 48.215 |

  29.284 label berubah (19,39%). Proporsi berubah per aspek: pola_konsumsi 39,7%, pengetahuan_pengobatan 20,9%, akses 11,8%, aktivitas 5,2%.
  `tidak_disebutkan` tidak berubah, karena ditentukan oleh deteksi kata kunci, bukan oleh θ.
* **Keluaran (file baru; tidak menimpa file lama):** `data_round3/silver_labeled_s15.csv`, `data_round3/train_pairs.csv`
  (`sentence_a` = `text_v1`, `sentence_b` = kalimat aspek yang sama, 150.988 baris).
* **Tetap:** `data/pairs/dev_pairs.csv`, `data/pairs/test_pairs.csv`, dan `data/gold/gold_labeled.csv` dipakai apa adanya (split tidak diubah).
* **Catatan kerapuhan (diungkapkan di naskah):** pilihan D1–D2 berasal dari 76 pasangan validasi. θ = 15,0 adalah kandidat terbesar di grid;
  selisihnya terhadap pesaing terdekat (`score_windowed`, θ = 5,0) hanya 0,0057 (CI 95% −0,137 s.d. +0,130); dan kombinasi ini hanya terpilih
  lagi pada 40,5% resample bootstrap validasi. Ini konsekuensi aturan yang ditetapkan sebelumnya, bukan bukti keunggulan.

## 2. Pelatihan (D4–D5): sama dengan run lama

| Parameter | Nilai | Catatan |
|---|---|---|
| Model | `indobenchmark/indobert-base-p1`, `xlm-roberta-base`, `bert-base-multilingual-cased` | |
| Seed | 42, 123, 7 | rata-rata probabilitas softmax antar-seed |
| Optimizer | AdamW, LR 2e-5, weight decay 0,01 | |
| Scheduler | linear dengan warmup 10% dari total langkah | |
| Batch | **4 (efektif 4)** | `GRADIENT_ACCUMULATION_STEPS = 4` didefinisikan di NB03 tetapi **tidak aktif**; dipertahankan demi kesamaan dengan run lama |
| Presisi | **FP32** | `USE_FP16 = True` didefinisikan tetapi **tidak aktif** |
| Epoch | 4, tanpa early stopping | |
| `MAX_LENGTH` | 80 | juga dipakai untuk inferensi ulang (bukan 128) |
| Loss | Cross-entropy berbobot, `compute_class_weight('balanced')` dari **label latih baru** | bobot berubah karena distribusi label berubah |
| Grad clipping | 1,0 | |
| Pemilihan checkpoint | epoch dengan macro-F1 4 kelas **dev** tertinggi | |

* **Perubahan prosedur demi isolasi data uji:** loop pelatihan **tidak** menjalankan inferensi uji dan **tidak** mencetak metrik uji.
  NB03 lama menginferensi uji setelah tiap seed dan mencetak macro-F1 uji di sel agregasi, jadi kedua bagian itu dihapus dari salinan untuk latih ulang.
  Inferensi uji dipindahkan ke Langkah 8.
* Checkpoint dan prediksi disimpan di folder baru (`outputs_round3/…`); checkpoint dan prediksi lama tidak ditimpa.

## 3. Ensemble (D6)

* Bobot per aspek = macro-F1 4 kelas dev tiap model (rata-rata 3 seed), dinormalisasi per aspek; hard vote mayoritas, dengan seri tiga arah →
  indeks label terkecil (sama dengan NB05).
* Margin = selisih dua probabilitas ensemble tertinggi; `vote_type` = bulat bila ketiga model sepakat.

## 4. LLM-as-a-Judge (D7, D11)

* Pemicu: `vote_type == 'terpecah'` atau `soft_vote_margin < 0.15`. Jumlah kasus akan berbeda dari 94, karena prediksi model berubah.
* Model `qwen3-max-2026-01-23`; fungsi `build_prompt` dan `call_llm` NB6 dieksekusi apa adanya (temperature 0, `max_tokens` 256, thinking off,
  maks. 3 percobaan, fallback ke soft vote). Respons, token, dan latensi disimpan ke cache JSONL seperti di `revision/04…`.
* **Isolasi:** prompt tidak memuat label gold. Panggilan LLM pada kasus uji yang ditandai dilakukan pada Langkah 8, dan keputusannya tidak dievaluasi
  sebelum seluruh pipeline beku.
* Baseline LLM-only tidak bergantung pada model (prompt hanya tweet + aspek), sehingga respons di cache lama dipakai ulang; tidak ada panggilan baru.
  LLM-always dijalankan ulang untuk kasus yang usulan ensemblenya berubah.

## 5. Model individu terbaik (D8)

* Dipilih sebelum evaluasi uji: **Overall macro-F1 dev tertinggi** di antara IndoBERT, XLM-R, mBERT (rata-rata 3 seed). Bila seri, ASC F1 dev; bila
  masih seri, urutan IndoBERT → XLM-R → mBERT.
* Model ini menjadi pembanding "model individu terbaik" dalam kerangka *ensemble paradox* dan uji signifikansi. Ketiga model tetap dilaporkan.

## 6. Ablasi konfigurasi ensemble — Tabel 17 (D9)

* Grid identik dengan `05B_Ensemble_Optimization` (69 skema): individu (3), baseline hard & weighted soft vote (2), subset 2 model × {hard, soft
  uniform, soft weighted} (9), power weight 1,5/2/3/4/5 untuk 3 model dan 2 model tanpa mBERT (10), temperature T ∈ {0,3; 0,5; 0,7; 0,8; 0,9; 1,0;
  1,2; 1,5; 2,0} × {hard, soft uniform, soft weighted} (27), dan kombinasi power × T (18).
* Seluruh grid dihitung pada **dev**. Skema terpilih = **ASC F1 dev** tertinggi; bila seri, Overall F1 dev, lalu skema paling sederhana
  (lebih sedikit model, tanpa power/temperature).
* Nilai uji **hanya** dilaporkan untuk skema terpilih (dan baseline). Bila grid lengkap ingin ditampilkan pada uji, beri label "eksplorasi; tidak
  dipakai memilih" dan jangan sebut skema mana pun "terbaik" berdasarkan uji. Klaim lama "+0,047 dari optimasi terbaik tanpa LLM" dihapus.

## 7. Ablasi preprocessing — Tabel 16 (D10)

* Inferensi V1–V6 dari checkpoint baru (3 seed, `MAX_LENGTH` 80), meniru `03.2_Ablation_Inference`, pada dev **dan** uji. Tidak ada konfigurasi yang
  dipilih dari tabel ini: model utama tetap V1.
* Naskah menyebut bahwa ablasi ini hanya mengganti preprocessing saat inferensi (model dilatih pada V1).

## 8. Evaluasi uji: satu kali (D12)

Urutan, setelah Langkah 1–7 selesai dan semua pilihan berbasis dev tercatat di `outputs_round3/frozen_choices.json`:
1. Inferensi uji untuk 9 checkpoint (3 model × 3 seed); agregasi antar-seed; hard vote dan weighted soft vote memakai bobot dev.
2. Penandaan kasus uji (D7) dan pemanggilan LLM snapshot pada kasus yang ditandai (D11); LLM-always untuk kasus yang usulan ensemblenya berubah.
3. Inferensi ablasi V1–V6 (D10) dan skema Tabel 17 terpilih (D9) pada uji.
4. Satu kali menjalankan notebook evaluasi yang sudah dibekukan:
   * Metrik ACD F1, ASC F1, Overall F1, akurasi (definisi NB07), P/R/F1 per kelas.
   * CI 95% cluster bootstrap per tweet (B = 10.000, seed 42), uji permutasi berpasangan tingkat cluster, koreksi Holm atas 15 perbandingan
     enam skema, McNemar berkluster (Durkalski) sebagai analisis sekunder — sama dengan `revision/01`.
   * Analisis kasus yang ditandai dan tabel transisi retain/revise (`revision/02`); perbandingan dengan LLM-only dan LLM-always (`revision/06`).
   * Tabel 2 (disagreement) dihitung pada **dev**, sesuai NB04; bila versi uji juga ditampilkan, splitnya disebut.
5. Semua hasil dilaporkan apa adanya, termasuk bila hybrid tidak lebih baik daripada run lama atau daripada LLM-only.

**Yang tidak boleh dilakukan setelah Langkah 8 dimulai:** mengubah θ, metode skor, hyperparameter, seed, bobot, ambang margin, prompt, atau pilihan
skema karena melihat hasil uji. Bila ternyata ada galat implementasi, perbaikannya dicatat di log beserta hasil sebelum dan sesudah perbaikan.

## 9. Pengungkapan di naskah

* θ dan metode skor silver kini dipilih hanya dengan validasi (θ lama 3,0 dipilih dengan validasi + uji).
* Kerapuhan pilihan D1–D2 (§1) dan perubahan besar distribusi label silver.
* Batch efektif 4 dan FP32 (bukan 16 / FP16 seperti yang tersirat di kode).
* Keputusan lama yang pernah dipilih dengan data uji (θ, skema Tabel 17, "best individual") dan inspeksi prediksi uji pada tahap pengembangan
  awal (`Diagnosis`, `diagnosis2`). Ini tidak dapat dibatalkan; latih ulang hanya membersihkan pilihan yang memengaruhi hasil akhir.
* Asal-usul label gold (lihat `leakage_audit.md` B4) dikonfirmasi peneliti sebelum naskah dikirim.

## 10. Bukti pembekuan (sha256, 16 karakter pertama)

| File | sha256 |
|---|---|
| `data/pairs/dev_pairs.csv` | `6a00462f29108022` |
| `data/gold/gold_tweet_ids.csv` | `e3721fbb6d42e501` |
| `data/processed/tweets_preprocessed.csv` | `87a2b09ecbf2a2fe` |
| `data/silver/silver_labeled.csv` (label lama, pembanding) | `86c1a8282b984e94` |
| `resources/aspect_keywords.json` | `ccf4ae7ae0386857` |
| `resources/inset_lexicon/positive.tsv` / `negative.tsv` | `4d0dc2a6b2fe88a4` / `b8d4b231077967d0` |
| `notebooks/02_Silverrr_Labeling_dan_SentencePair.ipynb` (sel loop pelabelan: `55acd499572ccb52…`) | `fffec9fef9f473f5` |
| `notebooks/06_LLM_as_Judge.ipynb` | `33c58c04346b8a1d` |
| `revisi_round3/final_silver_config.json` | `36cab3c1af17875a` |

File uji sengaja tidak di-hash pada tahap ini, agar data uji tidak disentuh sama sekali sebelum Langkah 8.

## 11. Log perubahan

| Tanggal | Perubahan | Alasan | Dibuat sebelum/sesudah melihat data uji |
|---|---|---|---|
| 2026-10-26 | Rencana dibekukan | — | sebelum |
| 2026-10-26 | **Asal-usul label gold dikonfirmasi peneliti:** seluruh label gold (validasi dan uji) dibuat oleh anotator manusia; draf yang pernah ada (`gold_labeled_draft_v2_by_claude.csv`, lihat `leakage_audit.md` B4) tidak dipakai. | Menjawab `leakage_audit.md` B4 | sebelum |
| 2026-10-26 | **Eksekusi dilakukan manual oleh peneliti** dengan notebook di `revisi_round3/notebooks_r3/` (urutan di `README_RUN_ORDER.md`). Notebook dibuat dan diuji asisten hanya dengan smoke test (subset kecil, folder sementara) dan mode `DRY_RUN` (data dev); tidak ada pelatihan penuh, data uji, atau panggilan API oleh asisten. | Pemisahan pembuatan kode dan eksekusi | sebelum |
| 2026-10-26 | Prosedur pelatihan ditambah **resume state** berkala (bobot, optimizer, scheduler, status RNG; juga di tengah epoch). Ini tidak mengubah konfigurasi §2: urutan operasi per langkah identik dengan `train_epoch` NB03 dan diverifikasi pada smoke test. | Sesi Kaggle/Colab dapat terputus | sebelum |
| 2026-10-30 | Observasi: xlm_roberta_seed42 tidak konvergen penuh (train loss datar 1,18–1,26 selama 4 epoch; prediksi dev tidak pernah positif/negatif; F1 dev terbaik 0,290 di epoch 4). Tidak di-resume (1 sesi), konfigurasi identik dengan seed 123 dan 7. Run DIPERTAHANKAN tanpa perubahan sesuai rencana (seed dibekukan); tidak ada pelatihan ulang atau penggantian seed. | Pemeriksaan anomali dev sebelum evaluasi uji | sebelum |
| 2026-10-30 | **Penggantian model LLM (D11): `qwen3-max-2026-01-23` → `qwen3.7-max-2026-06-08` untuk SEMUA komponen LLM.** Aturan: (1) satu model untuk arbitrase, LLM-always, dan LLM-only; (2) prompt, parameter (temperature 0, max_tokens 256, `enable_thinking` False), parsing, retry, dan fallback identik dengan NB06/`revision/04`; (3) respons qwen3-max lama TIDAK dicampur: cache lama tidak dibaca untuk kondisi utama (kunci cache = SHA-256 atas model+pesan+parameter, dan R3_05 memeriksa semua respons baru berasal dari model pengganti); LLM-only kini dipanggil ulang (560 pasangan); (4) cache qwen3-max lama disalin (tidak dihapus/diubah) ke `outputs_round3/llm_archive_qwen3-max-2026-01-23/` dan hanya dipakai untuk analisis sensitivitas LLM-only qwen3-max vs qwen3.7-max di R3_06; keluaran R3_05 sebelumnya juga dipindah ke sana; (5) model tidak akan diganti lagi setelah panggilan dimulai; bila panggilan berhenti (kuota/galat API), R3_05 dilanjutkan dengan model yang sama dari cache. Hash kode yang berubah (`frozen_choices.json`, juga dicatat di `amendments`): `config.py` 9a04f3ee…0b11 → 68bf8459…efb3; `R3_05_llm_calls.ipynb` a34d9b0e…8fb3 → 1afdda36…b0e4; `R3_06_evaluation.ipynb` 8e2c5663…0f68 → 975ebed1…2637. Tidak berubah: R3_04 (kode dan keluaran prediksi uji tetap sah), R3_07, `r3_lib.py`, `r3_eval.py`, legacy_code, data, checkpoint. Rincian: `ganti_model_llm.md`. | Snapshot `qwen3-max-2026-01-23` ditarik penyedia: 404 `model_not_found` untuk snapshot dan alias `qwen3-max` di endpoint workspace maupun publik internasional (bukti: `diagnosis_llm.md`). Pengganti dipilih peneliti karena merupakan snapshot bertanggal dari lini Max yang sama (penerus langsung) dan tersedia di akun; dipilih tanpa melihat kinerja (hanya uji kelayakan dengan teks netral dan tweet buatan). | sebelum (sebelum ada hasil LLM pada data uji dan sebelum evaluasi R3_06; prediksi R3_04 sudah dibuat tetapi label uji belum dimuat) |
