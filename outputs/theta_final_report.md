# Laporan akhir kalibrasi metode skor silver dan θ (validasi saja)

Dibuat oleh `revisi_round3/calibrate_method_theta_validation.ipynb`. Hanya 76 pasangan gold validasi bersentimen (60 tweet) yang dipakai
untuk memilih. `test_pairs.csv` tidak dimuat; baris gold uji di `gold_labeled.csv` ikut terbaca lalu langsung disaring, tanpa perhitungan
apa pun pada data uji. File lama tidak diubah; tidak ada pelatihan; tidak ada API.

## 1. Kesamaan dengan loop pelabelan silver
Kalibrasi menjalankan **sel loop pelabelan asli** dari `notebooks/02_Silverrr_Labeling_dan_SentencePair.ipynb` tanpa perubahan
(sha256 sel `55acd499572ccb52…`): deteksi aspek `detect_aspect` pada kolom `text` (teks asli); skor pada `text_v3`; `tidak_disebutkan`
bila aspek tidak terdeteksi; label `score_to_label(skor, θ)`. Varian `score_sentence` dibuat dengan mengganti fungsi skor yang dipanggil loop,
dengan leksikon dan aturan negasi yang sama.
* **Bukti 1:** loop yang sama dengan `score_windowed`, θ = 3,0 pada 37.747 tweet non-gold mereproduksi
  `data/silver/silver_labeled.csv` **100,00%** (label) dan 100,00% (skor) dari 150.988 pasangan.
* **Bukti 2:** pada 76 pasangan validasi, skor yang dipakai kalibrasi sama dengan `score_windowed(text_v3)` bila `detect_aspect(text)` benar
  (dan 0 bila tidak) — cocok.
* Perbedaan dengan sweep lama: sweep lama menilai `text_v1` dan **tanpa** deteksi aspek. Pada kalibrasi ini 15 dari 76 pasangan
  bersentimen tidak terdeteksi aspeknya oleh kata kunci, sehingga diberi `tidak_disebutkan` dan selalu dihitung salah, persis seperti
  yang akan terjadi pada data latih.

## 2. Hasil kalibrasi (macro-F1 3 kelas, 76 pasangan validasi)
Aturan: macro-F1 tertinggi; bila seri, `score_windowed` lebih dulu, lalu θ yang lebih kecil.

| θ | score_windowed (loop, text_v3) | score_sentence (loop, text_v3) | score_windowed (gaya lama, text_v1) | score_sentence (gaya lama, text_v1) |
|---|---|---|---|---|
| 0,5 | 0,2921 | 0,2412 | 0,3227 | 0,2899 |
| 1,0 | 0,3254 | 0,3244 | 0,3583 | 0,3864 |
| 1,5 | 0,3254 | 0,3244 | 0,3583 | 0,3864 |
| 2,0 | 0,3573 | 0,3488 | 0,4260 | 0,4255 |
| 2,5 | 0,3573 | 0,3488 | 0,4260 | 0,4255 |
| 3,0 | 0,3672 | 0,3243 | 0,4323 | 0,3821 |
| 4,0 | 0,3743 | 0,3489 | 0,4325 | 0,3954 |
| 5,0 | 0,4022 | 0,3727 | 0,4779 | 0,4379 |
| 7,0 | 0,3149 | 0,3410 | 0,4102 | 0,4186 |
| 10,0 | 0,2709 | 0,3608 | 0,3781 | 0,4453 |
| 15,0 | 0,2890 | 0,4079 **←** | 0,3713 | 0,4767 |

**Kombinasi terpilih: `score_sentence`, θ = 15,0** (macro-F1 validasi 0,4079; 1 kandidat dengan skor tertinggi).
Berbeda dari konfigurasi label latih saat ini (score_windowed, θ = 3,0).

Transparansi (gaya lama, `text_v1`, tanpa deteksi): terbaik `score_windowed`, θ = 5,0 (macro-F1 0,4779).
Kolom `score_windowed` gaya lama identik dengan `theta_validation_only.csv` ronde sebelumnya (yang menghasilkan θ = 5,0).
Tabel lengkap (termasuk akurasi dan distribusi prediksi): `method_theta_validation_grid.csv`.

### Stabilitas pilihan (validasi saja; tidak dipakai untuk memilih)
* Cluster bootstrap per tweet validasi (54 tweet, B = 10.000), aturan pemilihan yang sama: kombinasi terpilih muncul lagi pada
  **40,5%** resample. Tiga kombinasi paling sering terpilih: `score_sentence` θ = 15,0 (40,5%); `score_windowed` θ = 5,0 (32,5%); `score_sentence` θ = 5,0 (9,2%).
* Selisih terhadap pesaing terdekat (`score_windowed`, θ = 5,0): 0,0057 [CI 95% -0,1372; 0,1297].
  Selisih terhadap konfigurasi saat ini (`score_windowed`, θ = 3,0): 0,0408 [-0,0934; 0,1572].
* **Tepi grid:** θ terpilih adalah kandidat terbesar (15,0). Diagnostik di luar grid
  (tidak dipakai untuk memilih): θ = 15 → 0,4079 (netral 68%); θ = 20 → 0,2825 (netral 74%); θ = 25 → 0,2313 (netral 79%); θ = 30 → 0,2313 (netral 79%); θ = 50 → 0,2290 (netral 80%).

## 3. Dampak pada 150.988 label latih
* **29.284 label berubah (19,39%)** dibanding label latih saat ini.

| saat ini \\ baru | tidak_disebutkan | positif | negatif | netral |
|---|---|---|---|---|
| tidak_disebutkan | 93.203 | 0 | 0 | 0 |
| positif | 0 | 4.005 | 228 | 15.677 |
| negatif | 0 | 132 | 3.821 | 11.863 |
| netral | 0 | 621 | 763 | 20.675 |

| label | saat ini (windowed, θ=3,0) | baru | selisih |
|---|---|---|---|
| tidak_disebutkan | 93.203 | 93.203 | +0 |
| positif | 19.910 | 4.758 | −15.152 |
| negatif | 15.816 | 4.812 | −11.004 |
| netral | 22.059 | 48.215 | +26.156 |

Per aspek: `final_silver_changes_per_aspect.csv`.

## 4. Kesimpulan
* Metode dan θ untuk latih ulang: **`score_sentence`, θ = 15,0**, dipilih hanya dengan data validasi dan dengan teks yang identik
  dengan loop pelabelan. Keputusan ini dibekukan di `rencana_latih_ulang.md` sebelum data uji disentuh.
* Pilihan ini adalah hasil aturan yang ditetapkan sebelumnya, bukan bukti bahwa kandidat lain lebih buruk: dengan 76 pasangan validasi,
  selisih terhadap pesaing terdekat (0,0057) jauh di dalam ketidakpastian sampling, dan kombinasi terpilih hanya muncul lagi pada
  40,5% resample.
* **Perhatian sebelum melatih ulang:** θ terpilih berada di tepi grid, dan label latih bergeser besar ke netral
  (29.284 label berubah; positif 19.910 → 4.758, negatif
  15.816 → 4.812). Bila aturan pemilihan ingin diubah (misalnya memperluas grid atau
  membatasi θ), keputusan itu harus dibuat **sekarang**, hanya dengan data validasi, dan dicatat sebagai perubahan rencana. Data uji masih
  belum disentuh, jadi perubahan seperti ini belum menimbulkan kebocoran, tetapi harus diungkapkan.
