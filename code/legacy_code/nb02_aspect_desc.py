ASPECT_DESCRIPTIONS = {
    'pola_konsumsi': 'Tweet ini membahas pola konsumsi makanan atau minuman',
    'aktivitas_fisik': 'Tweet ini membahas aktivitas fisik atau olahraga',
    'akses_layanan_kesehatan': 'Tweet ini membahas akses terhadap layanan kesehatan',
    'pengetahuan_pengobatan': 'Tweet ini membahas pengetahuan atau pengobatan penyakit',
}

LABEL_MAP = {
    'tidak_disebutkan': 0,
    'positif': 1,
    'negatif': 2,
    'netral': 3
}

LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

print("Aspect descriptions:")
for asp, desc in ASPECT_DESCRIPTIONS.items():
    print(f"  {asp}: \"{desc}\"")
print(f"\nLabel mapping: {LABEL_MAP}")