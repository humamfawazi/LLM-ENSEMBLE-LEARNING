"""Preprocessing V1–V6 (salinan VERBATIM sel 33 dan 34 notebook asli `01_EDA_dan_Preprocessing.ipynb`).

Dipakai untuk membangun ulang kolom text_v1–text_v6 dari teks tweet yang di-rehydrate (lihat README.md).
text_v1 = input model (sentence_a); text_v3 = teks skor leksikon silver; text_v1–v6 = ablasi Tabel 16.
sha256 sel 33: 62c9b86e963d10ed786767d0fc340e2015c2a02a59ff0e68e69bd7776967a4a9
sha256 sel 34: 2846e65384d8284312fa0b5daa88582190542162d5454dbf2aa69e555e930f7b
"""
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ── sel 33 ──
SLANG_DICT = {
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'dlm': 'dalam',
    'tdk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak',
    'nggak': 'tidak', 'ngga': 'tidak', 'engga': 'tidak', 'enggak': 'tidak',
    'blm': 'belum', 'blum': 'belum', 'udh': 'sudah', 'udah': 'sudah',
    'sdh': 'sudah', 'krn': 'karena', 'krna': 'karena', 'karna': 'karena',
    'org': 'orang', 'orng': 'orang', 'jg': 'juga', 'jga': 'juga',
    'aja': 'saja', 'aj': 'saja', 'doang': 'saja',
    'bgt': 'banget', 'bngt': 'banget', 'bngtt': 'banget',
    'sm': 'sama', 'sma': 'sama', 'dr': 'dari', 'dri': 'dari',
    'tp': 'tapi', 'tpi': 'tapi', 'klo': 'kalau', 'kalo': 'kalau',
    'klu': 'kalau', 'bs': 'bisa', 'bsa': 'bisa', 'bkn': 'bukan',
    'lg': 'lagi', 'lgi': 'lagi', 'msh': 'masih', 'msih': 'masih',
    'hrs': 'harus', 'hrus': 'harus', 'dpt': 'dapat', 'dpat': 'dapat',
    'stlh': 'setelah', 'sblm': 'sebelum', 'sblmnya': 'sebelumnya',
    'sy': 'saya', 'ak': 'aku', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya',
    'lo': 'kamu', 'lu': 'kamu', 'loe': 'kamu',
    'emg': 'memang', 'emang': 'memang', 'mmg': 'memang',
    'gmn': 'bagaimana', 'gimana': 'bagaimana',
    'dmn': 'dimana', 'dmna': 'dimana',
    'knp': 'kenapa', 'knpa': 'kenapa',
    'rs': 'rumah sakit', 'skt': 'sakit',
    'obt': 'obat', 'mnum': 'minum', 'mkn': 'makan',
    'pnykt': 'penyakit', 'pnyakit': 'penyakit',
    'ksehatan': 'kesehatan', 'kshatan': 'kesehatan',
    'smga': 'semoga', 'smoga': 'semoga',
    'trmksh': 'terima kasih', 'makasih': 'terima kasih',
    'mksh': 'terima kasih', 'thx': 'terima kasih', 'ty': 'terima kasih',
    'bakal': 'akan', 'bkl': 'akan',
    'ngomong': 'berbicara', 'bilang': 'mengatakan',
}

STOPWORDS_ID = set([
    'yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'adalah',
    'dari', 'dalam', 'ke', 'tidak', 'ada', 'juga', 'akan', 'atau', 'bisa',
    'sudah', 'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita', 'kami',
    'se', 'ya', 'lah', 'pun', 'kan', 'sih', 'dong', 'nih', 'deh', 'tuh',
    'kalau', 'jika', 'maka', 'karena', 'sebab', 'oleh', 'agar', 'supaya',
    'tetapi', 'tapi', 'namun', 'sedangkan', 'meskipun', 'walau',
    'bahwa', 'bahkan', 'hanya', 'pernah', 'belum',
    'sangat', 'sekali', 'paling', 'lebih', 'kurang', 'agak', 'cukup',
    'sering', 'jarang', 'selalu', 'masih', 'lagi', 'sedang', 'telah',
    'begitu', 'seperti', 'lalu', 'kemudian', 'setelah', 'sebelum',
    'antara', 'tentang', 'tanpa', 'hingga', 'sampai', 'sejak',
    'banyak', 'sedikit', 'semua', 'setiap', 'para', 'beberapa',
    'harus', 'dapat', 'bisa', 'perlu', 'boleh', 'mau', 'ingin',
    'hal', 'cara', 'waktu', 'saat', 'hari', 'orang', 'lain',
    'apa', 'siapa', 'mana', 'kapan', 'kenapa', 'bagaimana',
    'dimana', 'kemana', 'serta', 'maupun', 'ataupun',
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'to',
    'in', 'for', 'on', 'with', 'at', 'by', 'it', 'this', 'that',
    'rt', 'amp',
])

print(f"✓ Kamus slang: {len(SLANG_DICT)} entri | Stopwords: {len(STOPWORDS_ID)} kata")

# ── sel 34 ──
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE
)

def preprocess_v1(text: str) -> str:
    t = str(text)
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def preprocess_v2(text: str) -> str:
    t = preprocess_v1(text)
    t = re.sub(r'#', '', t)
    t = EMOJI_PATTERN.sub('', t)
    t = re.sub(r'[^\w\s.,!?;:\-]', ' ', t)  # hapus simbol aneh
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def preprocess_v3(text: str) -> str:
    t = preprocess_v2(text)
    t = t.lower()
    t = re.sub(r'(.)\1{2,}', r'\1', t)  # huruf berulang
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def preprocess_v4(text: str) -> str:
    t = preprocess_v3(text)
    words = t.split()
    return ' '.join(SLANG_DICT.get(w, w) for w in words)

def preprocess_v5(text: str) -> str:
    t = preprocess_v4(text)
    words = t.split()
    return ' '.join(w for w in words if w not in STOPWORDS_ID and len(w) > 1)

# Sastrawi stemmer (singleton + cache)
print("Menginisialisasi Sastrawi stemmer...")
_stemmer = StemmerFactory().create_stemmer()
_stem_cache: dict = {}

def stem_word_cached(w: str) -> str:
    if w not in _stem_cache:
        _stem_cache[w] = _stemmer.stem(w)
    return _stem_cache[w]

def preprocess_v6(text: str) -> str:
    t = preprocess_v5(text)
    return ' '.join(stem_word_cached(w) for w in t.split())

print("✓ Semua fungsi preprocessing V1–V6 siap.")
