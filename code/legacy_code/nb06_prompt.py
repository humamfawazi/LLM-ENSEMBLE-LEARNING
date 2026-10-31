ASPECT_DESCRIPTIONS = {
    'pola_konsumsi'           : 'Tweet ini membahas pola konsumsi makanan atau minuman',
    'aktivitas_fisik'         : 'Tweet ini membahas aktivitas fisik atau olahraga',
    'akses_layanan_kesehatan' : 'Tweet ini membahas akses terhadap layanan kesehatan',
    'pengetahuan_pengobatan'  : 'Tweet ini membahas pengetahuan atau pengobatan penyakit',
}


def needs_llm_judge(vote_type: str, margin: float, margin_threshold: float = 0.15) -> bool:
    """Aturan pemicu LLM judge — WAJIB, jangan panggil LLM untuk semua baris."""
    return vote_type == 'terpecah' or margin < margin_threshold


def build_prompt(tweet_text: str, aspect: str, proposed_label_str: str,
                 pred_indobert: str, pred_xlm_r: str, pred_mbert: str) -> str:
    """Bangun prompt sesuai template spesifikasi."""
    aspect_desc = ASPECT_DESCRIPTIONS.get(aspect, aspect)
    prompt = f"""Kamu adalah verifikator label sentimen. Berikut informasi sebuah tweet dan hasil voting 3 model AI.

Tweet: \"{tweet_text}\"
Aspek yang dinilai: {aspect_desc}
Label yang diusulkan ensemble: {proposed_label_str}
Rincian suara 3 model: IndoBERT={pred_indobert}, XLM-RoBERTa={pred_xlm_r}, mBERT={pred_mbert}

Tugasmu: putuskan apakah label yang diusulkan sudah tepat (RETAIN) atau perlu diganti (REVISE).
Jika REVISE, sebutkan label yang benar dari salah satu: tidak_disebutkan, positif, negatif, netral.

Jawab HANYA dalam format JSON (tanpa teks lain, tanpa markdown code block):\n",
{{"decision": "retain" atau "revise", "final_label": "...", "reasoning": "alasan singkat 1 kalimat"}}"""
    return prompt


print("✓ Fungsi pemicu dan prompt builder siap.")