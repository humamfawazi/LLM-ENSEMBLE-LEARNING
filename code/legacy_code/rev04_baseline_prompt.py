def baseline_prompt(tweet_text: str, aspect: str) -> str:
    aspect_desc = ASPECT_DESCRIPTIONS.get(aspect, aspect)
    return f"""Kamu adalah verifikator label sentimen. Berikut informasi sebuah tweet.

Tweet: \"{tweet_text}\"
Aspek yang dinilai: {aspect_desc}

Tugasmu: tentukan label sentimen tweet terhadap aspek tersebut, dari salah satu: tidak_disebutkan, positif, negatif, netral.

Jawab HANYA dalam format JSON (tanpa teks lain, tanpa markdown code block):
{{"final_label": "...", "reasoning": "alasan singkat 1 kalimat"}}"""
