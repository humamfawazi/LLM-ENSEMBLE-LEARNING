def call_llm(prompt: str, provider: str, client, model_name: str,
             max_retries: int = 3, retry_delay: float = 5.0) -> dict | None:
    """
    Panggil LLM dan parse respons JSON.
    Return dict {'decision', 'final_label', 'reasoning'} atau None jika gagal.
    """
    for attempt in range(1, max_retries + 1):
        try:
            raw_text = ''

            if provider == 'alibaba':
                # Alibaba DashScope — OpenAI-compatible endpoint
                # enable_thinking=False agar respons bersih tanpa <think> block
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            'role': 'system',
                            'content': 'Kamu adalah verifikator label sentimen ABSA untuk teks Indonesia. '
                                       'Jawab HANYA dengan JSON yang diminta, tanpa teks tambahan.'
                        },
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    extra_body={'enable_thinking': False}
                )
                raw_text = response.choices[0].message.content.strip()

            elif provider == 'gemini':
                response = client.generate_content(prompt)
                raw_text = response.text.strip()

            elif provider == 'openai':
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    max_tokens=200
                )
                raw_text = response.choices[0].message.content.strip()

            elif provider == 'anthropic':
                import anthropic
                response = client.messages.create(
                    model=model_name,
                    max_tokens=200,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                raw_text = response.content[0].text.strip()

            # ── Bersihkan markdown code block jika ada ──
            if raw_text.startswith('```'):
                lines = raw_text.split('\n')
                raw_text = '\n'.join(
                    l for l in lines
                    if not l.startswith('```')
                ).strip()

            # ── Ekstrak JSON dari respons ──
            json_start = raw_text.find('{')
            json_end   = raw_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                parsed = json.loads(raw_text[json_start:json_end])

                # Validasi field
                decision    = str(parsed.get('decision', 'retain')).lower()
                final_label = str(parsed.get('final_label', 'tidak_disebutkan')).lower()
                reasoning   = str(parsed.get('reasoning', ''))

                # Normalisasi nilai decision
                if decision not in ('retain', 'revise'):
                    decision = 'retain'

                # Pastikan label valid
                valid_labels = {'tidak_disebutkan', 'positif', 'negatif', 'netral'}
                if final_label not in valid_labels:
                    final_label = 'tidak_disebutkan'

                return {
                    'decision'   : decision,
                    'final_label': final_label,
                    'reasoning'  : reasoning
                }

            else:
                print(f"    Attempt {attempt}: Tidak ada JSON valid dalam respons.")
                print(f"    Raw: {raw_text[:200]}")

        except json.JSONDecodeError as e:
            print(f"    Attempt {attempt}: JSON parse error — {e}")
        except Exception as e:
            print(f"    Attempt {attempt}: API error — {e}")

        if attempt < max_retries:
            wait = retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"    Retry dalam {wait:.1f} detik...")
            time.sleep(wait)

    return None


print("✓ Fungsi call_llm dengan retry/backoff siap.")