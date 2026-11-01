12 слоёв, ширина 768, 8 голов запроса и 4 головы K/V, K совпадает с V, полное внимание, словарь 32 000, контекст 512. 113.1M параметров. Словарь `tokenizers/bel_bpe_32k.model`.

## Веса

Файлы на Hugging Face: `base_r4.pt`, `chat_r2.pt`, `bel_bpe_32k.model`.

```
pip install -r requirements.txt huggingface_hub
huggingface-cli download Baindre/bel-lm base_r4.pt chat_r2.pt bel_bpe_32k.model --local-dir .
```

Инференс, чат и сырая база:

```
python generate.py --ckpt chat_r2.pt --tokenizer bel_bpe_32k.model --temperature 0 --repetition-penalty 1 --no-repeat-ngram 0 --max-new-tokens 32 --prompt "Якая сталіца Беларусі?"
python generate.py --ckpt base_r4.pt --tokenizer bel_bpe_32k.model --mode raw --prompt "Беларусь знаходзіцца ў"
```

Тот же прогон в `inference_demo_student.ipynb`

## Обучение


```
python data.py hf --config configs/sources.yaml
python data.py convert
python data.py clean --config configs/sources.yaml --workers 8
python data.py dedup --config configs/sources.yaml --workers 8
python data.py split --input data/dedup/corpus.jsonl.gz --out-dir data/splits
python data.py check --prompts eval_prompts.jsonl --train data/splits/train.jsonl.gz --remove

python tokenizer.py train --vocab-sizes 32000
python tokenizer.py encode --tokenizer tokenizers/bel_bpe_32k.model --out-dir data/tokens/bpe32k --workers 8
python train.py --config configs/base.yaml
python train.py --config configs/base_r4.yaml --init-from runs/mid_C_100m_32k_r2

python data.py chat
python tokenizer.py chat --tokenizer tokenizers/bel_bpe_32k.model --splits-dir data/splits/chat --out-dir data/tokens/bpe32k_chat
python train.py --config configs/chat.yaml --init-from runs/mid_C_100m_32k_r2
```

`base.yaml` учит 32k-базу с нуля. `base_r4.yaml` продолжает r2 с lr 3e-7. Чат учили с r2, не с r4. (он оказался просто чуть лучше)

## Данные

Белорусский текст: Wikipedia, Wikisource, тарашкевицкая Wikipedia, FineWeb-2, MADLAD-400, HPLT2, C4. Параллельные фразы en-be: CCMatrix. Диалоги: OpenOrca (в train 15%) и Alpaca.

Чистка (`data.py clean`): нормализация, снятие html, вики-разметки и ссылок, выкидывание битых строк и строк не на белорусском. Дедуп: точные копии, затем MinHash. Дальше split на train / validation / test.

Чат собирается отдельно: Alpaca, калибровочные пары, банк отказа из `configs/phrases.json`, короткие куски веба.

## 16k, маленькие и средние прогоны (для тестирования)

Словарь 16k, если не сказано иное. tiny около 31M, d=384, 16 слоёв. small около 65M, d=512, 20 слоёв.

| модель | что меняли | params | test ppl | test bpc | word ppl |
|---|---|---|---|---|---|
| tiny_A_full | полное внимание, отдельные K и V, контроль | 30.9M | 25.06 | 1.225 | 486.8 |
| mid_C_100m_v3 | 12×768, полное внимание, K=V | 100.8M | 14.41 | 1.025 | 180.1 |
| mid_C_100m_v3_kvnone | v3, но отдельные K и V | 104.3M | 14.02 | 1.015 | 171.0 |
| mid_C_100m_v3_linear | v3, все слои RetNet | 104.3M | 23.9 | 1.220 | - |

RetNet хуже полного внимания оказался. Отдельные K и V чуть лучше по bpc, кэш вдвое больше. В сданных весах K=V и softmax.

## 32k

Та же сеть 12×768, K=V, полное внимание. Меняется словарь, дообучение или чат.

| модель | что это | test ppl | test bpc | word ppl |
|---|---|---|---|---|
| mid_C_100m_32k | первая база, словарь 32k | 18.05 | 1.014 | 169.8 |
| mid_C_100m_32k_r1 | та же база, ещё 2k шагов, lr 1e-6 | 17.83 | 1.010 | 166.3 |
| mid_C_100m_32k_r2 | продолжение r1, 30k шагов, lr 1e-6 | 17.48 | 1.003 | 160.5 |
| mid_C_100m_32k_r4 | продолжение r2, lr 3e-7. Итоговая база | 17.47 | 1.0025 | 160.4 |
| mid_C_100m_32k_morph_v2 | словарь режет слова на морфемы | 15.39 | 1.021 | 176.1 |
| mid_C_100m_32k_morph_v2_flex_r3 | внимание аффикса только внутри своего слова | 19.22 | 1.104 | 268.4 |

r4 лучшая база по bpc. Morph-словарь и morph-маска хуже обычного 32k, но протестить такое очень хотелось

## Чат

Дообучение той же 113M. QA acc - доля попаданий по `eval_prompts.jsonl`.

| модель | что это | test bpc | word ppl | QA acc |
|---|---|---|---|---|
| base_r4 | база до чата, полный тест | 1.0025 | 160.4 | - |
| chat_r2 | SFT поверх r2, полный тест. Сданный чат | 1.033 | 186.8 | 0.628 |
| r4_chat, 1 эпоха | SFT поверх r4 | 1.029 | 183.8 | 0.535 |
| r4_chat, 2 эпохи | то же, вторая эпоха | 1.053 | 207.0 | 0.465 |

## Токенизаторы

Считалось на validation, tokens/word. В сданной модели словарь 32k.

| tokenizer | vocab | tok/word | доля слов из одного токена |
|---|---|---|---|
| bel_bpe_8k | 8000 | 2.073 | 0.541 |
| bel_bpe_16k | 16000 | 1.844 | 0.655 |
| bel_bpe_24k | 24000 | 1.736 | 0.716 |

## Маленькие модели

Словарь 16k, ширина 384, 16 слоёв, 8 голов Q и 2 kv, FFN 1024, 30.9M. Кэш - один пример, fp16, MiB: full держит всю историю, SWA держит окно и каждый n-й слой целиком, K=V хранит один тензор на слой. ECA - короткая свёртка после FFN, в кэш почти не входит.

Бюджет: 3.03B

| модель | что это | test ppl | test bpc | word ppl | кэш 512 | кэш 2048 |
|---|---|---|---|---|---|---|
| tiny_A_full | softmax на всю историю, отдельные K и V | 25.06 | 1.225 | 486.8 | 3.00 | 12.00 |
| tiny_B_swa3g1_w256 | окно 256: 12 локальных слоёв и 4 полных | 21.12 | 1.160 | 350.4 | 1.88 | 4.13 |
| tiny_C_kvshared | как B, K совпадает с V, FFN 1056 | 21.96 | 1.175 | 377.9 | 0.94 | 2.06 |
| tiny_D_kvshared_eca2 | как C, ECA на каждом 2-м слое | 21.91 | 1.174 | 376.3 | 0.95 | 2.07 |
| tiny_D_kvshared_eca4 | как C, ECA на каждом 4-м слое | 21.94 | 1.175 | 377.1 | 0.94 | 2.07 |
