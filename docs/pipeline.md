# Pipeline

## Цель

Решить задачу из `task_description.md`: для двух Telegram-каналов сделать перенос стиля. Конкретно — обучить два LoRA-адаптера (`type1`, `type2`), сгенерировать `outputs_type*.txt` для внешних `inputs_type*.txt`, и подтвердить, что fine-tune реально улучшил стиль на отложенных данных и не свёлся к запоминанию.

Ключевой принцип: **сначала** фиксируем `train/val/test` по времени, **потом** любые LLM-based neutralization / synthetic inputs. Это исключает leakage между обучением и оценкой.

## Production-скрипты

| Скрипт | Назначение |
|---|---|
| `scripts/prepare_dataset.py` | строит базовые `train/val/test` jsonl |
| `scripts/generate_openrouter_synthetic.py` | (опционально) split-safe synthetic neutral inputs через OpenRouter |
| `scripts/finetune.py` | LoRA-адаптер для одного типа |
| `scripts/generate_baseline.py` | генерация base model без fine-tune |
| `scripts/generate.py` | генерация fine-tuned модели |
| `scripts/evaluate.py` | offline evaluation (length / style / crosstype / cosine / MAUVE) |
| `scripts/memorization_check.py` | проверка, что модель не выучила train-примеры |
| `scripts/build_results.py` | сборка `RESULTS.md` из артефактов |

`scripts/pilot_*.py` — экспериментальные, для основного прогона не нужны.

---

## Шаг 0. Окружение

Основной сценарий для пайплайна обучения - `GPU`. Ниже команды для `conda`-окружения с Python 3.11.

```bash
conda create -n telegram-style-transfer python=3.11 -y
conda activate telegram-style-transfer
# пример для Ubuntu + CUDA 12.1
conda install -y -c pytorch -c nvidia pytorch==2.5.1 pytorch-cuda=12.1
pip install -r requirements.txt
pip install -e .
pip install -e ".[dev]"
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

Запускать fine-tune потом с `--env ubuntu_t4`.

`torch` намеренно не включён в `requirements.txt`; конкретная команда и версия PyTorch зависят от платформы, драйвера и версии CUDA. Пример выше подходит для `Ubuntu + CUDA 12.1`, а для других окружений ориентируйся на [docs/inference.md](/Users/anyarulina/telegram-style-transfer/docs/inference.md:70).

Для шага LLM нейтрализации через API OpenRouter добавь API ключ:

```bash
echo 'OPENROUTER_API_KEY=...' >> .env
```

Для Ubuntu при установке `unsloth` может понадобиться системный toolchain:

```bash
sudo apt update
sudo apt install -y build-essential libc6-dev linux-libc-dev python3-dev
```

## Шаг 1. Положить сырые Telegram exports

Ожидаемые пути:

- `data/raw/type1/telegram_export.json` — экспорт `https://t.me/banki_oil` из Telegram Desktop;
- `data/raw/type2/telegram_export.json` — экспорт `https://t.me/moscowach`.

Это единственный шаг, который требует ручных данных от тебя. Без них пайплайн не запустится.

## Размеры датасета: `configs/data.yaml`

Целевой размер обучающей выборки (после фильтрации LLM-нейтрализации) задаётся в `configs/data.yaml`:

```yaml
target_train_ok: 1600   # сколько ok-примеров хотим в train на каждый тип
target_eval_samples: 100 # сколько примеров брать в быстрый benchmark / smoke-eval

max_samples:
  type1: 2000   # banki_oil  ≈ 0% drop  → 1600 train × 1.00 ≈ 1600 ok
  type2: 3300   # moscowach  ≈ 38% drop → 2640 train × 0.62 ≈ 1637 ok
```

`max_samples` асимметричен потому, что type2 теряет ~38% записей при нейтрализации (`high_overlap` — moscowach короткий и LLM возвращает почти то же самое). Чтобы получить 1600 ok-примеров в train, надо взять с запасом 3300 сырых.

`prepare_dataset.py` и `generate_openrouter_synthetic.py` читают этот файл по умолчанию. `generate.py`, `generate_baseline.py` и `evaluate.py` тоже могут брать из него `target_eval_samples`, если запустить их с `--limit-from-config`. Если хочется поменять цели — правишь yaml, не CLI.

## Шаг 2. Зафиксировать базовый split

Делит каждый канал на `train/val/test` по времени (80/10/10 после dedup) и пишет `prompt`-поле для SFT.

```bash
python scripts/prepare_dataset.py --min-chars 50
```

Без `--max-samples` скрипт берёт значения per-type из `configs/data.yaml` (2000 для type1, 3300 для type2). При желании можно перебить одним числом для всех типов: `--max-samples 2000`.

Результат:

- `data/processed/type1_train.jsonl`, `type1_val.jsonl`, `type1_test.jsonl` (1600 / 200 / 200)
- `data/processed/type2_train.jsonl`, `type2_val.jsonl`, `type2_test.jsonl` (2640 / 330 / 330 — больше, чем у type1, чтобы после LLM-фильтра в train осталось ~1600)
- `data/processed/split_report.json` — сводка по размерам и стратегии разбивки.

Поле `input` в записях по умолчанию строится heuristic-нейтрализацией (`build_input_heuristic` чистит handles/emoji/CTA из `response_clean`). Этого достаточно, чтобы перейти к fine-tune без OpenRouter.

> **Опционально:** если уже есть готовый `brief_v4`-файл, можно подмешать его как override:
>
> ```bash
> python scripts/prepare_dataset.py --min-chars 50 \
>     --brief-v4-path data/processed/pilot_openrouter_brief_v4_train.jsonl \
>     --brief-v4-max-jaccard 0.3
> ```
>
> Override применяется только к тем `(style_type, post_id)`, которые есть в файле — обычно только `train`.

## Шаг 3 (опционально). Synthetic inputs через OpenRouter

Нужен, если хочешь получить более качественные нейтральные inputs не только в train, но и в val/test. Стоит денег (≈ $0.5–1 на полный прогон) и времени.

```bash
python scripts/generate_openrouter_synthetic.py \
  --types type1 type2 \
  --splits train val test \
  --out-dir data/processed/synthetic_openrouter \
  --report data/processed/synthetic_openrouter/report.json \
  --drop-failed
```

Что делает:

- читает уже зафиксированные `data/processed/type*_*.jsonl`;
- для каждой записи строит neutral input через LLM;
- **train: обрабатывает записи в порядке от новых к старым и останавливается, как только в файле накопится `target_train_ok` ok-записей** (значение из `configs/data.yaml`, по умолчанию 1600). Старый хвост train не трогает — экономит API.
- val/test: проходит полностью, без early-stop;
- кэширует уже обработанные `post_id` в выходных файлах — повторный запуск не делает повторных API-вызовов.

Train/val/test не смешиваются. Результат — шесть jsonl-файлов в `synthetic_openrouter/`. Чтобы они стали обучающими данными, передавай `--data-dir data/processed/synthetic_openrouter` в `finetune.py` (см. Шаг 4).

> **Если уже есть частичные результаты от предыдущего прогона** — просто перезапусти ту же команду. Скрипт читает `synthetic_openrouter/type*_*.jsonl` как кэш по `post_id` и продолжит с того места, где остановился. Если `prepare_dataset.py` потом запускался с другим `max_samples`, и сплиты сдвинулись — прежние результаты по post_id, попадающим в новые сплиты, переиспользуются автоматически; «лишние» строки в кэше игнорируются.

## Шаг 4. Обучить адаптеры

Конфиг гиперпараметров тренировки — `configs/train.yaml`; профиль среды — `configs/env.yaml`, `ubuntu_t4` (T4 16 GB, CUDA).

Обучаем на synthetic inputs из Шага 3 — обязательно с `--data-dir data/processed/synthetic_openrouter`. Без этого флага скрипт берёт `data/processed/type*_train.jsonl` с `input_source='heuristic'`, где «нейтральный» вход — это просто склейка фактов из самого ответа. Адаптер выучит тривиальный инверс «склей буллеты обратно в абзац», и на test 80/100 генераций совпадут с reference побайтно (проверено: `models/type1/all` именно так и сломался).

```bash
python scripts/finetune.py --type type1 --env ubuntu_t4 --config configs/train.yaml \
    --data-dir data/processed/synthetic_openrouter
python scripts/finetune.py --type type2 --env ubuntu_t4 --config configs/train.yaml \
    --data-dir data/processed/synthetic_openrouter
```

`finetune.py` сам выставляет `run_group="synthetic_openrouter"`, когда `--data-dir` указывает на этот каталог (см. код `scripts/finetune.py`).

Результат каждого запуска:

- `models/<type>/synthetic_openrouter/adapter/` — LoRA-адаптер;
- `models/<type>/synthetic_openrouter/run_manifest.json` — `base_model + hyperparams + train/val sample counts + train/eval loss + duration + timestamp` (используется для воспроизводимости и для `RESULTS.md`).

## Шаг 5. Offline benchmark: сравнить `до / после` на отложенном test

Это единственный артефакт, который доказывает, что fine-tune реально работает. `evaluate.py` читает входы и target прямо из `type*_test.jsonl`.

### 5.1. Self-check classifier (быстрая проверка, без генерации)

Убедись, что TF-IDF + LogReg вообще разделяет type1/type2 на reference-постах:

```bash
python scripts/evaluate.py --metrics length style
```

Если accuracy на reference < 0.85 — что-то не так с данными, дальше идти бессмысленно.

### 5.2. Генерация на test split

Все команды ниже используют `--limit-from-config` → берётся `target_eval_samples` из `configs/data.yaml` (сейчас **100**). Суффикс `_100` в именах артефактов соответствует этому значению — если поменяешь `target_eval_samples`, поменяй и суффикс. Без `--limit-from-config` скрипт прогоняет ВСЕ записи из jsonl (200 для type1_test, 330 для type2_test).

Создай каталог под bench-артефакты один раз:

```bash
mkdir -p output
```

Baseline (`output/bench_*_baseline_100.txt`):

```bash
python scripts/generate_baseline.py --type type1 --env ubuntu_t4 \
  --from-jsonl data/processed/type1_test.jsonl \
  --limit-from-config \
  --output output/bench_type1_test_baseline_100.txt

python scripts/generate_baseline.py --type type2 --env ubuntu_t4 \
  --from-jsonl data/processed/type2_test.jsonl \
  --limit-from-config \
  --output output/bench_type2_test_baseline_100.txt
```

Fine-tuned на synthetic_openrouter (`output/bench_*_finetuned_synth_100.txt`):

```bash
python scripts/generate.py --type type1 --env ubuntu_t4 \
  --adapter models/type1/synthetic_openrouter/adapter \
  --from-jsonl data/processed/type1_test.jsonl \
  --limit-from-config \
  --output output/bench_type1_test_finetuned_synth_100.txt

python scripts/generate.py --type type2 --env ubuntu_t4 \
  --adapter models/type2/synthetic_openrouter/adapter \
  --from-jsonl data/processed/type2_test.jsonl \
  --limit-from-config \
  --output output/bench_type2_test_finetuned_synth_100.txt
```

### 5.3. Метрики baseline и fine-tuned

`--metrics all` считает length / style / crosstype / cosine / MAUVE одной командой, включая cross-type confusion (пункт 7 чек-листа):

Полный прогон обоих типов (с cross-type confusion):

```bash
python scripts/evaluate.py --split test \
  --limit-from-config \
  --generated-type1 output/bench_type1_test_baseline_100.txt \
  --generated-type2 output/bench_type2_test_baseline_100.txt \
  --metrics all \
  --out-dir reports/eval/baseline_test_100

python scripts/evaluate.py --split test \
  --limit-from-config \
  --generated-type1 output/bench_type1_test_finetuned_synth_100.txt \
  --generated-type2 output/bench_type2_test_finetuned_synth_100.txt \
  --metrics all \
  --out-dir reports/eval/finetuned_test_synth_100
```

Быстрый прогон только type1 (без cross-type метрики, нужен только bench для type1):

```bash
python scripts/evaluate.py --types type1 --split test \
  --limit-from-config \
  --generated-type1 output/bench_type1_test_baseline_100.txt \
  --metrics length style cosine mauve \
  --out-dir reports/eval/baseline_test_type1_100

python scripts/evaluate.py --types type1 --split test \
  --limit-from-config \
  --generated-type1 output/bench_type1_test_finetuned_synth_100.txt \
  --metrics length style cosine mauve \
  --out-dir reports/eval/finetuned_synth_test_type1_100
```

В каждом `--out-dir` будет `eval_summary.json` (агрегаты), `cross_type_confusion.json` (для пункта 7), `length_*.csv`, `cosine_*.csv`.

### 5.4. Что значит «стало лучше»

| Метрика | Ключ | Что должно произойти |
|---|---|---|
| Style classifier accuracy | `style_gen.style_accuracy` | ↑ vs baseline. **Главная метрика.** |
| Cosine (LaBSE) | `cosine.cosine_mean` | **≈** baseline. Если сильно упала — fine-tune потерял содержание. |
| MAUVE | `mauve.mauve` | ↑. Распределение генераций ближе к корпусу канала. |
| Cross-type diagonal | `cross_type_confusion.diagonal_mean` | ↑ заметно выше, чем у baseline (~50%). Подтверждает разделимость стилей. |
| Длина | `length_*.csv` | Распределение длины ближе к reference, чем у baseline. |

**Критерий победы:** `style_accuracy ↑` **И** `cosine_mean ≈` **И** `mauve ↑`.

Если style accuracy вырос, а cosine сильно упал — модель имитирует форму без содержания. Это не зачёт.

### 5.5. Memorization check

Подтверждаем пункт 6 чек-листа: модель не выучила train-примеры наизусть.

```bash
python scripts/memorization_check.py \
  --generated-type1 output/bench_type1_test_finetuned_synth_100.txt \
  --generated-type2 output/bench_type2_test_finetuned_synth_100.txt \
  --out-dir reports/memorization_synth_100
```

Скрипт для каждого сгенерированного текста находит ближайший train-пост по `rapidfuzz.ratio` + Jaccard на 5-граммах + длине LCS, пишет `reports/memorization/memorization_type*.csv`.

Целевые значения:

- median `fuzz_ratio` < 60;
- max `fuzz_ratio` < 100 (нет exact matches);
- max `longest_common_substr_chars` < 30.

Если хоть одно нарушено — скрипт падает с exit 1 и сообщением `[FAIL]`. В `RESULTS.md` build-скрипт выведет top-5 строк по `fuzz_ratio`.

## Шаг 6. Сгенерировать финальные deliverables

Это шаг для **сдачи задачи**: внешние входы из `inputs_type*.txt` (приходят с задачей, в репо их нет) → `outputs_type*.txt`.

```bash
python scripts/generate.py --type type1 --env ubuntu_t4 \
  --input inputs_type1.txt --output outputs_type1.txt

python scripts/generate.py --type type2 --env ubuntu_t4 \
  --input inputs_type2.txt --output outputs_type2.txt
```

Если хочется приложить и baseline-версии (для наглядности в `RESULTS.md`):

```bash
python scripts/generate_baseline.py --type type1 --env ubuntu_t4 \
  --input inputs_type1.txt --output outputs_baseline_type1.txt

python scripts/generate_baseline.py --type type2 --env ubuntu_t4 \
  --input inputs_type2.txt --output outputs_baseline_type2.txt
```

Это не входит в строгий offline benchmark (он на test split), но полезно как qualitative check: переносится ли стиль на новые темы, а не только на train-distribution.

## Шаг 7. Собрать RESULTS.md

После Шагов 4–5 запусти:

```bash
python scripts/build_results.py
```

Скрипт читает артефакты и собирает `RESULTS.md`:

| Раздел RESULTS.md | Источник |
|---|---|
| 1. Датасет | `data/processed/split_report.json` |
| 2. Параметры обучения | `models/type*/synthetic_openrouter/run_manifest.json` |
| 3. Сравнение до/после | `reports/eval/*/eval_summary.json` |
| 4. Cross-type confusion | `reports/eval/*/cross_type_confusion.json` |
| 5. Memorization top-5 | `reports/memorization/memorization_type*.csv` |
| 6. Упрощения | захардкожены в скрипте |

Если какой-то файл отсутствует, раздел заменяется плейсхолдером `_(данные недоступны — ...)_`, скрипт не падает — это позволяет собрать частичный отчёт даже до окончания всех прогонов.

Опциональные флаги:

```bash
python scripts/build_results.py \
  --baseline-dir  reports/eval/baseline_test_100 \
  --finetuned-dir reports/eval/finetuned_test_synth_100 \
  --memorization-dir reports/memorization_synth_100 \
  --out RESULTS.md
```

## Минимальный happy-path для сдачи (N=1600 ok train на тип)

```bash
# 0. окружение
pip install -e ".[dev]"

# 1. сырые экспорты — лежат в data/raw/type{1,2}/telegram_export.json

# 2. split (per-type max_samples из configs/data.yaml: type1=2000, type2=3300)
python scripts/prepare_dataset.py --min-chars 50

# 3. (опционально) synthetic neutral inputs через OpenRouter,
#    early-stop при target_train_ok=1600 в train; val/test — целиком
python scripts/generate_openrouter_synthetic.py \
  --types type1 type2 --splits train val test \
  --out-dir data/processed/synthetic_openrouter \
  --report data/processed/synthetic_openrouter/report.json --drop-failed

# 4. обучение на synthetic inputs из Шага 3 (--data-dir обязателен!)
python scripts/finetune.py --type type1 --env ubuntu_t4 --config configs/train.yaml --data-dir data/processed/synthetic_openrouter
python scripts/finetune.py --type type2 --env ubuntu_t4 --config configs/train.yaml --data-dir data/processed/synthetic_openrouter

# 5.1 self-check
python scripts/evaluate.py --metrics length style

# 5.2-5.3 benchmark: baseline vs synthetic_openrouter (на target_eval_samples=100 из configs/data.yaml)
mkdir -p output
python scripts/generate_baseline.py --type type1 --env ubuntu_t4 --from-jsonl data/processed/type1_test.jsonl --limit-from-config --output output/bench_type1_test_baseline_100.txt
python scripts/generate_baseline.py --type type2 --env ubuntu_t4 --from-jsonl data/processed/type2_test.jsonl --limit-from-config --output output/bench_type2_test_baseline_100.txt

python scripts/generate.py --type type1 --env ubuntu_t4 --adapter models/type1/synthetic_openrouter/adapter --from-jsonl data/processed/type1_test.jsonl --limit-from-config --output output/bench_type1_test_finetuned_synth_100.txt
python scripts/generate.py --type type2 --env ubuntu_t4 --adapter models/type2/synthetic_openrouter/adapter --from-jsonl data/processed/type2_test.jsonl --limit-from-config --output output/bench_type2_test_finetuned_synth_100.txt

python scripts/evaluate.py --split test --limit-from-config --generated-type1 output/bench_type1_test_baseline_100.txt        --generated-type2 output/bench_type2_test_baseline_100.txt        --metrics all --out-dir reports/eval/baseline_test_100
python scripts/evaluate.py --split test --limit-from-config --generated-type1 output/bench_type1_test_finetuned_synth_100.txt --generated-type2 output/bench_type2_test_finetuned_synth_100.txt --metrics all --out-dir reports/eval/finetuned_test_synth_100

# 5.5 memorization
python scripts/memorization_check.py --generated-type1 output/bench_type1_test_finetuned_synth_100.txt --generated-type2 output/bench_type2_test_finetuned_synth_100.txt --out-dir reports/memorization_synth_100

# 6. deliverables
python scripts/generate.py --type type1 --env ubuntu_t4 --input inputs_type1.txt --output outputs_type1.txt
python scripts/generate.py --type type2 --env ubuntu_t4 --input inputs_type2.txt --output outputs_type2.txt

# 7. RESULTS.md
python scripts/build_results.py
```

После этого все пункты чек-листа в разделе ниже закрыты артефактами.

## Рекомендации по корректному решению задачи

Этот раздел сверяет текущий пайплайн с требованиями `task_description.md` и фиксирует, что именно нужно поменять, чтобы решение действительно отвечало задаче, а не выглядело как "запустили скрипты в правильном порядке". Ниже — слабые места и как их закрыть.

### 1. Убрать избыточные `cp`-шаги в Шаге 4

**Выполнено.** В `scripts/finetune.py` добавлен флаг `--data-dir` (по умолчанию `data/processed/`). Шаг 4 переписан: вместо 12 `cp`-команд используется `--data-dir data/processed/synthetic_openrouter`. Базовые processed-файлы больше не перезаписываются.

Если в основных jsonl несколько вариантов `input_source`, можно использовать `--train-input-source` / `--val-input-source` без отдельной директории — этот флаг в `finetune.py` уже был.

### 2. Привести offline benchmark к описанию из `evaluate.md`

**Выполнено.** Шаг B6 переписан: добавлена таблица 4 метрик с объяснением, что значит "стало лучше" для каждой, явный критерий победы (`style_accuracy ↑` И `cosine_mean ≈` И `mauve ↑`) и команда self-check на reference-постах без генерации.

### 3. Добавить memorization-check (требование task_description.md)

**Выполнено.** Создан `scripts/memorization_check.py`. Добавлен Шаг B7 в offline benchmark.

### 4. Добавить cross-type проверку (требование "не средний по больнице")

**Выполнено.** Добавлена метрика `crosstype` в `scripts/evaluate.py` и функция `cross_type_confusion_matrix` в `src/telegram_style_transfer/eval.py`.

Что делает метрика: прогоняет оба generated-файла через один и тот же TF-IDF + LogReg классификатор и печатает confusion-матрицу:

```text
              → type1    → type2
type1 adapter   85.3%     14.7%   (n=120)
type2 adapter   12.1%     87.9%   (n=115)

Diagonal mean (correct-class): 86.6%  |  random baseline: 50.0%
```

Для finetuned модели (шаг B5):

```bash
python scripts/evaluate.py \
  --limit-from-config \
  --generated-type1 output/bench_type1_test_finetuned_synth_100.txt \
  --generated-type2 output/bench_type2_test_finetuned_synth_100.txt \
  --metrics crosstype \
  --out-dir reports/eval/finetuned_test_synth_100
```

Для baseline (шаг B4):

```bash
python scripts/evaluate.py \
  --limit-from-config \
  --generated-type1 output/bench_type1_test_baseline_100.txt \
  --generated-type2 output/bench_type2_test_baseline_100.txt \
  --metrics crosstype \
  --out-dir reports/eval/baseline_test_100
```

Результаты сохраняются в `cross_type_confusion.json`. В `RESULTS.md` показать обе матрицы рядом — это прямой ответ на требование задачи. Высокий диагональ (> 70%) у fine-tuned при низком у baseline = явная разделимость стилей.

Критерий победы: diagonal mean у finetuned заметно выше, чем у baseline (который ожидаемо близок к 50%).

### 5. Упростить Шаг B1 и объединить с Шагом 6

**Выполнено.**

- Добавлен флаг `--from-jsonl FILE` (и `--field`, default `input`) в `scripts/generate.py` и `scripts/generate_baseline.py`. Теперь inputs читаются прямо из jsonl — промежуточные `bench_*_test_inputs.txt` не нужны.
- Шаг B1 (heredoc Python) удалён. Генерация baseline/finetuned (Шаги B2/B3) теперь напрямую принимает `--from-jsonl data/processed/type*_test.jsonl`.
- Добавлены вводные "Назначение" к блокам Offline benchmark, Шаг 6 и Showcase — чтобы явно разделить: benchmark на test split vs. deliverable на внешних входах vs. qualitative showcase.

### 6. Добавить шаг сборки RESULTS.md

**Выполнено.** Создан `scripts/build_results.py`. Добавлен Шаг 7 в offline benchmark.

Скрипт собирает все 6 разделов `RESULTS.md` из артефактов пайплайна, с graceful fallback для каждого отсутствующего файла.

### 7. Чек-лист соответствия `task_description.md`

Перед сдачей пробежать по требованиям задачи и убедиться, что каждое закрыто артефактом:

- [ ] датасет собран из обоих каналов → `data/processed/type*_*.jsonl` + `data/processed/split_report.json`;
- [ ] подстройка отдельно под type1 и type2 → `models/type1/synthetic_openrouter/adapter/`, `models/type2/synthetic_openrouter/adapter/`;
- [ ] `outputs_type1.txt` и `outputs_type2.txt` сгенерированы → Шаг 6;
- [ ] сравнение "до / после" → `reports/eval/baseline_test_100/` vs `reports/eval/finetuned_test_synth_100/`;
- [ ] проверка на отложенных текстах → `data/processed/type*_test.jsonl` + `split_report.json` (по дате, не random);
- [ ] не запоминание примеров → memorization-check (раздел 3 выше);
- [ ] стабильное соответствие type1 vs type2 → cross-type confusion (раздел 4 выше);
- [ ] воспроизводимость → `models/<type>/synthetic_openrouter/run_manifest.json` с `base_model` + `hyperparams` + `train/val_samples` + `train_loss`/`eval_loss` + `peak_vram_gb` + `timestamp_utc` (revision базовой модели и dataset hashes сейчас не пишутся — при необходимости пинить через `configs/train.yaml` / `configs/env.yaml` и расширять manifest);
- [ ] `RESULTS.md` содержит шаги, время, плюсы/минусы, метод проверки → раздел 6 выше.

Каждый пункт без артефакта — формальный недовес по `task_description.md`.
