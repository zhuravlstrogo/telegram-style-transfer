# Inference

## Цель

Как развернуть окружение, взять уже обученные LoRA-адаптеры и сгенерировать итоговые `outputs_type1.txt` / `outputs_type2.txt` для входных `inputs_type1.txt` / `inputs_type2.txt`.

Если нужен полный прогон от сырых Telegram сообщение, дообучение до `RESULTS.md`, см. `pipeline.md`.

## Что нужно для запуска

Минимальный набор артефактов:

- код этого репозитория;
- папка `models/` с обученными адаптерами;
- входные файлы `inputs_type1.txt` и `inputs_type2.txt`;
- `conda`-окружение с Python 3.11 и зависимостями;
- базовая модель из `run_manifest.json` или доступ в интернет, чтобы скачать её с Hugging Face при первом запуске.

Важно: **одной папки `models/` недостаточно**. В `scripts/generate.py` LoRA-адаптер загружается из `models/.../adapter`, но базовая модель загружается отдельно через `transformers`. Если базовая модель уже закэширована локально, интернет не нужен. Если нет, первый запуск должен её скачать.

### Сколько нужно свободного диска локально

Для локального инференса закладывай:

- **минимум 12 GB свободного диска** на одну модель/один стиль;
- **лучше 15 GB+**, чтобы спокойно поместились кэш `conda`/`pip`, PyTorch, базовая модель и выходные файлы;
- **20 GB+**, если хочешь хранить оба набора адаптеров, чекпоинты `checkpoint-*` и не чистить кэши после установки.

Откуда берётся этот объём:

- репозиторий + `models/` занимают несколько гигабайт;
- `conda`-окружение с `pytorch`, `transformers`, `peft` и зависимостями обычно занимает ещё несколько гигабайт;
- базовая модель из `run_manifest.json` скачивается отдельно в кэш Hugging Face и тоже требует несколько гигабайт.

Для текущего `type1` в этом репозитории базовая модель — `Qwen/Qwen2.5-3B-Instruct`, поэтому **10 GB свободного диска — рискованно и, как правило, недостаточно**.

## Основные скрипты

| Скрипт | Назначение |
|---|---|
| `scripts/generate.py` | генерация дообученной модели через LoRA-адаптер |
| `scripts/generate_baseline.py` | генерация базовой модели без дообучения |
| `scripts/evaluate.py` | опциональная проверка готовых `outputs_type*.txt` |

---

## Шаг 0. Развернуть окружение

Ниже оставлены два сценария: `CPU-only` и `GPU`. Во всех командах используется `conda`-окружение с зафиксированным Python 3.11.

### 0.1. Сценарий `CPU-only`

```bash
conda create -n telegram-style-transfer python=3.11 -y
conda activate telegram-style-transfer
conda install -y -c pytorch pytorch==2.5.1 cpuonly
pip install -r requirements.txt
pip install -e .
```

Запускать потом с `--env cpu`.

Ожидания по CPU-only режиму:

- работает для генерации нескольких или десятков текстов;
- может быть медленным на больших батчах;
- требует достаточный объём RAM под базовую модель;
- для обучения не подходит.

### 0.2. Сценарий `GPU`

```bash
conda create -n telegram-style-transfer python=3.11 -y
conda activate telegram-style-transfer
conda install -y -c pytorch -c nvidia pytorch==2.5.1 pytorch-cuda=12.1
pip install -r requirements.txt
pip install -e .
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

Запускать потом с `--env ubuntu_t4`.

Для Ubuntu при установке `unsloth` может понадобиться системный toolchain:

```bash
sudo apt update
sudo apt install -y build-essential libc6-dev linux-libc-dev python3-dev
```

## Шаг 1. Проверить, что модели лежат в ожидаемых путях

Для запуска нужны адаптеры примерно такого вида:

```text
models/
  type1/
    synthetic_openrouter/
      adapter/
      run_manifest.json
  type2/
    synthetic_openrouter/
      adapter/
      run_manifest.json
```

`run_manifest.json` важен по двум причинам:

- в нём зафиксировано имя `base_model`;
- по нему `generate.py` понимает, какую базовую модель надо подгрузить вместе с адаптером.

Если адаптеры лежат не в этих каталогах, просто передай путь явно через `--adapter`.

## Шаг 2. Положить входные файлы

В корне проекта должны лежать:

- `inputs_type1.txt`
- `inputs_type2.txt`

Формат входа:

- один входной текст на одну строку;
- пустые строки игнорируются;
- порядок строк сохраняется;
- это должен быть нейтральный исходный текст, который нужно переписать в стиль канала.

Если один и тот же набор фактов нужно прогнать через оба стиля, содержимое файлов может совпадать.

### 3. Инференс fine-tuned модели

GPU:

```bash
python scripts/generate.py --type type1 --env ubuntu_t4 \
  --adapter models/type1/synthetic_openrouter/adapter \
  --input inputs_type1.txt --output outputs_type1.txt

python scripts/generate.py --type type2 --env ubuntu_t4 \
  --adapter models/type2/synthetic_openrouter/adapter \
  --input inputs_type2.txt --output outputs_type2.txt
```

CPU-only:

```bash
python scripts/generate.py --type type1 --env cpu \
  --adapter models/type1/synthetic_openrouter/adapter \
  --input inputs_type1.txt --output outputs_type1.txt

python scripts/generate.py --type type2 --env cpu \
  --adapter models/type2/synthetic_openrouter/adapter \
  --input inputs_type2.txt --output outputs_type2.txt
```

Что получится:

- `outputs_type1.txt`
- `outputs_type2.txt`
- рядом с каждым файлом `*.manifest.json` с параметрами запуска.

Формат выходного файла:

- один сгенерированный ответ на один входной текст;
- порядок соответствует входному файлу;
- ответы разделяются `\n===\n`, потому что один ответ может быть многострочным.

## Шаг 4. Опционально сгенерировать результаты базовой модели для сравнения

Это полезно, если хочется показать, как базовая модель пишет без адаптера.

```bash
python scripts/generate_baseline.py --type type1 --env cpu \
  --input inputs_type1.txt --output outputs_baseline_type1.txt

python scripts/generate_baseline.py --type type2 --env cpu \
  --input inputs_type2.txt --output outputs_baseline_type2.txt
```

На GPU вместо `cpu` подставь `ubuntu_t4`.

## Шаг 5. Опционально проверить готовые результаты

Если нужны метрики на внешних `outputs_type*.txt`:

```bash
python scripts/evaluate.py \
  --generated-type1 outputs_type1.txt \
  --generated-type2 outputs_type2.txt \
  --metrics all \
  --out-dir results/eval/
```

Это уже не шаг генерации, а шаг проверки.
