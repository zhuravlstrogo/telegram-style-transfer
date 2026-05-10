# telegram-style-transfer

Fine-tune LLM под стиль постов telegram-каналов.

## Документация

- Описание задачи: [`docs/task_description.md`](docs/task_description.md)
- Результаты: [`RESULTS.md`](RESULTS.md)
- Этапы пайплайна: [`docs/pipeline.md`](docs/pipeline.md)
- Инференс fine-tuned модели: [`docs/inference.md`](docs/inference.md)

## Структура

```text
.
├── configs/             # yaml-конфиги пайплайна/окружения/обучения
├── scripts/            
├── src/               
├── tests/
├── data/               # сырые и обработанные данные (скачать по ссылке)
├── models/             # обученные LoRA-адаптеры (скачать по ссылке)
├── output/             # результаты генерации (скачать по ссылке)
├── reports/            # отчёты и метрики (скачать по ссылке)
├── docs/               # research
├── logs/               # логи запусков
├── task_description.md
├── pipeline.md
├── inference.md
├── RESULTS.md
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Скачивание данных и артефактов

- `data/` - [data](https://drive.google.com/file/d/18a2-BA4LNxqwGrFoztAs5B4Cz0VsmGp_)
- `models/` - [models](https://drive.google.com/file/d/1rWWAXY4Xp8zDUUuWG5_GnODvcR31hK3N)
- `output/` - [output](https://drive.google.com/file/d/1J5HHmH-rrp_Aq_sAJ1B2A5yJlpPJawGK)
- `reports/` - [reports](https://drive.google.com/file/d/1sdU0BnYJDq-uN8OsILWLp0GGY7xmzW9b)
