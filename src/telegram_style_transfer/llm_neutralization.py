from __future__ import annotations

import json
import math
import re
from typing import Any

from telegram_style_transfer.data import clean_for_input
from telegram_style_transfer.synthetic import score_neutralization

DEFAULT_PRIMARY_MODEL = "openai/gpt-4.1-mini"
DEFAULT_FALLBACK_MODEL = "openai/gpt-4.1"

OPENROUTER_PRICING_USD_PER_1M = {
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "openai/gpt-4.1": {"input": 2.00, "output": 8.00},
}

DEFAULT_PROMPT_OVERHEAD_TOKENS = 150
DEFAULT_CHARS_PER_TOKEN = 1.7
DEFAULT_COMPLETION_RATIO = 0.9
DEFAULT_MIN_COMPLETION_TOKENS = 32
DEFAULT_MAX_JACCARD = 0.5
PROMPT_MODES: dict[str, dict[str, Any]] = {
    "rewrite": {
        "system": """Ты редактор нейтральных новостных сводок на русском языке.
Переписывай Telegram-пост в сухой информационный стиль.

Обязательные правила:
1. Сохрани все факты без добавления новых фактов или выводов.
2. Обязательно сохрани числа, даты, время, проценты, суммы, имена собственные, названия компаний, мест и продуктов.
3. Удали эмодзи, хэштеги, @-хэндлы, ссылки, призывы подписаться, обращения к читателю, рекламные слоганы и фирменные обороты канала.
4. Убери оценочность и разговорность; стиль должен быть нейтральным.
5. Сохрани порядок фактов и, по возможности, абзацную структуру.
6. Если текст уже почти нейтральный, всё равно слегка перефразируй его, чтобы избежать verbatim copy.
7. Если после удаления шумовых маркеров в тексте не остаётся фактического содержания, верни пустую строку.
8. Ничего не объясняй и не комментируй. Верни только JSON по схеме.""",
        "user_template": "Нейтрализуй текст.\n<source_text>\n{text}\n</source_text>",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "neutralization_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "neutral_text": {
                            "type": "string",
                            "description": (
                                "Нейтральная русскоязычная версия исходного текста "
                                "без стилевых маркеров канала"
                            ),
                        }
                    },
                    "required": ["neutral_text"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "brief": {
        "system": """Ты редактор, который готовит сухой фактический бриф для другого автора.
Твоя задача — не переписывать пост близко к оригиналу, а извлекать тему и атомарные факты.

Обязательные правила:
1. Сохрани все факты без добавления новых фактов, выводов или оценок.
2. Обязательно сохрани числа, даты, время, проценты, суммы, имена собственные, компании, места и продукты.
3. Удали эмодзи, хэштеги, @-хэндлы, ссылки, призывы подписаться, обращения к читателю, рекламу и фирменные обороты канала.
4. Не копируй длинные фразы или целые предложения из исходника, если это не необходимо для точности фактов.
5. Верни короткую тему и список коротких фактологических пунктов.
6. Каждый пункт должен содержать один факт или одну тесно связанную группу фактов.
7. Если фактического содержания не остаётся, верни пустую тему и пустой список фактов.
8. Ничего не объясняй и не комментируй. Верни только JSON по схеме.""",
        "user_template": (
            "Преобразуй пост в краткий нейтральный бриф.\n"
            "Верни тему и не более {fact_limit} фактологических пунктов.\n"
            "<source_text>\n{text}\n</source_text>"
        ),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "brief_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "facts": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["topic", "facts"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "brief_v2": {
        "system": """Ты готовишь сверхкраткий редакторский бриф по сообщению из Telegram-канала.
Твоя задача — извлечь тему и факты в форме коротких заметок, а не пересказывать исходный текст.

Обязательные правила:
1. Сохрани все ключевые факты без добавления новых фактов, выводов и оценок.
2. Обязательно сохрани числа, даты, время, проценты, суммы, имена собственные, географию, компании, продукты и причинно-следственные связи.
3. Полностью убери эмоции, рекламные формулы, обращения к читателю, эмодзи, хэштеги, ссылки, @-хэндлы и фирменные обороты канала.
4. Не копируй длинные фразы из исходника. Избегай последовательностей длиннее 4-5 слов подряд из источника, если это не имя собственное или точное числовое обозначение.
5. Пиши в формате editor notes: короткая тема и короткие пункты фактов. Предпочитай телеграфный стиль, а не полные пересказные предложения.
6. Тема должна быть короткой: 5-12 слов.
7. Каждый факт должен быть коротким: одна мысль, одна строка.
8. Верни не более {fact_limit} пунктов. Если фактов больше, оставь самые важные.
9. Если фактического содержания не остаётся, верни пустую тему и пустой список фактов.
10. Ничего не объясняй и не комментируй. Верни только JSON по схеме.""",
        "user_template": (
            "Сделай компактный factual brief в формате editor notes.\n"
            "Тема: короткая.\n"
            "Факты: не более {fact_limit} коротких пунктов.\n"
            "<source_text>\n{text}\n</source_text>"
        ),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "brief_result_v2",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "facts": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["topic", "facts"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "brief_v3": {
        "system": """Ты готовишь структурированный factual brief для другой модели, которая потом будет писать текст в нужном стиле.
Твоя задача — извлечь содержание из Telegram-поста в максимально сжатую, структурированную и не похожую на исходник форму.

Обязательные правила:
1. Не пересказывай пост близко к оригиналу. Извлекай структуру содержания.
2. Сохрани все ключевые факты без добавления новых фактов, выводов или оценок.
3. Обязательно сохрани имена собственные, компании, географию, продукты, даты, время, проценты, суммы и другие числа.
4. Удали весь канал-специфичный шум: эмодзи, ссылки, @-хэндлы, CTA, фирменные обороты, обращения к читателю.
5. Тема должна быть короткой: 4-10 слов.
6. `entities` — только ключевые сущности, не более {entity_limit}.
7. `numbers` — только важные числа и даты в краткой форме, не более {number_limit}.
8. `facts` — короткие factual notes, не более {fact_limit} пунктов, по одной мысли на пункт.
9. Избегай длинных фраз из исходника. Если можно сократить формулировку без потери факта, делай это.
10. Ничего не объясняй и не комментируй. Верни только JSON по схеме.""",
        "user_template": (
            "Преобразуй пост в structured factual brief.\n"
            "Нужны поля: topic, entities, numbers, facts.\n"
            "Ограничения: entities <= {entity_limit}, numbers <= {number_limit}, facts <= {fact_limit}.\n"
            "<source_text>\n{text}\n</source_text>"
        ),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "brief_result_v3",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "entities": {"type": "array", "items": {"type": "string"}},
                        "numbers": {"type": "array", "items": {"type": "string"}},
                        "facts": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["topic", "entities", "numbers", "facts"],
                    "additionalProperties": False,
                },
            },
        },
    },
    "brief_v4": {
        "system": """Ты готовишь компактный factual brief для downstream-модели.
Финальный brief должен быть коротким, фактологичным и заметно отличаться по лексике от исходного Telegram-поста.

Обязательные правила:
1. Сохрани только ключевые факты.
2. Сохрани важные числа, даты и имена собственные.
3. Не копируй длинные фразы из исходника.
4. Верни только короткую тему и короткие factual bullets.
5. Ничего не объясняй и не комментируй. Верни только JSON по схеме.""",
        "user_template": (
            "Сделай компактный factual brief для другой модели.\n"
            "Тема: короткая.\n"
            "Факты: не более {fact_limit} коротких пунктов.\n"
            "<source_text>\n{text}\n</source_text>"
        ),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "brief_result_v4",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "facts": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["topic", "facts"],
                    "additionalProperties": False,
                },
            },
        },
    },
}

SYSTEM_PROMPT = PROMPT_MODES["rewrite"]["system"]
NEUTRALIZATION_RESPONSE_FORMAT = PROMPT_MODES["rewrite"]["response_format"]


def prompt_mode_names() -> list[str]:
    return list(PROMPT_MODES.keys())


def get_prompt_mode(name: str) -> dict[str, Any]:
    try:
        return PROMPT_MODES[name]
    except KeyError as exc:
        available = ", ".join(prompt_mode_names())
        raise ValueError(
            f"Unknown prompt mode {name!r}. Available: {available}"
        ) from exc


def build_brief_text(topic: str, facts: list[str]) -> str:
    topic = topic.strip()
    facts = [fact.strip() for fact in facts if fact.strip()]
    parts: list[str] = []
    if topic:
        parts.append(f"Topic: {topic}")
    if facts:
        parts.append("Facts:")
        parts.extend(f"- {fact}" for fact in facts)
    return "\n".join(parts).strip()


def build_brief_v3_text(
    topic: str,
    entities: list[str],
    numbers: list[str],
    facts: list[str],
) -> str:
    topic = topic.strip()
    entities = [item.strip() for item in entities if item.strip()]
    numbers = [item.strip() for item in numbers if item.strip()]
    facts = [item.strip() for item in facts if item.strip()]

    parts: list[str] = []
    if topic:
        parts.append(f"Topic: {topic}")
    if entities:
        parts.append("Entities:")
        parts.extend(f"- {item}" for item in entities)
    if numbers:
        parts.append("Numbers:")
        parts.extend(f"- {item}" for item in numbers)
    if facts:
        parts.append("Facts:")
        parts.extend(f"- {item}" for item in facts)
    return "\n".join(parts).strip()


def build_user_prompt(
    text: str,
    prompt_mode: str = "rewrite",
    fact_limit: int = 5,
) -> str:
    mode = get_prompt_mode(prompt_mode)
    return str(mode["user_template"]).format(
        text=text,
        fact_limit=fact_limit,
        entity_limit=fact_limit,
        number_limit=fact_limit,
    )


def response_format_for(prompt_mode: str = "rewrite") -> dict[str, Any]:
    return dict(get_prompt_mode(prompt_mode)["response_format"])


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def longest_common_ngram(a: str, b: str, max_n: int = 8) -> int:
    tokens_a = word_tokens(a)
    tokens_b = word_tokens(b)
    if not tokens_a or not tokens_b:
        return 0

    upper = min(max_n, len(tokens_a), len(tokens_b))
    for n in range(upper, 0, -1):
        grams_a = {tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)}
        if not grams_a:
            continue
        for i in range(len(tokens_b) - n + 1):
            if tuple(tokens_b[i : i + n]) in grams_a:
                return n
    return 0


def copied_ngram_ratio(source: str, candidate: str, n: int = 4) -> float:
    source_tokens = word_tokens(source)
    candidate_tokens = word_tokens(candidate)
    if len(candidate_tokens) < n or len(source_tokens) < n:
        return 0.0

    source_ngrams = {
        tuple(source_tokens[i : i + n]) for i in range(len(source_tokens) - n + 1)
    }
    candidate_ngrams = [
        tuple(candidate_tokens[i : i + n]) for i in range(len(candidate_tokens) - n + 1)
    ]
    if not candidate_ngrams:
        return 0.0

    copied = sum(1 for gram in candidate_ngrams if gram in source_ngrams)
    return copied / len(candidate_ngrams)


def strip_json_fences(text: str) -> str:
    fenced = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*", "", fenced, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    return fenced.strip()


def parse_neutralization_payload(
    content: str,
    prompt_mode: str = "rewrite",
    allow_plaintext_fallback: bool = False,
) -> str:
    payload_text = strip_json_fences(content)
    if not payload_text:
        return ""

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        if allow_plaintext_fallback:
            return payload_text
        raise ValueError("Invalid JSON response from neutralizer") from exc

    if prompt_mode == "rewrite":
        neutral_text = payload.get("neutral_text", "")
        if neutral_text is None:
            return ""
        return str(neutral_text).strip()
    if prompt_mode in {"brief", "brief_v2", "brief_v4"}:
        topic = str(payload.get("topic", "") or "")
        facts = payload.get("facts", [])
        if not isinstance(facts, list):
            raise ValueError("Invalid facts payload from neutralizer")
        return build_brief_text(topic, [str(item) for item in facts])
    if prompt_mode == "brief_v3":
        topic = str(payload.get("topic", "") or "")
        entities = payload.get("entities", [])
        numbers = payload.get("numbers", [])
        facts = payload.get("facts", [])
        if not isinstance(entities, list):
            raise ValueError("Invalid entities payload from neutralizer")
        if not isinstance(numbers, list):
            raise ValueError("Invalid numbers payload from neutralizer")
        if not isinstance(facts, list):
            raise ValueError("Invalid facts payload from neutralizer")
        return build_brief_v3_text(
            topic,
            [str(item) for item in entities],
            [str(item) for item in numbers],
            [str(item) for item in facts],
        )
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def estimate_usage_from_text(
    text: str,
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    completion_ratio: float = DEFAULT_COMPLETION_RATIO,
    min_completion_tokens: int = DEFAULT_MIN_COMPLETION_TOKENS,
) -> dict[str, int]:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")

    clean_text = text.strip()
    source_tokens = math.ceil(len(clean_text) / chars_per_token)
    prompt_tokens = prompt_overhead_tokens + source_tokens
    completion_tokens = max(
        min_completion_tokens,
        math.ceil((len(clean_text) * completion_ratio) / chars_per_token),
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float | None:
    pricebook = pricing or OPENROUTER_PRICING_USD_PER_1M
    model_pricing = pricebook.get(model)
    if model_pricing is None:
        return None

    prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
    completion_cost = (completion_tokens / 1_000_000) * model_pricing["output"]
    return round(prompt_cost + completion_cost, 6)


def validate_neutralization(
    response_clean: str,
    candidate: str,
    max_jaccard: float = DEFAULT_MAX_JACCARD,
    max_shared_ngram: int | None = None,
    max_copy_ngram_ratio: float | None = None,
    copy_ngram_size: int = 4,
) -> dict[str, Any]:
    neutralized = clean_for_input(candidate)
    if not neutralized:
        failed = True
        failure_reason = "empty_after_clean"
    else:
        failed = False
        failure_reason = None

    metrics = score_neutralization(
        response_clean=response_clean,
        neutralized=neutralized,
        failed=failed,
    )
    shared_ngram = longest_common_ngram(response_clean, neutralized)
    copy_ratio = copied_ngram_ratio(
        response_clean,
        neutralized,
        n=copy_ngram_size,
    )

    if failed:
        return {
            "neutralized_clean": "",
            "failed": True,
            "failure_reason": failure_reason,
            "longest_common_ngram": shared_ngram,
            "copy_ngram_ratio": round(copy_ratio, 4),
            **metrics,
        }

    if not metrics["numbers_preserved"]:
        return {
            "neutralized_clean": "",
            "failed": True,
            "failure_reason": "numbers_not_preserved",
            **score_neutralization(response_clean, neutralized, failed=True),
        }
    if metrics["is_identity"]:
        return {
            "neutralized_clean": "",
            "failed": True,
            "failure_reason": "identity",
            **score_neutralization(response_clean, neutralized, failed=True),
        }
    if (metrics["jaccard"] or 1.0) >= max_jaccard:
        return {
            "neutralized_clean": "",
            "failed": True,
            "failure_reason": "high_overlap",
            "longest_common_ngram": shared_ngram,
            "copy_ngram_ratio": round(copy_ratio, 4),
            **score_neutralization(response_clean, neutralized, failed=True),
        }
    if max_shared_ngram is not None and shared_ngram >= max_shared_ngram:
        return {
            "neutralized_clean": "",
            "failed": True,
            "failure_reason": "copied_span",
            "longest_common_ngram": shared_ngram,
            "copy_ngram_ratio": round(copy_ratio, 4),
            **score_neutralization(response_clean, neutralized, failed=True),
        }
    if max_copy_ngram_ratio is not None and copy_ratio >= max_copy_ngram_ratio:
        return {
            "neutralized_clean": "",
            "failed": True,
            "failure_reason": f"copied_{copy_ngram_size}gram_ratio",
            "longest_common_ngram": shared_ngram,
            "copy_ngram_ratio": round(copy_ratio, 4),
            **score_neutralization(response_clean, neutralized, failed=True),
        }

    return {
        "neutralized_clean": neutralized,
        "failed": False,
        "failure_reason": None,
        "longest_common_ngram": shared_ngram,
        "copy_ngram_ratio": round(copy_ratio, 4),
        **metrics,
    }


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None) or {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        completion_tokens = usage.get("completion_tokens", completion_tokens)
        total_tokens = usage.get("total_tokens", total_tokens)

    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))
    return {
        "usage_prompt_tokens": prompt_tokens,
        "usage_completion_tokens": completion_tokens,
        "usage_total_tokens": total_tokens,
    }


def summarize_usage_and_cost(
    records: list[dict[str, Any]],
    model: str | None = None,
    estimate_missing: bool = False,
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    completion_ratio: float = DEFAULT_COMPLETION_RATIO,
    min_completion_tokens: int = DEFAULT_MIN_COMPLETION_TOKENS,
) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    actual_records = 0
    estimated_records = 0
    total_cost = 0.0
    cost_records = 0

    for record in records:
        record_prompt = int(record.get("usage_prompt_tokens") or 0)
        record_completion = int(record.get("usage_completion_tokens") or 0)
        record_cost = record.get("cost_usd_estimate")
        if record_prompt or record_completion:
            prompt_tokens += record_prompt
            completion_tokens += record_completion
            actual_records += 1
            if record_cost is None and model:
                record_cost = estimate_cost_usd(record_prompt, record_completion, model)
        elif estimate_missing and model:
            source_text = record.get("response_clean") or record.get("text") or ""
            estimated = estimate_usage_from_text(
                source_text,
                prompt_overhead_tokens=prompt_overhead_tokens,
                chars_per_token=chars_per_token,
                completion_ratio=completion_ratio,
                min_completion_tokens=min_completion_tokens,
            )
            prompt_tokens += estimated["prompt_tokens"]
            completion_tokens += estimated["completion_tokens"]
            estimated_records += 1
            record_cost = estimate_cost_usd(
                estimated["prompt_tokens"],
                estimated["completion_tokens"],
                model,
            )

        if record_cost is not None:
            total_cost += float(record_cost)
            cost_records += 1

    summary: dict[str, Any] = {
        "usage_prompt_tokens": prompt_tokens,
        "usage_completion_tokens": completion_tokens,
        "usage_total_tokens": prompt_tokens + completion_tokens,
        "usage_actual_records": actual_records,
        "usage_estimated_records": estimated_records,
    }
    if cost_records:
        summary["cost_usd_estimate"] = round(total_cost, 6)
    return summary
