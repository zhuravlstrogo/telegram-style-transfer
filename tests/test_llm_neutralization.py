from telegram_style_transfer.llm_neutralization import (
    build_user_prompt,
    estimate_cost_usd,
    estimate_usage_from_text,
    parse_neutralization_payload,
    prompt_mode_names,
    summarize_usage_and_cost,
    validate_neutralization,
)


def test_parse_neutralization_payload_reads_json():
    content = '{"neutral_text":"Нейтральный текст 123"}'
    assert parse_neutralization_payload(content) == "Нейтральный текст 123"


def test_parse_neutralization_payload_handles_fenced_json():
    content = '```json\n{"neutral_text":"Текст"}\n```'
    assert parse_neutralization_payload(content) == "Текст"


def test_parse_neutralization_payload_builds_brief_text():
    content = '{"topic":"Тема","facts":["Факт 1","Факт 2"]}'
    assert (
        parse_neutralization_payload(content, prompt_mode="brief")
        == "Topic: Тема\nFacts:\n- Факт 1\n- Факт 2"
    )


def test_brief_v2_prompt_mode_is_available():
    assert "brief_v2" in prompt_mode_names()
    prompt = build_user_prompt("Тест", prompt_mode="brief_v2", fact_limit=4)
    assert "не более 4" in prompt


def test_brief_v3_prompt_mode_is_available():
    assert "brief_v3" in prompt_mode_names()
    prompt = build_user_prompt("Тест", prompt_mode="brief_v3", fact_limit=4)
    assert "entities <= 4" in prompt


def test_brief_v4_prompt_mode_is_available():
    assert "brief_v4" in prompt_mode_names()
    prompt = build_user_prompt("Тест", prompt_mode="brief_v4", fact_limit=4)
    assert "не более 4 коротких пунктов" in prompt


def test_parse_neutralization_payload_builds_brief_v2_text():
    content = '{"topic":"Тема","facts":["Факт 1","Факт 2"]}'
    assert (
        parse_neutralization_payload(content, prompt_mode="brief_v2")
        == "Topic: Тема\nFacts:\n- Факт 1\n- Факт 2"
    )


def test_parse_neutralization_payload_builds_brief_v3_text():
    content = (
        '{"topic":"Тема","entities":["Сущность 1"],'
        '"numbers":["2026","50%"],"facts":["Факт 1","Факт 2"]}'
    )
    assert (
        parse_neutralization_payload(content, prompt_mode="brief_v3")
        == "Topic: Тема\nEntities:\n- Сущность 1\nNumbers:\n- 2026\n- 50%\nFacts:\n- Факт 1\n- Факт 2"
    )


def test_parse_neutralization_payload_builds_brief_v4_text():
    content = '{"topic":"Тема","facts":["Факт 1","Факт 2"]}'
    assert (
        parse_neutralization_payload(content, prompt_mode="brief_v4")
        == "Topic: Тема\nFacts:\n- Факт 1\n- Факт 2"
    )


def test_estimate_usage_and_cost_are_positive():
    usage = estimate_usage_from_text("Короткий пост с числом 123")
    cost = estimate_cost_usd(
        usage["prompt_tokens"],
        usage["completion_tokens"],
        "openai/gpt-4.1-mini",
    )

    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert cost is not None and cost > 0


def test_summarize_usage_and_cost_can_estimate_missing_usage():
    records = [{"response_clean": "Текст 1"}, {"response_clean": "Текст 2"}]
    summary = summarize_usage_and_cost(
        records,
        model="openai/gpt-4.1-mini",
        estimate_missing=True,
    )

    assert summary["usage_estimated_records"] == 2
    assert summary["usage_total_tokens"] > 0
    assert summary["cost_usd_estimate"] > 0


def test_validate_neutralization_rejects_identity_and_high_overlap():
    identity = validate_neutralization("Текст 123", "Текст 123")
    overlap = validate_neutralization(
        "Это очень длинный текст 123 с фактами",
        "Это очень длинный текст 123 с фактами и совсем немного правок",
    )

    assert identity["failed"] is True
    assert identity["failure_reason"] == "identity"
    assert overlap["failed"] is True


def test_validate_neutralization_rejects_copied_spans():
    copied = validate_neutralization(
        "один два три четыре пять шесть семь",
        "ноль один два три четыре пять шесть восемь",
        max_jaccard=0.99,
        max_shared_ngram=5,
    )

    assert copied["failed"] is True
    assert copied["failure_reason"] == "copied_span"
