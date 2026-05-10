from telegram_style_transfer.dataset_inputs import load_brief_overrides_from_records


def test_load_brief_overrides_accepts_only_successful_low_jaccard_records():
    overrides, stats = load_brief_overrides_from_records(
        [
            {
                "post_id": 1,
                "style_type": "type1",
                "neutral": "Topic: Тема\nFacts:\n- Факт 1",
                "failed": False,
                "jaccard": 0.21,
                "prompt_mode": "brief_v4",
            },
            {
                "post_id": 2,
                "style_type": "type1",
                "neutral": "Topic: Слишком близко\nFacts:\n- Факт 2",
                "failed": False,
                "jaccard": 0.31,
                "prompt_mode": "brief_v4",
            },
            {
                "post_id": 3,
                "style_type": "type2",
                "neutral": "Topic: Ошибка\nFacts:\n- Факт 3",
                "failed": True,
                "jaccard": 0.12,
                "prompt_mode": "brief_v4",
            },
        ],
        max_jaccard=0.3,
    )

    assert overrides == {
        ("type1", 1): {
            "input": "Topic: Тема\nFacts:\n- Факт 1",
            "input_source": "llm_brief_v4",
        }
    }
    assert stats == {
        "loaded": 3,
        "accepted": 1,
        "rejected_failed": 1,
        "rejected_missing_input": 0,
        "rejected_jaccard": 1,
    }


def test_load_brief_overrides_supports_synthetic_ok_records():
    overrides, stats = load_brief_overrides_from_records(
        [
            {
                "post_id": 10,
                "style_type": "type2",
                "synthetic_input": "Topic: Данные\nFacts:\n- Пункт",
                "synthetic_ok": True,
                "jaccard": 0.18,
            }
        ],
        max_jaccard=0.3,
    )

    assert overrides[("type2", 10)]["input"] == "Topic: Данные\nFacts:\n- Пункт"
    assert overrides[("type2", 10)]["input_source"] == "llm_synthetic"
    assert stats["accepted"] == 1
