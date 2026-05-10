from __future__ import annotations

import re

_CTA_PATTERNS = [
    r"подпис(ывайтесь|ывайся|аться|аны|ка|ки)",
    r"жми(те)?\s+(сюда|на\s+ссылку|ниже|👇)",
    r"читайте\s+подробнее",
    r"подробнее\s+по\s+ссылке",
    r"переходи(те)?\s+по\s+ссылке",
    r"(узнай(те)?|смотри(те)?)\s+больше",
    r"подробности\s+(в|на|по)",
    r"ссылка\s+в\s+био",
    r"в\s+описании\s+канала",
]
_CTA_RE = re.compile("|".join(_CTA_PATTERNS), re.IGNORECASE)

INSTRUCTION_TEMPLATE = (
    "Rewrite the input into a Telegram post in style type {style_type}. "
    "Preserve the facts, but match the tone, structure, pacing, "
    "and ending typical for this style."
)

TRAINING_PROMPT = """\
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

INFERENCE_PROMPT = """\
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def _strip_cta(text: str) -> str:
    lines = text.splitlines()
    cleaned = [line for line in lines if not _CTA_RE.search(line)]
    return "\n".join(cleaned).strip()


def build_input_heuristic(text: str, max_facts: int = 5) -> str:
    """Build a neutral 'Topic + Facts' brief from a channel post.

    Strips common Russian CTA phrases and sign-offs; preserves numbers and
    named entities. Used when no external LLM is available for summarization.
    """
    text = _strip_cta(text)
    sentences = _split_sentences(text)
    if not sentences:
        return text

    topic = sentences[0]
    facts = sentences[1 : max_facts + 1]

    parts = [f"Topic: {topic}"]
    if facts:
        parts.append("Facts:")
        for fact in facts:
            parts.append(f"- {fact}")
    return "\n".join(parts)


def format_training_prompt(style_type: str, input_text: str, response: str) -> str:
    instruction = INSTRUCTION_TEMPLATE.format(style_type=style_type)
    return TRAINING_PROMPT.format(
        instruction=instruction,
        input=input_text,
        response=response,
    )


def format_inference_prompt(style_type: str, input_text: str) -> str:
    instruction = INSTRUCTION_TEMPLATE.format(style_type=style_type)
    return INFERENCE_PROMPT.format(
        instruction=instruction,
        input=input_text,
    )
