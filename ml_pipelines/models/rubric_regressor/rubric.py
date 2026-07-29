import json
from collections.abc import Mapping



AXES_DEFINITIONS = {
    "global_coherence": (
        "How clearly the writing develops a unified line of thought."
    ),
    "claim_development": (
        "How well claims are explained, supported, and qualified."
    ),
    "specificity": (
        "How concrete, precise, and context-specific the writing is."
    ),
    "voice_and_stance": (
        "How distinctive, consistent, and purposeful the writer's perspective is."
    ),
    "contextual_judgment": (
        "How well the writing recognizes tradeoffs, exceptions, and constraints."
    ),
    "semantic_novelty": (
        "How consistently each sentence adds information not already stated."
    ),
    "stylistic_variation": (
        "How naturally sentence structures, openings, and pacing vary."
    ),
    "idea_compression": (
        "How efficiently the writing communicates ideas without unnecessary expansion."
    ),
    "formality_fit": (
        "How well the level of formality matches the audience and purpose."
    ),
    "substantive_value": (
        "How much the writing contributes something meaningful or worth saying."
    ),
}


LEVEL_DEFINITIONS = {
    1: (
        "Strongly machine-like: generic, repetitive, formulaic, mechanically "
        "structured, or lacking meaningful judgment."
    ),
    2: (
        "Somewhat machine-like: noticeable repetition, predictability, weak "
        "specificity, or limited development of ideas."
    ),
    3: (
        "Ambiguous: contains a mixture of human-like and machine-like qualities "
        "without a clear overall signal."
    ),
    4: (
        "Somewhat human-like: specific, purposeful, varied, and context-sensitive, "
        "with only minor signs of formulaic construction."
    ),
    5: (
        "Strongly human-like: distinctive, economical, context-aware, naturally "
        "varied, and driven by genuine reasoning or communicative purpose."
    ),
}


TASK_PROMPT = """

Score every requested axis using an integer from 1 through 5.
Evaluate only the supplied writing. Do not infer the author's identity.
""".strip()


def make_sys_prompt(
    axes_defs: Mapping[str, str] = AXES_DEFINITIONS,
) -> str:
    output_schema = {axis: 1 for axis in axes_defs}

    return f"""
{TASK_PROMPT}

Scoring levels:
{json.dumps(LEVEL_DEFINITIONS, separators=(",", ":"))}

Axes:
{json.dumps(dict(axes_defs), separators=(",", ":"))}

Return one valid JSON object matching this structure:
{json.dumps(output_schema, separators=(",", ":"))}

Requirements:
- Include every requested axis exactly once.
- Use only integer values from 1 through 5.
- Do not include explanations, markdown, preamble, or additional keys.


""".strip()
