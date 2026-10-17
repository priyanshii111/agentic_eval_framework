"""The `quality` eval_type: a no-reference 1-5 Likert judgement of an answer,
using PyRIT's SelfAskLikertScorer with a local Ollama judge.

No reference answer is needed: the judge sees the customer's question and the
bot's answer and rates one criterion (e.g. relevance). The 1-5 value is
normalized to 0-1 by PyRIT and compared to a threshold like every other
statistical score — it stays a probability, not a certainty (Rule 7).

PyRIT's bundled Likert instructions are written for HARM scoring (higher =
more severe, "not harmful -> rate lower"), which scores a perfect answer 1/5.
So this checker ships its own quality template where higher = better.
"""
from pathlib import Path

from ...utils import pyrit_runtime

_SCALES_DIR = Path(__file__).parent / "likert_scales"

# criterion -> Likert scale YAML (PyRIT format). A new criterion = one new YAML file + one entry.
CRITERIA = {
    "relevance": _SCALES_DIR / "relevance.yaml",
}

_QUALITY_TEMPLATE = """# Instructions
You are grading the QUALITY of a customer support bot's answer on a Likert scale for {{category}}.
Higher is better: {{min_scale_value}} is the worst quality and {{max_scale_value}} is the best quality.
The message contains the customer's question followed by the bot's answer. Judge only the answer, against the question.

# Likert Scale
{{ likert_scale }}

# Response format
Respond with ONLY a JSON object with the keys "score_value", "description", "rationale":
- "score_value" (str): one of the levels above, between {{min_scale_value}} and {{max_scale_value}} inclusive.
- "description" (str): the description of that level.
- "rationale" (str): a short step-by-step explanation, ending with the score."""

async def _likert_score_async(text: str, scale_path: Path, config: dict) -> dict:
    """Score `text` once with PyRIT's Likert scorer. Raises on any failure."""
    # Imported here, not at the top: static mode and the other statistical
    # checkers must work without PyRIT installed.
    from pyrit.score import SelfAskLikertScorer
    from pyrit.score.float_scale.likert_scale import LikertScale

    await pyrit_runtime.ensure_initialized_async()

    judge = pyrit_runtime.ollama_judge_target(config)
    scale = LikertScale.from_yaml(scale_path)
    scorer = SelfAskLikertScorer.from_likert_scale(
        chat_target=judge, likert_scale=scale, system_prompt_template=_QUALITY_TEMPLATE
    )
    score = (await scorer.score_text_async(text))[0]
    return {
        "score": float(score.score_value),
        "likert_value": score.score_metadata["likert_value"],
        "likert_max": scale.maximum_value,
        "judge_reason": score.score_rationale,
    }


def evaluate(actual_output: dict, expected: dict, config: dict | None = None) -> dict:
    config = config or {}
    criterion = expected.get("criterion", "relevance")
    scale_path = CRITERIA.get(criterion)
    if scale_path is None:
        return {
            "status": "error",
            "details": {"reason": f"unknown quality criterion '{criterion}' (known: {sorted(CRITERIA)})"},
        }

    question = expected["question"]
    threshold = expected.get("threshold", config.get("default_quality_threshold", 0.5))
    actual_text = actual_output.get("output_text") or ""
    text = f"Customer question: {question}\n\nSupport bot answer: {actual_text}"

    try:
        result = pyrit_runtime.run(_likert_score_async(text, scale_path, config))
    except ImportError as e:
        return {
            "status": "error",
            "details": {"reason": f"PyRIT is not installed ({e}); install with: pip install -e \".[pyrit]\""},
        }
    except Exception as e:
        return {"status": "error", "details": {"reason": f"quality scoring failed: {e!r}"}}

    score = result["score"]
    return {
        "status": "pass" if score >= threshold else "fail",
        "details": {
            "criterion": criterion,
            "likert_value": result["likert_value"],
            "likert_max": result["likert_max"],
            "score": round(score, 4),
            "threshold": threshold,
            "judge_reason": result["judge_reason"],
            "judge_model": config["llm_judge_model"],
        },
    }
