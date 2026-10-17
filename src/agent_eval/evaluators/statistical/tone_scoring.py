from . import scoring

_TONE_PROMPT_TEMPLATE = """You are evaluating the TONE of a customer support response, not its \
factual correctness.

Situation: {situation}
Bot's response: "{actual}"

Rate how empathetic, calm, and professional this response is, on a scale from 0.0 (cold, \
dismissive, or inappropriate) to 1.0 (warm, empathetic, and professional). Respond with ONLY \
a JSON object in this exact format, nothing else:
{{"score": <number between 0.0 and 1.0>, "reason": "<one short sentence>"}}"""


def evaluate(actual_output: dict, expected: dict, config: dict | None = None) -> dict:
    config = config or {}
    situation = expected["situation"]
    threshold = expected.get("threshold", config.get("default_tone_threshold", 0.6))
    actual_text = actual_output.get("output_text") or ""

    prompt = _TONE_PROMPT_TEMPLATE.format(situation=situation, actual=actual_text)

    try:
        result = scoring.call_judge_with_retries(prompt, config)
    except Exception as e:
        return {"status": "error", "details": {"reason": f"tone scoring failed: {e}"}}

    score = result["score"]
    status = "pass" if score >= threshold else "fail"

    return {
        "status": status,
        "details": {
            "score": round(score, 4),
            "threshold": threshold,
            "judge_reason": result["reason"],
            "judge_model": result["model"],
        },
    }
