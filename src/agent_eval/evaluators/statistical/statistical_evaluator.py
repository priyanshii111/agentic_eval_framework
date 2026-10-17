import time

from . import aggregation, likert_scoring, scoring, tone_scoring

DISPATCH = {
    "similarity": scoring.evaluate,
    "consistency": aggregation.evaluate,
    "tone": tone_scoring.evaluate,
    "quality": likert_scoring.evaluate,
}

# eval_types whose case carries "actual_outputs" (a list of samples) instead
# of a single "actual_output" — consistency-checking needs multiple samples
# of the same case to compare against each other.
MULTI_SAMPLE_EVAL_TYPES = {"consistency"}


def _reason_similarity(status: str, details: dict) -> str:
    base = f"Score {details['score']} vs threshold {details['threshold']} (method={details['used_method']})"
    if details.get("primary_error"):
        base += f" — fell back from '{details['primary_method']}' after error: {details['primary_error']}"
    if details.get("judge_reason"):
        base += f" — judge said: {details['judge_reason']}"
    return base


def _reason_consistency(status: str, details: dict) -> str:
    if status == "pass":
        return f"Field '{details['field']}' was consistent across all samples: {details['values']}"
    return details.get("reason", "Consistency check failed")


def _reason_tone(status: str, details: dict) -> str:
    base = f"Tone score {details['score']} vs threshold {details['threshold']}"
    if details.get("judge_reason"):
        base += f" — judge said: {details['judge_reason']}"
    return base


def _reason_quality(status: str, details: dict) -> str:
    base = (
        f"{details['criterion'].capitalize()} {details['likert_value']}/{details['likert_max']} "
        f"(score {details['score']}) vs threshold {details['threshold']}"
    )
    if details.get("judge_reason"):
        base += f" - judge said: {details['judge_reason']}"
    return base


REASON_BUILDERS = {
    "similarity": _reason_similarity,
    "consistency": _reason_consistency,
    "tone": _reason_tone,
    "quality": _reason_quality,
}


def _build_reason(eval_type: str, status: str, details: dict) -> str:
    if status == "error":
        return details.get("reason", f"{eval_type} checker encountered an error")
    builder = REASON_BUILDERS.get(eval_type)
    if builder is None:
        return f"{eval_type} check {status}"
    return builder(status, details)


def evaluate_case(case: dict) -> dict:
    case_id = case["id"]
    eval_type = case["eval_type"]
    severity = case.get("severity")

    checker = DISPATCH.get(eval_type)
    if checker is None:
        status, details, duration_ms = "error", {"reason": f"unknown eval_type '{eval_type}'"}, 0.0
    else:
        input_data = case["actual_outputs"] if eval_type in MULTI_SAMPLE_EVAL_TYPES else case["actual_output"]

        start = time.perf_counter()
        try:
            outcome = checker(input_data, case["expected"], case.get("config", {}))
            status, details = outcome["status"], outcome["details"]
        except Exception as e:
            status, details = "error", {"reason": f"unhandled exception in {eval_type} checker: {e}"}
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "case_id": case_id,
        "eval_type": eval_type,
        "status": status,
        "severity": severity,
        "reason": _build_reason(eval_type, status, details),
        "details": details,
        "duration_ms": duration_ms,
    }
