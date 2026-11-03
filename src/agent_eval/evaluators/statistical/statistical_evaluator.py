from .. import case_runner
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


def evaluate_case(case: dict) -> dict:
    return case_runner.evaluate_case(case, DISPATCH, REASON_BUILDERS, MULTI_SAMPLE_EVAL_TYPES)
