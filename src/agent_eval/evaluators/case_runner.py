"""The case loop shared by every layer evaluator: dispatch a case to its checker,
time it, and wrap the outcome in the standard result shape with a readable reason.
Each layer only supplies its own DISPATCH and REASON_BUILDERS tables.
"""
import time


def apply_layer_config(case: dict, layer_config: dict) -> dict:
    """The case with the layer config merged into its "config" (the case's own config wins)."""
    return {**case, "config": {**layer_config, **case.get("config", {})}}


def _build_reason(reason_builders: dict, eval_type: str, status: str, details: dict) -> str:
    if status == "error":
        return details.get("reason", f"{eval_type} checker encountered an error")
    builder = reason_builders.get(eval_type)
    if builder is None:
        return f"{eval_type} check {status}"
    return builder(status, details)


def evaluate_case(case: dict, dispatch: dict, reason_builders: dict, multi_sample_eval_types=frozenset()) -> dict:
    """Grade one case. eval_types in multi_sample_eval_types read case["actual_outputs"]
    (a list of samples) instead of case["actual_output"]."""
    case_id = case["id"]
    eval_type = case["eval_type"]
    severity = case.get("severity")

    checker = dispatch.get(eval_type)
    if checker is None:
        status, details, duration_ms = "error", {"reason": f"unknown eval_type '{eval_type}'"}, 0.0
    else:
        multi_sample = eval_type in multi_sample_eval_types
        samples = case["actual_outputs"] if multi_sample else None

        start = time.perf_counter()
        try:
            input_data = samples if multi_sample else case["actual_output"]
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
        "reason": _build_reason(reason_builders, eval_type, status, details),
        "details": details,
        "duration_ms": duration_ms,
    }
