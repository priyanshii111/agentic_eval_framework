from .. import case_runner
from . import constraints, exact_match, json_schema, regex, tool_call

DISPATCH = {
    "exact_match": exact_match.evaluate,
    "json_schema": json_schema.evaluate,
    "constraints": constraints.evaluate,
    "tool_call": tool_call.evaluate,
    "regex": regex.evaluate,
}


def _reason_exact_match(status: str, details: dict) -> str:
    if status == "pass":
        return f"Output matched expected value {details['expected']!r}"
    return f"Expected {details['expected']!r} but got {details.get('actual')!r}"


def _reason_json_schema(status: str, details: dict) -> str:
    if status == "pass":
        return "Output matches the expected schema"
    if details.get("parse_failed"):
        return "Output is not valid JSON"
    errors = details.get("errors", [])
    if errors:
        return f"Schema validation failed ({len(errors)} error(s)): {errors[0]['message']}"
    return "Schema validation failed"


def _reason_constraints(status: str, details: dict) -> str:
    failed = details.get("failed_rules", [])
    if status == "pass":
        return "All constraints were satisfied"
    return f"{len(failed)} constraint(s) failed: " + "; ".join(f["reason"] for f in failed)


def _reason_tool_call(status: str, details: dict) -> str:
    if status == "pass":
        return "All expected tool calls were made correctly"
    parts = []
    if details.get("missing_tools"):
        parts.append(f"missing calls to {details['missing_tools']}")
    if details.get("unexpected_tools"):
        parts.append(f"unexpected calls to {details['unexpected_tools']}")
    if details.get("arg_mismatches"):
        parts.append(f"{len(details['arg_mismatches'])} argument mismatch(es)")
    if details.get("order_violated"):
        parts.append("calls were made in the wrong order")
    return "; ".join(parts) if parts else "Tool call check failed"


def _reason_regex(status: str, details: dict) -> str:
    if status == "pass":
        return f"Value matched pattern {details['pattern']!r}"
    return f"Value {details.get('value')!r} did not match pattern {details['pattern']!r}"


REASON_BUILDERS = {
    "exact_match": _reason_exact_match,
    "json_schema": _reason_json_schema,
    "constraints": _reason_constraints,
    "tool_call": _reason_tool_call,
    "regex": _reason_regex,
}


def evaluate_case(case: dict) -> dict:
    return case_runner.evaluate_case(case, DISPATCH, REASON_BUILDERS)
