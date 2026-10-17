import json
import operator

from ...utils.field_lookup import FieldNotFoundError, resolve_field


def _resolve_or_none(actual_output: dict, field: str):
    try:
        return resolve_field(actual_output, field), None
    except FieldNotFoundError as e:
        return None, str(e)


def _check_range(actual_output: dict, rule: dict, config: dict):
    value, err = _resolve_or_none(actual_output, rule["field"])
    if err:
        return False, err

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False, f"field '{rule['field']}' is not numeric (got {type(value).__name__})"

    tolerance = config.get("float_tolerance", 0)
    lo = rule.get("min")
    hi = rule.get("max")
    if lo is not None and value < lo - tolerance:
        return False, f"{rule['field']}={value} is below min {lo}"
    if hi is not None and value > hi + tolerance:
        return False, f"{rule['field']}={value} is above max {hi}"
    return True, None


def _check_contains(actual_output: dict, rule: dict, config: dict):
    value, err = _resolve_or_none(actual_output, rule["field"])
    if err:
        return False, err

    haystack = value if isinstance(value, str) else json.dumps(value)
    needle = rule["value"]
    if needle not in haystack:
        return False, f"'{needle}' not found in field '{rule['field']}'"
    return True, None


def _check_one_of(actual_output: dict, rule: dict, config: dict):
    value, err = _resolve_or_none(actual_output, rule["field"])
    if err:
        return False, err

    value_str = value if isinstance(value, str) else json.dumps(value)
    allowed = rule["values"]
    if value_str not in allowed:
        return False, f"{rule['field']}={value_str!r} is not in allowed set {allowed}"
    return True, None


_COMPARISON_OPERATORS = {
    "lte": operator.le,
    "gte": operator.ge,
    "lt": operator.lt,
    "gt": operator.gt,
    "eq": operator.eq,
}


def _check_field_comparison(actual_output: dict, rule: dict, config: dict):
    value, err = _resolve_or_none(actual_output, rule["field"])
    if err:
        return False, err

    compare_value, err = _resolve_or_none(actual_output, rule["compare_to_field"])
    if err:
        return False, err

    for label, v in (("field", value), ("compare_to_field", compare_value)):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, f"{label} is not numeric (got {type(v).__name__})"

    op_name = rule["operator"]
    op = _COMPARISON_OPERATORS.get(op_name)
    if op is None:
        return False, f"unknown comparison operator '{op_name}'"

    if not op(value, compare_value):
        return False, (
            f"{rule['field']}={value} does not satisfy '{op_name}' against "
            f"{rule['compare_to_field']}={compare_value}"
        )
    return True, None


RULE_HANDLERS = {
    "range": _check_range,
    "contains": _check_contains,
    "one_of": _check_one_of,
    "field_comparison": _check_field_comparison,
}


def evaluate(actual_output: dict, expected: dict, config: dict | None = None) -> dict:
    config = config or {}
    rules = expected["constraints"]["all"]

    failed_rules = []
    for rule in rules:
        handler = RULE_HANDLERS.get(rule["type"])
        if handler is None:
            return {
                "status": "error",
                "details": {"reason": f"unknown constraint type '{rule['type']}'"},
            }

        passed, reason = handler(actual_output, rule, config)
        if not passed:
            failed_rules.append({"rule": rule, "reason": reason})

    return {
        "status": "pass" if not failed_rules else "fail",
        "details": {"failed_rules": failed_rules},
    }
