import json

from ...utils.field_lookup import FieldNotFoundError, resolve_field


def evaluate(actual_output: dict, expected: dict, config: dict | None = None) -> dict:
    config = config or {}       #use empty config, when nothing is passed
    case_sensitive = config.get("case_sensitive", True)
    trim_whitespace = config.get("trim_whitespace", True)

    field = expected.get("field_path", "output_text")       #field we want to check, return output_text in default
    expected_value: str = expected["expected_output"]       #expected value for that field

    try:
        actual_value = resolve_field(actual_output, field)      #actual value from agent output
    except FieldNotFoundError as e:                             # if field is missing
        return {
            "status": "fail",
            "details": {"expected": expected_value, "actual": None, "reason": str(e)},
        }

    #converting non string values into json-style 

    actual_str = actual_value if isinstance(actual_value, str) else json.dumps(actual_value)
    expected_str = expected_value

    if trim_whitespace:             # if trim spaces enabled
        actual_str = actual_str.strip()
        expected_str = expected_str.strip()
    if not case_sensitive:          #ignore case, used if case sensitivity is disabled
        actual_str = actual_str.lower()
        expected_str = expected_str.lower()

    matched = actual_str == expected_str
    return {
        "status": "pass" if matched else "fail",
        "details": {"expected": expected_value, "actual": actual_value},
    }
