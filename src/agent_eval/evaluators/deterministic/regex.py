import json
import re

from ...utils.field_lookup import FieldNotFoundError, resolve_field


def evaluate(actual_output: dict, expected: dict, config: dict | None = None) -> dict:
    config = config or {}
    field = expected.get("field_path", "output_text")
    pattern = expected["pattern"]

    try:
        actual_value = resolve_field(actual_output, field)
    except FieldNotFoundError as e:
        return {
            "status": "fail",
            "details": {"pattern": pattern, "matched": False, "reason": str(e)},
        }

    text = actual_value if isinstance(actual_value, str) else json.dumps(actual_value)

    try:
        matched = re.search(pattern, text) is not None
    except re.error as e:
        return {
            "status": "error",
            "details": {"reason": f"invalid pattern '{pattern}': {e}"},
        }

    return {
        "status": "pass" if matched else "fail",
        "details": {"pattern": pattern, "matched": matched, "value": actual_value},
    }
