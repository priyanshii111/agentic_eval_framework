from ...utils.field_lookup import FieldNotFoundError, resolve_field


def _compare_field(actual_outputs: list, expected: dict, config: dict) -> dict:
    """Default mode: one field must be identical (or within tolerance) across all samples."""
    field = expected.get("field_path", "output_text")
    tolerance = expected.get("tolerance", config.get("float_tolerance", 0))

    values = []
    for i, sample in enumerate(actual_outputs):
        try:
            value = resolve_field(sample, field)
        except FieldNotFoundError as e:
            return {
                "status": "fail",
                "details": {"field": field, "reason": f"sample {i}: {e}"},
            }
        values.append(value)

    first = values[0]

    if isinstance(first, (int, float)) and not isinstance(first, bool):
        inconsistent = [
            v for v in values
            if not isinstance(v, (int, float)) or isinstance(v, bool) or abs(v - first) > tolerance
        ]
    else:
        inconsistent = [v for v in values if v != first]

    if inconsistent:
        return {
            "status": "fail",
            "details": {
                "field": field,
                "values": values,
                "reason": f"inconsistent values across {len(values)} sample(s): {values}",
            },
        }

    return {
        "status": "pass",
        "details": {"field": field, "values": values},
    }


def _compare_tool_names(actual_outputs: list, expected: dict, config: dict) -> dict:
    """Paraphrase mode: every sample must call the same SET of tools.

    Order and arguments are ignored on purpose (a rephrased question may
    legitimately search differently). Tools in `ignore_tools` are removed from
    every sample before comparing; details always keep the full raw lists.
    """
    ignore_tools = sorted(set(expected.get("ignore_tools", [])))
    raw_tool_calls = [[call.get("tool") for call in (sample.get("tool_calls") or [])] for sample in actual_outputs]
    compared = [sorted(set(tools) - set(ignore_tools)) for tools in raw_tool_calls]

    details = {
        # "field"/"values" keep the existing consistency reason text working unchanged.
        "field": "tool names",
        "values": compared,
        "compare": "tool_names",
        "ignore_tools": ignore_tools,
        "raw_tool_calls": raw_tool_calls,
    }
    if any(tools != compared[0] for tools in compared):
        details["reason"] = (
            f"different tools across {len(compared)} wording(s): {compared} "
            f"(raw calls: {raw_tool_calls}, ignored: {ignore_tools})"
        )
        return {"status": "fail", "details": details}
    return {"status": "pass", "details": details}


COMPARE_MODES = {
    "field": _compare_field,
    "tool_names": _compare_tool_names,
}


def evaluate(actual_outputs: list, expected: dict, config: dict | None = None) -> dict:
    config = config or {}

    if not actual_outputs:
        return {
            "status": "error",
            "details": {"reason": "no samples provided to check consistency"},
        }

    mode = expected.get("compare", "field")  # default = the original field comparison
    compare = COMPARE_MODES.get(mode)
    if compare is None:
        return {
            "status": "error",
            "details": {"reason": f"unknown consistency compare mode '{mode}' (known: {sorted(COMPARE_MODES)})"},
        }
    return compare(actual_outputs, expected, config)
