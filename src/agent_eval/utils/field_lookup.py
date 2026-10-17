from typing import Any


class FieldNotFoundError(Exception):
    """Raised when a field path cannot be resolved against the given data."""


def get_nested_field(data: dict, path: str) -> Any:
    """Resolve a dot-separated path (e.g. "usage.data_used_gb") against a nested dict."""
    if not isinstance(data, dict):
        raise FieldNotFoundError(
            f"Cannot resolve '{path}': expected an object, got {type(data).__name__}"
        )

    current: Any = data
    parts = path.split(".")
    for i, part in enumerate(parts):
        if not isinstance(current, dict):
            walked = ".".join(parts[:i])
            raise FieldNotFoundError(
                f"Cannot resolve '{path}': '{walked}' is not an object (got {type(current).__name__})"
            )
        if part not in current:
            raise FieldNotFoundError(f"Cannot resolve '{path}': missing key '{part}'")
        current = current[part]

    return current


def resolve_field(actual_output: dict, field: str) -> Any:
    """
    Resolve a field reference against an AgentResponse-shaped dict
    ({output_text, output_json, tool_calls}).

    "output_text" is a literal reference to the raw string. Any other
    field is treated as a dot-path into output_json.
    """
    if field == "output_text":
        return actual_output.get("output_text")

    output_json = actual_output.get("output_json")
    if output_json is None:
        raise FieldNotFoundError(
            f"Cannot resolve '{field}': output_json is not available (missing or invalid JSON)"
        )

    return get_nested_field(output_json, field)
