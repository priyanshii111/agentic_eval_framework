import jsonschema


def _apply_strict_extra_fields(schema):
    #making schema strict by disallowing extra fields

    if not isinstance(schema, dict):
        return schema

    result = dict(schema)       
    if result.get("type") == "object" and "additionalProperties" not in result:     # if it's an object and does not specify additional properties, set it to false
        result["additionalProperties"] = False

    if "properties" in result:          
        result["properties"] = {
            key: _apply_strict_extra_fields(value)
            for key, value in result["properties"].items()
        }

    return result

# agent's output against expected schema

def evaluate(actual_output: dict, expected: dict, config: dict | None = None) -> dict:
    config = config or {}

    output_json = actual_output.get("output_json")      # getting agent's output_json
    if output_json is None:
        return {"status": "fail", "details": {"parse_failed": True}}

    schema = expected["expected_schema"]                # get expected schema
    if config.get("strict_extra_fields", True):         # if strict mode enabled, disallowed extra fields
        schema = _apply_strict_extra_fields(schema)

    try:                # first check if schema itself is valid
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as e:
        return {
            "status": "error",
            "details": {"reason": f"invalid expected_schema: {e.message}"},
        }
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    #validator = jsonschema.Draft202012Validator(schema)         # validate output_json against schema
    errors = [
        {"path": ".".join(str(p) for p in err.path) or "(root)", "message": err.message}
        for err in validator.iter_errors(output_json)
    ]

    if errors:
        return {"status": "fail", "details": {"errors": errors}}
    return {"status": "pass", "details": {}}
