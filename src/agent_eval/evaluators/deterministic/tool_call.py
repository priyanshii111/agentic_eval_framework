#tool call
def _args_match(expected_args: dict, actual_args: dict, mode: str) -> bool:
    if mode == "strict":        # strict mode-> must match exactly
        return expected_args == actual_args
    return all(                 # flexible mode -> expected keys must match, extra keys allowed
        key in actual_args and actual_args[key] == value
        for key, value in expected_args.items()
    )


def evaluate(actual_output: dict, expected: dict, config: dict | None = None) -> dict:
    config = config or {}

    # get actual and expected tool calls
    actual_calls = actual_output.get("tool_calls") or []
    expected_calls = expected["expected_tool_calls"]

    #config options
    enforce_order = expected.get("enforce_order", False)
    arg_match_mode = expected.get("arg_match_mode", config.get("arg_match_mode", "flexible"))
    must_follow_rules = expected.get("must_follow", [])
    # off by default: any call that wasn't expected fails the case. Live agents often make
    # harmless extra lookups, so a case can opt in; extras are still listed in details.
    allow_extra_calls = expected.get("allow_extra_calls", config.get("allow_extra_calls", False))

    # tracing list

    used_actual_indices = set()
    matched_positions = []
    matched_index_by_tool ={}
    missing_tools = []
    arg_mismatches = []

    for expected_call in expected_calls:        #loop through expected tool calls
        tool_name = expected_call["tool"]
        expected_args = expected_call.get("args", {})

        exact_match_index = None
        same_tool_index = None
        for i, actual_call in enumerate(actual_calls):      #search for actual matching tool call
            if i in used_actual_indices or actual_call.get("tool") != tool_name:
                continue
            if same_tool_index is None:
                same_tool_index = i
            if _args_match(expected_args, actual_call.get("args", {}), arg_match_mode):
                exact_match_index = i
                break

        # handle match/mismatch/missing cases

        if exact_match_index is not None:
            used_actual_indices.add(exact_match_index)
            matched_positions.append(exact_match_index)
            matched_index_by_tool[tool_name] = exact_match_index
        elif same_tool_index is not None:
            used_actual_indices.add(same_tool_index)
            arg_mismatches.append({
                "tool": tool_name,
                "expected_args": expected_args,
                "actual_args": actual_calls[same_tool_index].get("args", {}),
            })
        else:
            missing_tools.append(tool_name)

    # find unexpected tools
    unexpected_tools = [
        actual_calls[i]["tool"]
        for i in range(len(actual_calls))
        if i not in used_actual_indices
    ]
    #check order violation
    order_violated = enforce_order and matched_positions != sorted(matched_positions)
    prerequisite_violations = []
    for rule in must_follow_rules:
        tool_index = matched_index_by_tool.get(rule["tool"])
        prerequisite_index = matched_index_by_tool.get(rule["after"])
        if tool_index is None or prerequisite_index is None:
            continue
        if tool_index < prerequisite_index:
            prerequisite_violations.append(f"'{rule['tool']}' was called before'{rule['after']}', not after")

    # final status
    extra_calls_fail = bool(unexpected_tools) and not allow_extra_calls
    failed = bool(missing_tools or arg_mismatches or extra_calls_fail or order_violated or prerequisite_violations)

    return {
        "status": "fail" if failed else "pass",
        "details": {
            "missing_tools": missing_tools,
            "unexpected_tools": unexpected_tools,
            "allow_extra_calls": allow_extra_calls,
            "arg_mismatches": arg_mismatches,
            "order_violated": order_violated,
            "prerequisite_violations": prerequisite_violations, 
        },
    }
