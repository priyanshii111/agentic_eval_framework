"""Manual check: ask the live agent one question through the real adapter.

Shows exactly what the framework will receive (the AgentResponse shape).
Run from the project root, e.g.:

    python scripts/try_agent.py "Find me a hotel in Basel"
    python scripts/try_agent.py "What time is my flight?" --passenger "3442 587242" --trace
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from agent_eval.integrations.agent import AgentCallError, create_adapter


def main():
    parser = argparse.ArgumentParser(description="Ask the live agent one question.")
    parser.add_argument("question", help="What to ask the agent")
    parser.add_argument("--passenger", help="Passenger ID to log in as (default: from config)")
    parser.add_argument("--config", default="configs/airline/agent.yaml")
    parser.add_argument("--trace", action="store_true", help="Also print every message incl. tool results")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    print("Starting the agent (building policy search, ~1 min)...", flush=True)
    start = time.perf_counter()
    try:
        adapter = create_adapter(config)
        print(f"Ready in {time.perf_counter() - start:.0f}s. Asking: {args.question!r}", flush=True)

        case_input = {"user_message": args.question}
        if args.passenger:
            case_input["passenger_id"] = args.passenger
        response = adapter.call(case_input)
    except AgentCallError as e:
        print(f"\nAGENT ERROR (would be recorded as status=error): {e}")
        sys.exit(1)

    output = response["actual_output"]
    trace = response["trace"]

    print(f"\n--- answered in {trace['duration_ms'] / 1000:.0f}s by {trace['model']} "
          f"(passenger {trace['passenger_id']}) ---")
    for call in output["tool_calls"]:
        print(f"TOOL CALL -> {call['tool']}({call['args']})")
    if not output["tool_calls"]:
        print("TOOL CALL -> (none)")
    print(f"\nANSWER:\n{output['output_text']}")

    print("\n--- actual_output (what the evaluators receive) ---")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if args.trace:
        print("\n--- trace (every message) ---")
        print(json.dumps(trace["messages"], indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
