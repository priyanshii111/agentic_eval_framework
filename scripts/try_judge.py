"""Manual check: ask the PyRIT Likert judge to rate one answer to one question.

Goes through the real `quality` checker, exactly as a live run would.
Run from the project root, e.g.:

    python scripts/try_judge.py "Find me a hotel in Basel" "Here are 3 hotels in Basel: Hilton, Hyatt, Holiday Inn."
    python scripts/try_judge.py "What time is my flight?" "Please give me your ticket number." --model qwen2.5:7b
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from agent_eval.evaluators.statistical import likert_scoring


def main():
    parser = argparse.ArgumentParser(description="Rate one answer with the PyRIT Likert judge.")
    parser.add_argument("question", help="The customer's question")
    parser.add_argument("answer", help="The bot's answer to rate")
    parser.add_argument("--criterion", default="relevance", choices=sorted(likert_scoring.CRITERIA))
    parser.add_argument("--config", default="configs/airline/statistical.yaml")
    parser.add_argument("--model", help="Judge model override, e.g. qwen2.5:7b (default: from config)")
    parser.add_argument("--threshold", type=float, help="Pass threshold override, 0-1 (default: from config)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        config["llm_judge_model"] = args.model

    expected = {"criterion": args.criterion, "question": args.question}
    if args.threshold is not None:
        expected["threshold"] = args.threshold

    print(f"Judging {args.criterion} with {config['llm_judge_model']} ...", flush=True)
    start = time.perf_counter()
    result = likert_scoring.evaluate({"output_text": args.answer}, expected, config)
    elapsed = time.perf_counter() - start

    details = result["details"]
    if result["status"] == "error":
        print(f"\nJUDGE ERROR (would be recorded as status=error): {details['reason']}")
        sys.exit(1)

    print(f"\n{result['status'].upper()}: {args.criterion} {details['likert_value']}/{details['likert_max']} "
          f"(score {details['score']} vs threshold {details['threshold']}) in {elapsed:.0f}s")
    print(f"judge said: {details['judge_reason']}")
    print("\n--- details (what the report receives) ---")
    print(json.dumps(details, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
