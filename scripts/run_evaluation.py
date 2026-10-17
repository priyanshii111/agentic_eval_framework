import argparse
import json
import sys
from pathlib import Path

import yaml

from agent_eval.evaluators.deterministic.deterministic_evaluator import (
    evaluate_case as evaluate_deterministic,
)
from agent_eval.evaluators.statistical.statistical_evaluator import (
    evaluate_case as evaluate_statistical,
)
from agent_eval.evaluators.adversarial.adversarial_evaluator import (
    evaluate_case as evaluate_adversarial,
)
# from agent_eval.evaluators.security.security_evaluator import (
#     evaluate_case as evaluate_security,
# )  # add once the security layer exists
from agent_eval.runner import live

# Live domains: a real agent is called and its answers are captured at runtime.
# agent_config is relative to --configs-dir.
LIVE_DOMAINS = {
    "airline": {"agent_config": "airline/agent.yaml"},
}

LAYER_EVALUATORS = {
    "deterministic": evaluate_deterministic,
    "statistical": evaluate_statistical,
    "adversarial": evaluate_adversarial,
}

FIBERNET_DETERMINISTIC_EXPECTED = {
    "fiber-tool-01": "pass",
    "fiber-exact-01": "pass",
    "fiber-constraints-01": "pass",
    "fiber-schema-01-pass": "pass",
    "fiber-schema-01-fail": "fail",
}

FIBERNET_STATISTICAL_EXPECTED = {
    "fiber-similarity-01-pass": "pass",
    "fiber-similarity-02-fail": "fail",
    "fiber-similarity-03-pass": "pass",
    "fiber-similarity-04-fail": "fail",
}

FIBERNET_ADVERSARIAL_EXPECTED = {
    "fiber-injection-01-pass": "pass",
    "fiber-injection-01-fail": "fail",
    "fiber-injection-02-pass": "pass",
    "fiber-injection-02-fail": "fail",
    "fiber-injection-03-pass": "pass",
    "fiber-injection-03-fail": "fail",
    "fiber-injection-04-pass": "pass",
    "fiber-injection-04-fail": "fail",
}

BANKBOT_DETERMINISTIC_EXPECTED = {
    "bank-exact-01": "pass",
    "bank-schema-01-pass": "pass",
    "bank-schema-01-fail": "fail",
    "bank-constraints-01-pass": "pass",
    "bank-constraints-01-fail": "fail",
    "bank-tool-01-pass": "pass",
    "bank-tool-01-fail": "fail",
    "bank-regex-01-pass": "pass",
    "bank-regex-01-fail": "fail",
}

BANKBOT_STATISTICAL_EXPECTED = {
    "bank-similarity-01-pass": "pass",
    "bank-similarity-01-fail": "fail",
    "bank-similarity-02-pass": "pass",
    "bank-similarity-02-fail": "fail",
    "bank-consistency-01-pass": "pass",
    "bank-consistency-01-fail": "fail",
    "bank-tone-01-pass": "pass",
    "bank-tone-01-fail": "fail",
}

BANKBOT_ADVERSARIAL_EXPECTED = {
    "bank-conflict-01-pass": "pass",
    "bank-conflict-01-fail": "fail",
    "bank-jailbreak-01-pass": "pass",
    "bank-jailbreak-01-fail": "fail",
}


def load_cases(fixture_path: str) -> list:
    return json.loads(Path(fixture_path).read_text())


def load_yaml_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def build_domains(fixtures_dir: str, configs_dir: str) -> dict:
    fixtures_dir = Path(fixtures_dir)
    configs_dir = Path(configs_dir)

    return {
        "fibernet": [
            {
                "name": "deterministic",
                "fixture_path": str(fixtures_dir / "fiberNet" / "deterministic.json"),
                "evaluate_fn": evaluate_deterministic,
                "expected_status": FIBERNET_DETERMINISTIC_EXPECTED,
                "global_config": {},
            },
            {
                "name": "statistical",
                "fixture_path": str(fixtures_dir / "fiberNet" / "statistical.json"),
                "evaluate_fn": evaluate_statistical,
                "expected_status": FIBERNET_STATISTICAL_EXPECTED,
                "global_config": load_yaml_config(str(configs_dir / "fiberNet" / "statistical.yaml")),
            },
            {
                "name": "adversarial",
                "fixture_path": str(fixtures_dir / "fiberNet" / "adversarial.json"),
                "evaluate_fn": evaluate_adversarial,
                "expected_status": FIBERNET_ADVERSARIAL_EXPECTED,
                "global_config": load_yaml_config(str(configs_dir / "fiberNet" / "adversarial.yaml")),
            },
        ],
        "bankbot": [
            {
                "name": "deterministic",
                "fixture_path": str(fixtures_dir / "bankBot" / "deterministic.json"),
                "evaluate_fn": evaluate_deterministic,
                "expected_status": BANKBOT_DETERMINISTIC_EXPECTED,
                "global_config": {},
            },
            {
                "name": "statistical",
                "fixture_path": str(fixtures_dir / "bankBot" / "statistical.json"),
                "evaluate_fn": evaluate_statistical,
                "expected_status": BANKBOT_STATISTICAL_EXPECTED,
                "global_config": load_yaml_config(str(configs_dir / "bankBot" / "statistical.yaml")),
            },
            {
                "name": "adversarial",
                "fixture_path": str(fixtures_dir / "bankBot" / "adversarial.json"),
                "evaluate_fn": evaluate_adversarial,
                "expected_status": BANKBOT_ADVERSARIAL_EXPECTED,
                "global_config": load_yaml_config(str(configs_dir / "bankBot" / "adversarial.yaml")),
            },
        ],
    }


def run_layer(name: str, fixture_path: str, evaluate_fn, expected_status: dict, global_config: dict):
    cases = load_cases(fixture_path)
    mismatches = []

    print(f"\n=== {name} layer ({fixture_path}) ===")
    for case in cases:
        if global_config:
            case = {**case, "config": {**global_config, **case.get("config", {})}}

        result = evaluate_fn(case)

        expected = expected_status.get(case["id"])
        actual = result["status"]

        if expected is None:
            marker = "NO-PREDICTION"
            ok = None
        else:
            ok = actual == expected
            marker = "PASS" if ok else "MISMATCH"

        print(
            f"[{marker}] {case['id']:<26} eval_type={case['eval_type']:<15} "
            f"status={actual:<6} expected={expected or '?':<6} reason={result['reason']}"
        )

        if ok is False:
            mismatches.append((case["id"], expected, actual))

    return len(cases), mismatches


def _run_static_suite(args) -> tuple:
    """Run the static fixtures and print their report. Returns (cases run, mismatches)."""
    domains = build_domains(args.fixtures_dir, args.configs_dir)
    selected = list(domains.keys()) if args.domain == "all" else [args.domain]

    total_cases = 0
    total_mismatches = []

    for domain_name in selected:
        layers = domains[domain_name]
        if not layers:
            print(f"\n(no layers configured yet for domain '{domain_name}')")
            continue

        print(f"\n########## DOMAIN: {domain_name} ##########")
        for layer in layers:
            if args.layer and layer["name"] != args.layer:
                continue
            n, mismatches = run_layer(
                layer["name"],
                layer["fixture_path"],
                layer["evaluate_fn"],
                layer["expected_status"],
                layer["global_config"],
            )
            total_cases += n
            total_mismatches += mismatches

    print()
    print(f"{total_cases} case(s) run, {len(total_mismatches)} mismatch(es).")

    if total_mismatches:
        print("\nMismatches:")
        for case_id, expected, actual in total_mismatches:
            print(f"  - {case_id}: expected {expected}, got {actual}")
    else:
        print("All cases matched expected outcomes.")
    return total_cases, total_mismatches


def run_static(args):
    _, mismatches = _run_static_suite(args)
    sys.exit(1 if mismatches else 0)


def _has_error(results: list) -> bool:
    """Live/replay: an `error` means our tooling broke. A `fail` is a finding
    about the agent, not a broken run."""
    return any(r["status"] == "error" for r in results)


def _exit_for_live_results(all_results):
    """Live/replay exit code: 1 only if our tooling broke (an `error`)."""
    live.print_summary(all_results)
    sys.exit(1 if _has_error(all_results) else 0)


def _run_live_suite(args) -> list:
    """Capture + evaluate every selected live domain. Returns all results."""
    selected = list(LIVE_DOMAINS) if args.domain == "all" else [args.domain]
    all_results = []
    for domain in selected:
        all_results += live.run_live(
            domain,
            Path(args.configs_dir) / LIVE_DOMAINS[domain]["agent_config"],
            Path(args.fixtures_dir),
            Path(args.configs_dir),
            Path(args.runs_dir),
            LAYER_EVALUATORS,
            layers=_selected_layers(args),
        )
    return all_results


def run_live_mode(args):
    _exit_for_live_results(_run_live_suite(args))


def run_all_mode(args):
    """The whole project in one command: static regression first, then live evaluation.

    Exit code 1 if the static suite has a mismatch OR a live case is `error`
    (something in our framework/tooling is wrong). Live `fail`s are findings
    about the agent and don't change the exit code.
    """
    print("\n==================== PART 1/2: STATIC REGRESSION ====================")
    static_cases, mismatches = _run_static_suite(args)

    print("\n==================== PART 2/2: LIVE EVALUATION ======================")
    live_results = _run_live_suite(args)
    live.print_summary(live_results)

    by_status = {s: sum(r["status"] == s for r in live_results) for s in ("pass", "fail", "error")}
    print("\n======================== OVERALL SUMMARY ============================")
    print(f"STATIC: {static_cases} case(s), {len(mismatches)} mismatch(es)")
    print(f"LIVE:   {len(live_results)} case(s): pass {by_status['pass']}, "
          f"fail {by_status['fail']}, error {by_status['error']}")
    ok = not mismatches and not _has_error(live_results)
    print("Framework OK." if ok else "Framework problem: see mismatches / errors above.")
    sys.exit(0 if ok else 1)


def run_replay_mode(args):
    _exit_for_live_results(live.run_replay(Path(args.run), LAYER_EVALUATORS, layers=_selected_layers(args)))


def _selected_layers(args) -> tuple:
    return (args.layer,) if args.layer else live.LAYER_ORDER


MODES = {
    "static": run_static,
    "live": run_live_mode,
    "replay": run_replay_mode,
    "all": run_all_mode,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the evaluation framework for one specific use case, or all of them."
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="static",
        help="static: hand-authored fixtures with predicted outcomes (default, regression check). "
        "live: call the real agent, save its answers under --runs-dir, grade them. "
        "replay: re-grade a saved live run (--run) without calling the agent. "
        "all: the whole project in one go - static regression, then live evaluation.",
    )
    parser.add_argument(
        "--domain",
        choices=["fibernet", "bankbot", *LIVE_DOMAINS, "all"],
        default="all",
        help="Which use case to run. 'all' runs every domain of the chosen mode — in static mode, "
        "use this after editing a shared evaluator file, as a regression check against every "
        "domain at once.",
    )
    parser.add_argument(
        "--layer",
        choices=list(live.LAYER_ORDER),
        help="Run only this layer (default: all). Useful in live mode, where a full run takes long on CPU.",
    )
    parser.add_argument("--fixtures-dir", default="fixtures/golden")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--runs-dir", default="runs", help="live mode: where captured runs are saved")
    parser.add_argument("--run", help="replay mode: the saved run folder, e.g. runs/2026-10-07_153000_airline")
    args = parser.parse_args()

    if args.mode == "static" and args.domain in LIVE_DOMAINS:
        parser.error(f"'{args.domain}' is a live domain - use --mode live")
    if args.mode == "live" and args.domain not in (*LIVE_DOMAINS, "all"):
        parser.error(f"'{args.domain}' is a static domain - live domains: {list(LIVE_DOMAINS)}")
    if args.mode == "replay" and not args.run:
        parser.error("--mode replay needs --run <saved run folder>")
    if args.mode == "all" and args.domain != "all":
        parser.error("--mode all runs every domain (static + live) - leave out --domain")

    MODES[args.mode](args)


if __name__ == "__main__":
    main()
