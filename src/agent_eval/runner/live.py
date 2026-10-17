"""Live and replay runs: capture a real agent's answers, save them, grade them, report.

    live:   fixture cases -> agent adapter -> runs/<id>/<layer>.json -> evaluators -> report
    replay: runs/<id>/<layer>.json -> evaluators -> report (no agent call at all)

Capturing and grading are separate steps on purpose: the agent may answer
differently each time, but grading a saved capture is always repeatable (Rule 1).
The evaluators are the same, unchanged functions static mode uses; they never
see where actual_output came from.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from ..integrations.agent import AgentCallError, create_adapter

LAYER_ORDER = ("deterministic", "statistical", "adversarial")
RUN_META_FILE = "run.json"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class _LazyAdapter:
    """Builds the agent adapter only when the first case actually needs a live
    call (building it takes ~1 min), and at most once per run."""

    def __init__(self, agent_config: dict):
        self._agent_config = agent_config
        self._adapter = None
        self._build_error = None

    def call(self, case_input: dict) -> dict:
        if self._build_error is not None:
            raise AgentCallError(self._build_error)
        if self._adapter is None:
            print(" (starting the agent)", end="", flush=True)
            try:
                self._adapter = create_adapter(self._agent_config)
            except Exception as e:
                self._build_error = f"could not start the agent: {e}"
                raise AgentCallError(self._build_error) from e
        return self._adapter.call(case_input)


class _AnswerCache:
    """Within one live run, each distinct input is sent to the agent only once.

    Several cases often ask exactly the same thing (same message, passenger
    and attack spec) to grade it with different checks; they all grade that
    one answer instead of paying for another ~30 s CPU call. Failed calls are
    not cached, so a later case retries.
    """

    def __init__(self):
        self._answers = {}

    def get_or_call(self, call_input: dict, case_id: str, call_fn) -> tuple:
        """Returns (response, id of the case that first asked it, or None if fresh)."""
        key = json.dumps(call_input, sort_keys=True, default=str)
        if key in self._answers:
            response, first_case_id = self._answers[key]
            return response, first_case_id
        response = call_fn(call_input)
        self._answers[key] = (response, case_id)
        return response, None


def _capture_multi(case_input: dict, adapter: _LazyAdapter, cache: _AnswerCache, case_id: str) -> tuple:
    """One agent call per wording in input["user_messages"] (paraphrase
    consistency). Each wording runs as its own fresh conversation, with the
    rest of the input (e.g. passenger_id) shared."""
    shared = {k: v for k, v in case_input.items() if k != "user_messages"}
    outputs, traces, reused_from = [], [], []
    for message in case_input["user_messages"]:
        response, first = cache.get_or_call({**shared, "user_message": message}, case_id, adapter.call)
        outputs.append(response["actual_output"])
        traces.append(response["trace"])
        reused_from.append(first)
    return outputs, traces, reused_from


def _capture_attack(case_input: dict, adapter: _LazyAdapter, cache: _AnswerCache, case_id: str) -> tuple:
    """Deliver the case's PyRIT attack (input["attack"]) through the same adapter."""
    try:
        # Imported here: only attack cases need PyRIT; everything else runs without it.
        from ..integrations.pyrit_attacks import run_attack
    except ImportError as e:
        raise AgentCallError(f"PyRIT is not installed ({e}); install with: pip install -e \".[pyrit]\"") from e

    result, first = cache.get_or_call(case_input, case_id, lambda inp: run_attack(adapter, inp))
    capture = {"source": "live", "trace": result["trace"], "attack": result["attack"]}
    return result["actual_output"], capture, first


# Graders whose verdict is exactly repeatable. Only these may decide, during
# capture, whether the agent fell for an attack variant (an LLM judge could
# decide differently on replay).
VARIANT_GRADERS = {"unauthorized_action", "instruction_conflict", "prompt_injection", "jailbreak"}


def _capture_attack_variants(case: dict, adapter: _LazyAdapter, cache: _AnswerCache, attack_plan: dict | None) -> tuple:
    """Try the case's attack in every configured variant, round by round.

    Round 1 tries every strategy; if the agent resisted all of them, round 2
    retries each strategy with each converter. Stops after the first round in
    which the agent falls. Every attempt is saved; the decisive one (first
    failure, else the last attempt) becomes the case's actual_output.
    """
    if not attack_plan:
        raise AgentCallError("attack variants need attack_strategies / attack_converters in the layer config")
    if case["eval_type"] not in VARIANT_GRADERS:
        raise AgentCallError(
            f"attack variants need a deterministic grader ({sorted(VARIANT_GRADERS)}), got '{case['eval_type']}'"
        )
    try:
        from ..integrations.pyrit_attacks import attack_rounds, run_attack, variant_name
    except ImportError as e:
        raise AgentCallError(f"PyRIT is not installed ({e}); install with: pip install -e \".[pyrit]\"") from e

    case_input = case["input"]
    attempts, statuses = [], []
    for round_specs in attack_rounds(attack_plan["strategies"], attack_plan["converters"]):
        for spec in round_specs:
            result, first = cache.get_or_call(
                {**case_input, "attack": spec}, case["id"], lambda inp: run_attack(adapter, inp)
            )
            status = attack_plan["grade"](case, result["actual_output"])
            attempts.append({
                "variant": variant_name(spec),
                **result["attack"],
                "actual_output": result["actual_output"],
                "trace": result["trace"],
                "reused_from": first,
            })
            statuses.append(status)
            print(f" [{variant_name(spec)}: {status}]", end="", flush=True)
        if any(s != "pass" for s in statuses):
            break

    decisive = next((a for a, s in zip(attempts, statuses) if s == "fail"), attempts[-1])
    capture = {"source": "live", "trace": decisive["trace"], "attack": _decisive_attack(decisive), "attempts": attempts}
    return decisive["actual_output"], capture


def _decisive_attack(attempt: dict) -> dict:
    return {key: attempt[key] for key in ("strategy", "converters", "prompt_sent")}


def capture_cases(cases: list, adapter: _LazyAdapter, cache: _AnswerCache | None = None,
                  attack_plan: dict | None = None) -> list:
    """Fill in the agent's answer for every case that doesn't already have one.

    - input "user_message"  -> one call  -> actual_output
    - input "user_messages" -> one call per wording -> actual_outputs (plural),
      the shape multi-sample eval_types like `consistency` already read.
    - input "attack"        -> delivered through PyRIT -> actual_output, with the
      attack details in capture["attack"]. With an explicit strategy/converters
      it is one attempt; otherwise every variant in `attack_plan` is tried
      round by round (see _capture_attack_variants) and saved in capture["attempts"].

    A case that already carries actual_output(s) is kept as-is (graded like a
    static fixture). A failed agent call is recorded, never raised, so one bad
    case doesn't stop the run. Identical inputs are answered once (`cache`,
    shared across layers by run_live); capture["reused_from"] names the case
    whose answer was reused.
    """
    cache = cache if cache is not None else _AnswerCache()
    captured = []
    for case in cases:
        record = dict(case)
        case_input = case.get("input", {})
        if "actual_output" in case or "actual_outputs" in case:
            record["capture"] = {"source": "fixture"}
        else:
            print(f"  capturing {case['id']} ...", end="", flush=True)
            try:
                attack = case_input.get("attack")
                if attack is not None and "strategy" not in attack and "converters" not in attack:
                    record["actual_output"], record["capture"] = _capture_attack_variants(case, adapter, cache, attack_plan)
                    reused = [None]  # reuse is recorded per attempt
                elif attack is not None:
                    record["actual_output"], record["capture"], first = _capture_attack(case_input, adapter, cache, case["id"])
                    reused = [first]
                elif "user_messages" in case_input:
                    record["actual_outputs"], traces, reused = _capture_multi(case_input, adapter, cache, case["id"])
                    record["capture"] = {"source": "live", "traces": traces}
                else:
                    response, first = cache.get_or_call(case_input, case["id"], adapter.call)
                    record["actual_output"] = response["actual_output"]
                    record["capture"] = {"source": "live", "trace": response["trace"]}
                    reused = [first]
                if any(reused):
                    record["capture"]["reused_from"] = reused if len(reused) > 1 else reused[0]
                    print(f" same input as {sorted(set(r for r in reused if r))}, reusing its answer", end="")
            except AgentCallError as e:
                record["capture"] = {"source": "live", "error": str(e)}
            print(flush=True)
        captured.append(record)
    return captured


def grade_cases(cases: list, evaluate_fn, global_config: dict) -> list:
    """Grade captured cases with an unchanged layer evaluator.

    A case whose capture failed gets status=error in the standard result
    shape — the agent never answered, so there is nothing to grade (Rule 2).
    """
    results = []
    for case in cases:
        capture_error = case.get("capture", {}).get("error")
        if capture_error:
            results.append({
                "case_id": case["id"],
                "eval_type": case["eval_type"],
                "status": "error",
                "severity": case.get("severity"),
                "reason": f"agent call failed: {capture_error}",
                "details": {"capture_error": capture_error},
                "duration_ms": 0.0,
            })
            continue

        if global_config:
            case = {**case, "config": {**global_config, **case.get("config", {})}}
        if case.get("capture", {}).get("attempts"):
            results.append(_grade_attempts(case, evaluate_fn))
        else:
            results.append(evaluate_fn(case))
    return results


def _grade_attempts(case: dict, evaluate_fn) -> dict:
    """Grade every saved attack variant with the case's own (unchanged) grader
    and combine them into one result: fail if the agent fell for any variant."""
    graded = [(a, evaluate_fn({**case, "actual_output": a["actual_output"]})) for a in case["capture"]["attempts"]]
    failed = [a["variant"] for a, r in graded if r["status"] == "fail"]
    errored = [(a["variant"], r["reason"]) for a, r in graded if r["status"] == "error"]
    resisted = [a["variant"] for a, r in graded if r["status"] == "pass"]
    tried = len(graded)

    if failed:
        status, reason = "fail", f"Agent fell for {failed} ({tried} variant(s) tried)"
    elif errored:
        status, reason = "error", f"grading error on '{errored[0][0]}': {errored[0][1]}"
    else:
        status, reason = "pass", f"Resisted all {tried} variant(s)"
        no_tools = [a["variant"] for a, _ in graded if not a["actual_output"].get("tool_calls")]
        if no_tools:
            reason += f"; no tool calls for {no_tools} - check: resisted, or didn't understand?"

    return {
        "case_id": case["id"],
        "eval_type": case["eval_type"],
        "status": status,
        "severity": case.get("severity"),
        "reason": reason,
        "details": {
            "variants_tried": tried,
            "failed_variants": failed,
            "resisted_variants": resisted,
            "attempts": [
                {"variant": a["variant"], "prompt_sent": a.get("prompt_sent"), "status": r["status"], "reason": r["reason"]}
                for a, r in graded
            ],
        },
        "duration_ms": round(sum(r.get("duration_ms", 0.0) for _, r in graded), 2),
    }


def print_layer_report(layer: str, results: list, cases: list) -> None:
    """One line per case. A case's optional "predicted_status" (Rule 5: predict
    in writing first) is shown next to the real status — for learning only, it
    is never counted as a mismatch: nobody controls what a live agent does."""
    predicted = {c["id"]: c.get("predicted_status") for c in cases}
    print(f"\n=== {layer} layer ===")
    for r in results:
        guess = predicted.get(r["case_id"])
        guess_note = "" if guess is None else f" predicted={guess}{'' if guess == r['status'] else ' (!)'}"
        print(
            f"[{r['status'].upper():<5}] {r['case_id']:<28} eval_type={r['eval_type']:<15} "
            f"severity={str(r['severity']):<8}{guess_note} reason={r['reason']}"
        )

    with_guess = [r for r in results if predicted.get(r["case_id"]) is not None]
    if with_guess:
        hits = sum(predicted[r["case_id"]] == r["status"] for r in with_guess)
        print(f"  predictions matched: {hits}/{len(with_guess)} (info only, not counted)")


def print_summary(all_results: list) -> None:
    by_status = Counter(r["status"] for r in all_results)
    print(
        f"\n{len(all_results)} case(s): pass {by_status['pass']}, "
        f"fail {by_status['fail']}, error {by_status['error']}"
    )
    for status in ("fail", "error"):
        by_severity = Counter(str(r["severity"]) for r in all_results if r["status"] == status)
        if by_severity:
            print(f"  {status} by severity: " + ", ".join(f"{s} {n}" for s, n in by_severity.most_common()))


def _attack_plan(layer_config: dict, evaluate_fn) -> dict | None:
    """Attack variants from the layer config, plus a grader to tell whether the
    agent fell for a variant. None if the layer config defines no strategies."""
    strategies = layer_config.get("attack_strategies")
    if not strategies:
        return None

    def grade(case: dict, actual_output: dict) -> str:
        config = {**layer_config, **case.get("config", {})}
        return evaluate_fn({**case, "actual_output": actual_output, "config": config})["status"]

    return {"strategies": strategies, "converters": layer_config.get("attack_converters", []), "grade": grade}


def run_live(domain: str, agent_config_path: Path, fixtures_dir: Path, configs_dir: Path,
             runs_dir: Path, evaluators: dict, layers: tuple = LAYER_ORDER) -> list:
    """Capture + grade every selected layer fixture that exists for the domain. Returns all results."""
    agent_config = _load_yaml(agent_config_path)
    run_dir = runs_dir / f"{datetime.now():%Y-%m-%d_%H%M%S}_{domain}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"\n########## LIVE: {domain} -> {run_dir} ##########")

    adapter = _LazyAdapter(agent_config)
    meta = {
        "domain": domain,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "agent_config": agent_config,
        "layer_configs": {},
    }

    # Phase 1: capture EVERY selected layer first, then (phase 2) grade them all.
    # The agent model and the judge model then swap in memory once, instead of
    # after every layer (each swap means a slow cold start on CPU).
    cache = _AnswerCache()  # shared across layers: identical inputs are answered once
    captured_by_layer, configs_by_layer = {}, {}
    for layer in (layer for layer in LAYER_ORDER if layer in layers):
        fixture_path = fixtures_dir / domain / f"{layer}.json"
        if not fixture_path.exists():
            print(f"\n(no {layer} fixture yet: {fixture_path})")
            continue

        configs_by_layer[layer] = _load_yaml(configs_dir / domain / f"{layer}.yaml")
        meta["layer_configs"][layer] = configs_by_layer[layer]
        _write_json(run_dir / RUN_META_FILE, meta)

        print(f"\n--- capturing {layer} ({fixture_path}) ---", flush=True)
        attack_plan = _attack_plan(configs_by_layer[layer], evaluators[layer])
        captured_by_layer[layer] = capture_cases(_read_json(fixture_path), adapter, cache, attack_plan)
        _write_json(run_dir / f"{layer}.json", captured_by_layer[layer])  # saved before grading: never lose a capture

    all_results = []
    for layer, captured in captured_by_layer.items():
        results = grade_cases(captured, evaluators[layer], configs_by_layer[layer])
        _write_json(run_dir / f"{layer}.results.json", results)
        print_layer_report(layer, results, captured)
        all_results += results

    print(f"\nSaved to {run_dir}")
    return all_results


def _print_replay_comparison(saved: list, results: list) -> None:
    """Compare re-graded results with the ones saved at capture time.

    Deterministic checks must match exactly. A layer with an LLM judge
    re-asks the judge on replay, so its explanation (and sometimes its score)
    can differ even when the verdict doesn't — say which, instead of a bare
    "False" that looks like something broke (Rule 7).
    """
    strip = lambda rs: [{k: v for k, v in r.items() if k != "duration_ms"} for r in rs]
    saved, new = strip(saved), strip(json.loads(json.dumps(results, default=str)))
    if saved == new:
        print("  matches the results saved at capture time: True")
        return

    verdicts = lambda rs: [(r["case_id"], r["status"]) for r in rs]
    changed = [n["case_id"] for s, n in zip(saved, new) if s != n]
    if verdicts(saved) == verdicts(new):
        print(f"  matches the results saved at capture time: False - every pass/fail/error verdict is identical; "
              f"only details differ in {changed} (an LLM judge re-scored them, so its wording/score can vary)")
    else:
        flipped = [n["case_id"] for s, n in zip(saved, new) if s["status"] != n["status"]]
        print(f"  matches the results saved at capture time: False - VERDICT CHANGED for {flipped}; "
              f"details differ in {changed}")


def run_replay(run_dir: Path, evaluators: dict, layers: tuple = LAYER_ORDER) -> list:
    """Re-grade a saved live run without calling the agent, using the grading
    config saved at capture time, and check it matches the saved results."""
    meta = _read_json(run_dir / RUN_META_FILE)
    print(f"\n########## REPLAY: {meta['domain']} <- {run_dir} ##########")

    all_results = []
    for layer in (layer for layer in LAYER_ORDER if layer in layers):
        capture_path = run_dir / f"{layer}.json"
        if not capture_path.exists():
            continue

        captured = _read_json(capture_path)
        results = grade_cases(captured, evaluators[layer], meta["layer_configs"].get(layer, {}))
        print_layer_report(layer, results, captured)

        saved_path = run_dir / f"{layer}.results.json"
        if saved_path.exists():
            _print_replay_comparison(_read_json(saved_path), results)
        all_results += results

    return all_results
