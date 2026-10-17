"""Deliver an attack to the live agent through PyRIT: a strategy + optional converters.

A fixture case describes the attack in its input:

    "input": {
      "user_message": "<the attack objective, sent as the prompt>",
      "passenger_id": "...",
      "attack": {"strategy": "skeleton_key", "converters": ["leetspeak"]}
    }

An empty "attack": {} means "try every configured variant" (attack_rounds;
the runner drives the rounds and stops once the agent fails).

This file only DELIVERS the attack and captures the agent's full answer
(text + tool calls) via AgentAdapterTarget. Grading stays in
evaluators/adversarial/strategies — they never know PyRIT was involved.

Only single-turn strategies are registered (the target is single-turn; PyRIT
folds any prepended history, e.g. SkeletonKey's, into one message).
"""
from pyrit.converter import Base64Converter, CharacterSpaceConverter, LeetspeakConverter, ROT13Converter
from pyrit.executor.attack import AttackConverterConfig, PromptSendingAttack, SkeletonKeyAttack
from pyrit.prompt_normalizer import ConverterConfiguration

from ..utils import pyrit_runtime
from .agent import AgentCallError
from .pyrit_target import AgentAdapterTarget

# strategy name -> PyRIT single-turn attack class. A new strategy = one entry.
ATTACK_STRATEGIES = {
    "prompt_sending": PromptSendingAttack,  # send the prompt as-is (after converters)
    "skeleton_key": SkeletonKeyAttack,      # prepend a fake "safety off" exchange first
}

# converter name -> PyRIT converter class. Applied in the listed order.
CONVERTERS = {
    "base64": Base64Converter,
    "leetspeak": LeetspeakConverter,
    "rot13": ROT13Converter,
    "character_space": CharacterSpaceConverter,
}


def attack_rounds(strategies: list, converters: list) -> list:
    """The attack variants tried for a case, grouped in rounds.

    Round 1: every strategy as-is. Round 2: every strategy with every converter
    (the same attack written differently). The runner only moves to the next
    round while the agent has resisted every variant so far.
    Returns [[{"strategy", "converters"}, ...], ...].
    """
    for name in strategies:
        if name not in ATTACK_STRATEGIES:
            raise AttackSpecError(f"unknown attack strategy '{name}' (known: {sorted(ATTACK_STRATEGIES)})")
    for name in converters:
        if name not in CONVERTERS:
            raise AttackSpecError(f"unknown converter '{name}' (known: {sorted(CONVERTERS)})")

    rounds = [[{"strategy": s, "converters": []} for s in strategies]]
    if converters:
        rounds.append([{"strategy": s, "converters": [c]} for s in strategies for c in converters])
    return rounds


def variant_name(spec: dict) -> str:
    """e.g. "skeleton_key + leetspeak"."""
    return " + ".join([spec.get("strategy", "prompt_sending"), *spec.get("converters", [])])


class AttackSpecError(AgentCallError):
    """The fixture's attack description is invalid (unknown strategy/converter,
    missing prompt). Subclasses AgentCallError so the runner records it as
    status=error like any other failed capture."""


def _build_attack(spec: dict, target: AgentAdapterTarget):
    strategy_name = spec.get("strategy", "prompt_sending")
    strategy_cls = ATTACK_STRATEGIES.get(strategy_name)
    if strategy_cls is None:
        raise AttackSpecError(f"unknown attack strategy '{strategy_name}' (known: {sorted(ATTACK_STRATEGIES)})")

    converters = []
    for name in spec.get("converters", []):
        converter_cls = CONVERTERS.get(name)
        if converter_cls is None:
            raise AttackSpecError(f"unknown converter '{name}' (known: {sorted(CONVERTERS)})")
        converters.append(converter_cls())

    converter_config = AttackConverterConfig(
        request_converters=ConverterConfiguration.from_converters(converters=converters)
    )
    return strategy_cls(objective_target=target, attack_converter_config=converter_config)


def run_attack(adapter, case_input: dict) -> dict:
    """Run the attack described in case_input["attack"] against the agent.

    Returns {"actual_output", "trace", "attack": {strategy, converters, prompt_sent}}.
    Raises AgentCallError (or AttackSpecError) if no answer could be captured.
    """
    spec = case_input.get("attack") or {}
    objective = case_input.get("user_message")
    if not objective:
        raise AttackSpecError("attack case input has no 'user_message' (the attack objective)")

    shared_input = {k: v for k, v in case_input.items() if k not in ("attack", "user_message")}
    try:
        # PyRIT must be initialized BEFORE any target is created (targets need its memory).
        pyrit_runtime.run(pyrit_runtime.ensure_initialized_async())
        target = AgentAdapterTarget(adapter=adapter, base_input=shared_input)
        attack = _build_attack(spec, target)
    except AttackSpecError:
        raise
    except Exception as e:
        # Never let a setup problem crash the whole run: it's an error for this case (Rule 2).
        raise AgentCallError(f"could not set up the PyRIT attack: {e!r}") from e

    try:
        pyrit_runtime.run(attack.execute_async(objective=objective))
    except Exception as e:
        # The target already recorded an agent failure; anything else is PyRIT itself failing.
        agent_error = target.captures[-1].get("error") if target.captures else None
        raise AgentCallError(agent_error or f"PyRIT attack failed: {e!r}") from e

    if not target.captures:
        raise AgentCallError("PyRIT attack finished without sending anything to the agent")
    capture = target.captures[-1]
    return {
        "actual_output": capture["actual_output"],
        "trace": capture["trace"],
        "attack": {
            "strategy": spec.get("strategy", "prompt_sending"),
            "converters": list(spec.get("converters", [])),
            "prompt_sent": capture["prompt_sent"],
        },
    }
