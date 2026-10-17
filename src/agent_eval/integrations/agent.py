"""Adapters that call a real, live agent and return its reply in the AgentResponse
shape ({output_text, output_json, tool_calls}) every evaluator already understands.

The evaluators never care where actual_output came from. This file is the only
place in the framework that knows how to talk to a live agent; adding another
agent means one new adapter class + one ADAPTERS entry .
"""
import json
import os
import shutil
import tempfile
import time
import uuid


class AgentCallError(Exception):
    """The live agent could not produce a response at all.

    This is "our tooling / the agent run broke", never "the agent answered
    wrongly" — the runner must record it as status=error, not fail (Rule 2).
    """


def _parse_json_or_none(text) -> dict | list | None:
    """The agent's final text parsed as JSON, or None if it isn't JSON (same
    meaning as output_json=null in the static fixtures)."""
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _serialize_message(message) -> dict:
    """One conversation message as plain JSON, for the trace."""
    entry = {"role": message.type, "content": message.content}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = [{"tool": tc["name"], "args": tc["args"]} for tc in tool_calls]
    if message.type == "tool":
        entry["tool"] = message.name
        entry["status"] = getattr(message, "status", None)
    return entry


class AirlineAgentAdapter:
    """Calls the LangGraph airline support agent (agents/airline_support)."""

    name = "airline_support"

    def __init__(self, config: dict):
        # Imported here, not at the top of the file, so static mode never loads
        # LangGraph / Ollama libraries at all.
        from agents.airline_support.graph import build_agent, build_retriever

        self._config = config
        self._build_agent = build_agent
        try:
            # Slow (~1 min on CPU) — built once per adapter, i.e. once per run.
            self._retriever = build_retriever(config)
        except Exception as e:
            raise AgentCallError(f"could not build the policy retriever: {e!r}") from e

    def call(self, case_input: dict) -> dict:
        """Ask the agent one question on a fresh copy of the database.

        Returns {"actual_output": AgentResponse, "trace": {...}}. The trace
        (every message incl. tool results, timing, model) is for debugging and
        later groundedness checks; evaluators only ever see actual_output.
        """
        user_message = case_input.get("user_message")
        if not user_message:
            raise AgentCallError("case input has no 'user_message'")

        agent_cfg = self._config["agent"]
        passenger_id = case_input.get("passenger_id", agent_cfg["default_passenger_id"])
        run_config = {
            "configurable": {"passenger_id": passenger_id, "thread_id": str(uuid.uuid4())},
            "recursion_limit": agent_cfg.get("recursion_limit", 25),
        }

        work_dir = tempfile.mkdtemp(prefix="airline_case_")
        start = time.perf_counter()
        try:
            db_path = os.path.join(work_dir, "case.sqlite")
            shutil.copy(self._config["data"]["db_backup"], db_path)  # fresh DB per case (Rule 1)
            agent = self._build_agent(self._config, db_path, self._retriever)
            result = agent.invoke({"messages": [("user", user_message)]}, run_config)
        except Exception as e:
            raise AgentCallError(f"airline agent failed: {e!r}") from e
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        messages = result["messages"]
        final_content = messages[-1].content
        output_text = final_content if isinstance(final_content, str) else json.dumps(final_content)
        tool_calls = [
            {"tool": tc["name"], "args": tc["args"]}
            for m in messages
            for tc in (getattr(m, "tool_calls", None) or [])
        ]

        return {
            "actual_output": {
                "output_text": output_text,
                "output_json": _parse_json_or_none(output_text),
                "tool_calls": tool_calls,
            },
            "trace": {
                "adapter": self.name,
                "model": agent_cfg["model"],
                "passenger_id": passenger_id,
                "duration_ms": duration_ms,
                "messages": [_serialize_message(m) for m in messages],
            },
        }


ADAPTERS = {
    "airline_support": AirlineAgentAdapter,
}


def create_adapter(config: dict):
    """Build the adapter named by config["agent"]["adapter"]."""
    name = config["agent"].get("adapter")
    adapter_cls = ADAPTERS.get(name)
    if adapter_cls is None:
        raise AgentCallError(f"unknown agent adapter '{name}' (known: {sorted(ADAPTERS)})")
    return adapter_cls(config)
