"""A PyRIT target that attacks our live agent through the existing adapter.

PyRIT's attack strategies (PromptSendingAttack, SkeletonKeyAttack, ...) and
converters (Base64, leetspeak, ...) send their attack text here; we forward it
via the same adapter every other live case uses, so each attempt gets a fresh
database, the case's passenger_id and the AgentCallError -> error handling.

PyRIT only understands text, but our graders (e.g. the jailbreak strategy)
also need the agent's tool calls. So every full adapter response
(actual_output + trace) is kept in `captures`, in send order.

Single-turn only (PyRIT's default target configuration): multi-turn attacks
need a second "attacker" model for many rounds, too slow on a CPU-only setup.
"""
import asyncio

from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import PromptTarget

from .agent import AgentCallError


class AgentAdapterTarget(PromptTarget):
    """Single-turn PyRIT target backed by an integrations.agent adapter."""

    def __init__(self, *, adapter, base_input: dict | None = None) -> None:
        """
        Args:
            adapter: anything with .call(case_input) -> {"actual_output", "trace"}
                (e.g. AirlineAgentAdapter).
            base_input: case input shared by every attempt (e.g. passenger_id);
                the attack text is added as "user_message".
        """
        super().__init__(model_name=getattr(adapter, "name", "agent"))
        self._adapter = adapter
        self._base_input = {k: v for k, v in (base_input or {}).items() if k != "user_message"}
        self.captures: list[dict] = []

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1].get_piece()
        attack_text = request.converted_value  # after PyRIT converters (e.g. Base64)
        capture = {"prompt_original": request.original_value, "prompt_sent": attack_text}

        try:
            # The adapter is synchronous (LangGraph invoke): run it off the event loop.
            response = await asyncio.to_thread(
                self._adapter.call, {**self._base_input, "user_message": attack_text}
            )
        except AgentCallError as e:
            capture["error"] = str(e)  # recorded even if PyRIT swallows the exception
            self.captures.append(capture)
            raise

        capture["actual_output"] = response["actual_output"]
        capture["trace"] = response["trace"]
        self.captures.append(capture)

        # PyRIT's copy of the answer only; `captures` keeps the true (possibly empty) text.
        answer = response["actual_output"]["output_text"] or "(empty response)"
        return [construct_response_from_request(request=request, response_text_pieces=[answer])]
