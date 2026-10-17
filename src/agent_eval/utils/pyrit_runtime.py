"""One PyRIT runtime per process, shared by every PyRIT user in the framework
(the `quality` judge and the attack delivery).

- PyRIT is initialized once (in-memory database, no default initializers).
- All PyRIT coroutines run on ONE event loop kept for the whole process:
  asyncio.run() closes its loop after every call, but PyRIT's HTTP clients
  clean up later — on the closed loop — and flood the output with
  "Event loop is closed" errors.

PyRIT itself is imported lazily, so importing this module never requires it.
"""
import asyncio

_loop: asyncio.AbstractEventLoop | None = None
_initialized = False


async def ensure_initialized_async() -> None:
    global _initialized
    if not _initialized:
        from pyrit.setup import initialize_pyrit_async

        await initialize_pyrit_async("InMemory", load_defaults=False, silent=True)
        _initialized = True


def ollama_judge_target(config: dict):
    """A PyRIT chat target for the LLM judge, pointing at Ollama's OpenAI-style API.

    Reads llm_judge_model, ollama_host, llm_judge_timeout_seconds and the
    optional judge_temperature / judge_seed from a layer config.
    """
    from pyrit.prompt_target import OpenAIChatTarget

    return OpenAIChatTarget(
        endpoint=f"{config.get('ollama_host', 'http://localhost:11434')}/v1",
        api_key="ollama",  # Ollama ignores it; the OpenAI client only needs a non-empty value
        model_name=config["llm_judge_model"],
        temperature=config.get("judge_temperature"),
        seed=config.get("judge_seed"),
        httpx_client_kwargs={"timeout": config.get("llm_judge_timeout_seconds", 30)},
    )


def run(coro):
    """Run a coroutine on the shared event loop and return its result."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)
