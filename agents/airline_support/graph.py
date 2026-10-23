"""The airline support agent: Qwen (via Ollama) + the 17 tools, wired as a LangGraph loop.

Ported from the LangGraph "Build a Customer Support Bot" :
assistant -> tools -> assistant ... until the model answers without a tool call.

- The prompt's "Current time" is the frozen "now", not datetime.now.
- The empty-response re-prompt loop is capped (tutorial: `while True`), so a
  small local model can never freeze an evaluation run.
- Tool errors use ToolNode's built-in handle_tool_errors (LangGraph 1.x) with
  the source's exact message, instead of a custom fallback node.
"""
from datetime import datetime
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from .policy import PolicyRetriever
from .tools import build_tools

MAX_EMPTY_RESPONSE_RETRIES = 3


class EmptyResponseError(RuntimeError):
    """The model kept returning empty answers even after being re-prompted."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable, max_retries: int = MAX_EMPTY_RESPONSE_RETRIES):
        self.runnable = runnable
        self.max_retries = max_retries

    def __call__(self, state: State, config: RunnableConfig):
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        state = {**state, "user_info": passenger_id}

        for _ in range(self.max_retries + 1):
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                return {"messages": result}

        raise EmptyResponseError(
            f"model returned an empty response {self.max_retries + 1} time(s) in a row"
        )


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
)


def _tool_error_message(error: Exception) -> str:
    return f"Error: {repr(error)}\n please fix your mistakes."


def build_retriever(config: dict) -> PolicyRetriever:
    """Build the policy search engine: ~1 min on CPU the first time, near-instant
    afterwards when `policy_lookup.embeddings_cache` is set. Call once per run."""
    return PolicyRetriever.from_faq(
        config["data"]["policy_faq"],
        config["policy_lookup"]["embedding_model"],
        config["agent"]["ollama_host"],
        cache_path=config["policy_lookup"].get("embeddings_cache"),
    )


def build_agent(config: dict, db_path: str, retriever: PolicyRetriever):
    """Build a ready-to-use agent whose tools read and write `db_path`. Fast: call once per case."""
    agent_cfg = config["agent"]
    frozen_now = datetime.fromisoformat(agent_cfg["frozen_now"])

    llm = ChatOllama(
        model=agent_cfg["model"],
        base_url=agent_cfg["ollama_host"],
        temperature=agent_cfg["temperature"],
        seed=agent_cfg["seed"],
        num_ctx=agent_cfg["num_ctx"],
    )
    tools = build_tools(db_path, frozen_now, retriever, config["policy_lookup"]["top_k"])
    assistant_runnable = primary_assistant_prompt.partial(time=str(frozen_now)) | llm.bind_tools(tools)

    builder = StateGraph(State)
    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=_tool_error_message))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=InMemorySaver())
