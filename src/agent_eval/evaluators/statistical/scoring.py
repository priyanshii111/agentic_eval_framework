import json
import re
import time

import requests


def _word_overlap_score(actual_text: str, reference_text: str) -> float:
    def tokenize(text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))

    actual_tokens = tokenize(actual_text)
    reference_tokens = tokenize(reference_text)

    union = actual_tokens | reference_tokens
    if not union:
        return 1.0

    intersection = actual_tokens & reference_tokens
    return len(intersection) / len(union)


def _word_overlap_method(actual_text: str, reference_text: str, config: dict) -> dict:
    return {"score": _word_overlap_score(actual_text, reference_text)}


_JUDGE_PROMPT_TEMPLATE = """You are comparing two answers to see if they mean the same thing.

Reference answer: "{reference}"
Actual answer: "{actual}"

Does the actual answer convey the same meaning as the reference answer, even if worded \
differently? Respond with ONLY a JSON object in this exact format, nothing else:
{{"score": <number between 0.0 and 1.0>, "reason": "<one short sentence>"}}"""


def _call_ollama(prompt: str, config: dict) -> str:
    """Calls the local Ollama server and returns the raw text response.
    Kept isolated from parsing so it can be mocked in tests (Rule 11)."""
    host = config.get("ollama_host", "http://localhost:11434")
    model = config["llm_judge_model"]
    timeout = config.get("llm_judge_timeout_seconds", 30)

    payload = {"model": model, "prompt": prompt, "stream": False}
    # Optional: make the judge as repeatable as possible. Only sent when the
    # config sets them, so configs without these keys behave exactly as before.
    options = {
        ollama_key: config[config_key]
        for config_key, ollama_key in (("judge_temperature", "temperature"), ("judge_seed", "seed"))
        if config_key in config
    }
    if options:
        payload["options"] = options

    response = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"]


def _parse_judge_response(raw_text: str) -> dict:
    """Parse the judge's raw text into {"score": float, "reason": str}.
    Falls back to extracting a {...} substring if the model added extra text
    around the JSON. Raises ValueError if truly unparseable."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in judge response: {raw_text!r}")
        parsed = json.loads(match.group(0))

    if "score" not in parsed:
        raise ValueError(f"judge response missing 'score' field: {raw_text!r}")

    score = float(parsed["score"])
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"judge score out of range [0,1]: {score}")

    return {"score": score, "reason": parsed.get("reason", "")}


def call_judge_with_retries(prompt: str, config: dict) -> dict:
    """Calls the Ollama judge with the given prompt, retrying on failure.
    Returns {"score": float, "reason": str, "raw_response": str, "model": str}.
    Raises RuntimeError if every attempt fails. Shared by any statistical
    checker that needs an LLM judge — not specific to similarity scoring."""
    model = config["llm_judge_model"]
    max_retries = config.get("llm_judge_max_retries", 2)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw_text = _call_ollama(prompt, config)
            parsed = _parse_judge_response(raw_text)
            return {
                "score": parsed["score"],
                "reason": parsed["reason"],
                "raw_response": raw_text,
                "model": model,
            }
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)

    raise RuntimeError(f"llm judge call failed after {max_retries + 1} attempt(s): {last_error}")


def _llm_judge_score(actual_text: str, reference_text: str, config: dict) -> dict:
    prompt = _JUDGE_PROMPT_TEMPLATE.format(reference=reference_text, actual=actual_text)
    return call_judge_with_retries(prompt, config)


SCORING_METHODS = {
    "word_overlap": _word_overlap_method,
    "llm_judge": _llm_judge_score,
}

# If a method errors out (not just scores low — a genuine failure), fall back
# to the other one so a case still gets a real result. Never triggered by a
# legitimate low/failing score, only by an exception (Rule 2).
_FALLBACK_METHOD = {
    "llm_judge": "word_overlap",
    "word_overlap": "llm_judge",
}


def evaluate(actual_output: dict, expected: dict, config: dict | None = None) -> dict:
    config = config or {}
    primary_method_name = expected.get("method", config.get("default_method", "llm_judge"))
    primary_method = SCORING_METHODS.get(primary_method_name)
    if primary_method is None:
        return {
            "status": "error",
            "details": {"reason": f"unknown scoring method '{primary_method_name}'"},
        }

    actual_text = actual_output.get("output_text") or ""
    reference_text = expected["reference_output"]
    threshold = expected.get("threshold", config.get("default_threshold", 0.5))

    primary_error = None
    used_method_name = primary_method_name
    try:
        result = primary_method(actual_text, reference_text, config)
    except Exception as e:
        primary_error = str(e)
        fallback_method_name = _FALLBACK_METHOD.get(primary_method_name)
        fallback_method = SCORING_METHODS.get(fallback_method_name)
        try:
            result = fallback_method(actual_text, reference_text, config)
            used_method_name = fallback_method_name
        except Exception as e2:
            return {
                "status": "error",
                "details": {
                    "reason": (
                        f"scoring method '{primary_method_name}' failed: {primary_error}; "
                        f"fallback '{fallback_method_name}' also failed: {e2}"
                    ),
                },
            }

    score = result["score"]
    status = "pass" if score >= threshold else "fail"

    details = {
        "score": round(score, 4),
        "threshold": threshold,
        "primary_method": primary_method_name,
        "used_method": used_method_name,
    }
    if primary_error is not None:
        details["primary_error"] = primary_error
    if "reason" in result:
        details["judge_reason"] = result["reason"]
    if "model" in result:
        details["judge_model"] = result["model"]
    if "raw_response" in result:
        details["judge_raw_response"] = result["raw_response"]

    return {"status": status, "details": details}
