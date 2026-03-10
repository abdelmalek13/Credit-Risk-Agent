"""Execute LLM-generated Python code and capture outputs."""

import io
import re
import sys
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def extract_code(llm_response: str) -> str:
    """Extract Python code from a markdown code block in the LLM response."""
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        match = re.search(pattern, llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()

    match = re.search(r"```python\s*\n(.*)", llm_response, re.DOTALL)
    if match:
        return re.sub(r"```\s*$", "", match.group(1).strip()).strip()

    text = llm_response.strip()
    text = re.sub(r"^```(?:python)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "len", "list", "map", "max",
    "min", "print", "range", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip", "True", "False", "None",
}


def _make_safe_builtins() -> dict:
    import builtins
    safe = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS if hasattr(builtins, name)}
    safe["__import__"] = _blocked_import
    return safe


def _blocked_import(name, *args, **kwargs):
    raise ImportError(f"Imports are not allowed. '{name}' is already available in the namespace.")


def execute_code(
    code: str,
    df: pd.DataFrame,
    precomputed: dict,
) -> dict:
    """Execute a code string and return figures, tables, text, result, and error."""
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout

    namespace = {
        "__builtins__": _make_safe_builtins(),
        "df": df,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "precomputed": precomputed,
    }

    output = {"figures": [], "dataframes": [], "text": "", "result": None, "error": None}

    try:
        sys.stdout = stdout_capture
        exec(code, namespace)
        sys.stdout = old_stdout

        result = namespace.get("result")
        if result is not None:
            if isinstance(result, go.Figure):
                output["figures"].append(result)
            elif isinstance(result, pd.DataFrame):
                output["dataframes"].append(result)
            elif isinstance(result, pd.Series):
                output["dataframes"].append(result.to_frame())
            else:
                output["result"] = result

        output["text"] = stdout_capture.getvalue()

    except Exception:
        sys.stdout = old_stdout
        output["error"] = traceback.format_exc()
        output["text"] = stdout_capture.getvalue()

    return output


def run_with_retries(
    code: str,
    df: pd.DataFrame,
    precomputed: dict,
    llm_provider,
    system_prompt: str,
    user_question: str,
    max_retries: int = 2,
) -> dict:
    """Execute code, and on failure ask the LLM to fix it (up to max_retries)."""
    result = execute_code(code, df, precomputed)

    attempt = 0
    while result["error"] is not None and attempt < max_retries:
        attempt += 1
        precomputed_hint = (
            f"Available precomputed keys: {list(precomputed.keys())}. "
            f"dataset_info keys: {list(precomputed.get('dataset_info', {}).keys())}. "
            f"Use precomputed['dataset_info']['default_rate'] for the overall default rate."
        )
        fix_prompt = (
            f"The previous code produced an error:\n\n"
            f"```\n{result['error']}\n```\n\n"
            f"Original question: {user_question}\n\n"
            f"{precomputed_hint}\n\n"
            f"IMPORTANT: Output ONLY Python code in a ```python``` block. Do NOT include ```python fences inside the code itself.\n\n"
            f"Please fix the code."
        )
        fixed_response = llm_provider.generate(system_prompt, fix_prompt)
        code = extract_code(fixed_response)

        result = execute_code(code, df, precomputed)
        result["_retry_attempt"] = attempt

    return result
