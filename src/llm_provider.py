from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOCAL_MODELS = {
    "7b": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "label": "Qwen2.5-Coder 7B",
        "size_label": "~4.7 GB",
    },
    "32b": {
        "repo_id": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        "label": "Qwen2.5-Coder 32B",
        "size_label": "~20 GB",
    },
}


class LLMProvider(ABC):
    """Base interface for all LLM backends."""

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt and return the model's text response."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""


class GeminiProvider(LLMProvider):
    """Google Gemini via the google-generativeai SDK."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Create a .env file from .env.example and add your key."
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        self._model_name = model_name

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = self._model.generate_content(
            [
                {"role": "user", "parts": [f"{system_prompt}\n\n{user_message}"]},
            ]
        )
        return response.text

    def name(self) -> str:
        return f"Google Gemini ({self._model_name})"


class LocalModelProvider(LLMProvider):
    """Local Qwen2.5-Coder via llama-cpp-python (auto-downloads on first use)."""

    _loaded: dict[str, object] = {}

    def __init__(self, model_key: str = "7b"):
        cfg = LOCAL_MODELS.get(model_key)
        if cfg is None:
            raise ValueError(f"Unknown local model key '{model_key}'. Choose from: {list(LOCAL_MODELS)}")
        self._model_key = model_key
        self._repo_id = cfg["repo_id"]
        self._filename = cfg["filename"]
        self._label = cfg["label"]

    @staticmethod
    def is_model_downloaded(model_key: str = "7b") -> bool:
        """Check whether the GGUF file is already in the HuggingFace cache."""
        cfg = LOCAL_MODELS.get(model_key, {})
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(cfg["repo_id"], cfg["filename"])
            return isinstance(result, str) and Path(result).exists()
        except Exception:
            return False

    def _download(self) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=self._repo_id, filename=self._filename)

    def _get_or_load(self):
        if self._model_key in LocalModelProvider._loaded:
            return LocalModelProvider._loaded[self._model_key]
        from llama_cpp import Llama
        model_path = self._download()
        llm = Llama(
            model_path=model_path,
            n_ctx=32768,
            n_threads=max(os.cpu_count() or 4, 4),
            verbose=False,
        )
        LocalModelProvider._loaded[self._model_key] = llm
        return llm

    def is_loaded(self) -> bool:
        return self._model_key in LocalModelProvider._loaded

    def ensure_ready(self):
        """Download (if needed) and load the model. Call to pre-warm."""
        self._get_or_load()

    def generate(self, system_prompt: str, user_message: str) -> str:
        llm = self._get_or_load()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        return response["choices"][0]["message"]["content"]

    def name(self) -> str:
        return f"{self._label} (Local)"


def get_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Factory function to get the requested LLM provider."""
    providers = {
        "gemini": GeminiProvider,
        "local": LocalModelProvider,
    }
    cls = providers.get(provider_name.lower())
    if cls is None:
        raise ValueError(f"Unknown provider '{provider_name}'. Choose from: {list(providers)}")
    return cls(**kwargs)
