"""Provider-agnostic LLM wrapper for TrustYojana.

Supports OpenAI, Google Gemini, Anthropic, and Ollama.
Configured via config.yaml and .env with temperature=0.
Provides LLM instantiation, connectivity checks, and structured invocation.
"""

import os
import time
import logging
from typing import Any, Dict, Optional, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_config

logger = logging.getLogger(__name__)

# Cache for health check deep connectivity test: (timestamp, reachable_status)
_connectivity_cache: Optional[Tuple[float, bool]] = None
CACHE_TTL_SECONDS = 60


def check_llm_config() -> Tuple[bool, str]:
    """Check if LLM provider, model, and required environment variables/keys are set.
    
    Returns (is_configured, reason).
    """
    config = get_config()
    provider = config.llm.provider.lower()
    model = config.llm.model

    if not provider:
        return False, "LLM provider is not configured."
    if not model:
        return False, "LLM model name is not configured."

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key or key.strip() == "" or "your_openai_api_key" in key:
            return False, "OPENAI_API_KEY environment variable is not set."
        return True, "OpenAI configuration is valid."

    elif provider in ("gemini", "google"):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key or key.strip() == "" or "your_gemini_api_key" in key:
            return False, "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set."
        return True, "Gemini configuration is valid."

    elif provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key or key.strip() == "" or "your_anthropic_api_key" in key:
            return False, "ANTHROPIC_API_KEY environment variable is not set."
        return True, "Anthropic configuration is valid."

    elif provider == "ollama":
        # Ollama typically does not require an API key, just a reachable base_url
        return True, "Ollama configuration is valid."

    return False, f"Unsupported LLM provider: {provider}"


def get_llm(custom_temperature: Optional[float] = None) -> BaseChatModel:
    """Instantiate and return the configured LangChain ChatModel with temperature 0."""
    config = get_config()
    provider = config.llm.provider.lower()
    model_name = config.llm.model
    temp = custom_temperature if custom_temperature is not None else config.llm.temperature
    timeout = config.llm.request_timeout

    logger.debug(f"Initializing LLM wrapper: provider={provider}, model={model_name}, temp={temp}")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=config.llm.max_retries,
        )

    elif provider in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            google_api_key=api_key,
            timeout=timeout,
            max_retries=config.llm.max_retries,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=model_name,
            temperature=temp,
            api_key=api_key,
            timeout=timeout,
            max_retries=config.llm.max_retries,
        )

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model_name,
                temperature=temp,
                base_url=base_url,
            )
        except Exception:
            # Fallback to OpenAI-compatible endpoint of Ollama
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temp,
                base_url=f"{base_url.rstrip('/')}/v1",
                api_key="ollama",
                timeout=timeout,
            )

    raise ValueError(f"Unsupported LLM provider: {provider}. Options: openai, gemini, anthropic, ollama")


def check_llm_reachability(force: bool = False) -> Tuple[bool, Optional[str]]:
    """Test LLM reachability using a minimal prompt with 60s cache."""
    global _connectivity_cache
    now = time.time()

    if not force and _connectivity_cache is not None:
        cached_time, cached_status = _connectivity_cache
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_status, "Cached result"

    is_configured, reason = check_llm_config()
    if not is_configured:
        _connectivity_cache = (now, False)
        return False, reason

    try:
        llm = get_llm()
        # Minimal probe
        response = llm.invoke([HumanMessage(content="ping")])
        status = response is not None and bool(response.content)
        _connectivity_cache = (now, status)
        return status, None
    except Exception as e:
        logger.warning(f"LLM reachability check failed: {e}")
        _connectivity_cache = (now, False)
        return False, str(e)
