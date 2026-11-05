from __future__ import annotations

import contextlib
import json
import logging
import re
import socket
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import httpx

from .settings import settings

logger = logging.getLogger("cognicrew.ai_provider")


class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    pass


class AIProviderAuthError(AIProviderError):
    """Authentication or authorization failure with the AI provider."""
    pass


class AIProviderConfigurationError(AIProviderAuthError):
    """AI provider is not initialized or required configuration (API key, endpoint) is missing."""
    pass


class AIProviderUnavailableError(AIProviderError):
    """AI provider is unreachable, timed out, rate-limited, or experiencing downtime."""
    pass


def strip_internal_reasoning(text: str) -> str:
    """Ensure no internal <think> or reasoning tags leak into user-facing content."""
    if not text:
        return ""
    # Strip <think>...</think> blocks
    cleaned = re.sub(r"(?s)<think>.*?</think>", "", text)
    # Also strip [REASONING]...[/REASONING] if present
    cleaned = re.sub(r"(?s)\[REASONING\].*?\[/REASONING\]", "", cleaned)
    return cleaned.strip()


class StreamingReasoningFilter:
    """Filters internal reasoning tags (<think>...</think> and [REASONING]...[/REASONING])
    progressively across stream chunk boundaries while preserving token whitespace.
    """

    def __init__(self) -> None:
        self.in_think = False
        self.in_reasoning = False

    def process(self, chunk: str) -> str:
        if not chunk:
            return ""

        text = chunk
        if self.in_think:
            if "</think>" in text:
                _, _, text = text.partition("</think>")
                self.in_think = False
            else:
                return ""

        if self.in_reasoning:
            if "[/REASONING]" in text:
                _, _, text = text.partition("[/REASONING]")
                self.in_reasoning = False
            else:
                return ""

        if "<think>" in text:
            before, _, after = text.partition("<think>")
            if "</think>" in after:
                _, _, after_think = after.partition("</think>")
                text = before + after_think
            else:
                self.in_think = True
                text = before

        if "[REASONING]" in text:
            before, _, after = text.partition("[REASONING]")
            if "[/REASONING]" in after:
                _, _, after_reasoning = after.partition("[/REASONING]")
                text = before + after_reasoning
            else:
                self.in_reasoning = True
                text = before

        return text


def _resolve_ipv4(host: str) -> str | None:
    """Best-effort resolution of a hostname to a single IPv4 address."""
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if not infos:
            return None
        addr = infos[0][4][0]
        return addr if isinstance(addr, str) else None
    except Exception:
        return None


@contextlib.contextmanager
def _prefer_ipv4_resolution():
    """Make DNS resolution return IPv4 (A) records first while active.

    Some networks advertise IPv6 (including NAT64) routes that never complete a
    TCP handshake. httpx tries each resolved address serially, so a dead AAAA
    record stalls every request for the full connect timeout per address.
    Reordering A records first keeps the real hostname for TLS SNI and the
    HTTP Host header, exactly like ``curl --resolve`` does.
    """
    original = socket.getaddrinfo

    def ipv4_first(*args: Any, **kwargs: Any):
        results = original(*args, **kwargs)
        v4 = [r for r in results if r[0] == socket.AF_INET]
        rest = [r for r in results if r[0] != socket.AF_INET]
        return v4 + rest if v4 and rest else results

    socket.getaddrinfo = ipv4_first
    try:
        yield
    finally:
        socket.getaddrinfo = original


def _post_with_ipv4_fallback(
    endpoint: str, json_payload: Any, headers: dict[str, str], timeout_seconds: float
):
    """POST with IPv4-first DNS when the host also advertises unreachable IPv6.

    Falls back to normal (OS-resolved) behavior when no IPv4 record exists or
    the IPv4 connection itself fails. Strictly enforces that total elapsed time
    across any fallback attempts does not exceed timeout_seconds.
    """
    t_start = time.perf_counter()
    connect_to = min(5.0, timeout_seconds)
    client_timeout = httpx.Timeout(timeout=timeout_seconds, connect=connect_to)

    host = urlparse(endpoint).hostname
    if host and _resolve_ipv4(host):
        try:
            with _prefer_ipv4_resolution():
                with httpx.Client(timeout=client_timeout) as client:
                    return client.post(endpoint, json=json_payload, headers=headers)
        except (httpx.UnsupportedProtocol, httpx.ConnectError, httpx.ConnectTimeout):
            pass  # IPv4 path unavailable -> let normal resolution try

    elapsed = time.perf_counter() - t_start
    remaining = max(1.0, timeout_seconds - elapsed)
    fallback_timeout = httpx.Timeout(timeout=remaining, connect=min(3.0, remaining))
    with httpx.Client(timeout=fallback_timeout) as client:
        return client.post(endpoint, json=json_payload, headers=headers)


@contextlib.contextmanager
def _stream_with_ipv4_fallback(
    endpoint: str, json_payload: Any, headers: dict[str, str], timeout_seconds: float
):
    """Context manager for streaming HTTP POST with IPv4-first DNS."""
    connect_to = min(5.0, timeout_seconds)
    client_timeout = httpx.Timeout(timeout=timeout_seconds, connect=connect_to)

    host = urlparse(endpoint).hostname
    if host and _resolve_ipv4(host):
        try:
            with _prefer_ipv4_resolution():
                with httpx.Client(timeout=client_timeout) as client:
                    with client.stream("POST", endpoint, json=json_payload, headers=headers) as response:
                        yield response
                        return
        except (httpx.UnsupportedProtocol, httpx.ConnectError, httpx.ConnectTimeout):
            pass

    with httpx.Client(timeout=client_timeout) as client:
        with client.stream("POST", endpoint, json=json_payload, headers=headers) as response:
            yield response


class AIProvider(ABC):
    """Abstract interface for all CogniCrew AI reasoning providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Generate a grounded completion given a prompt and optional system instructions."""
        raise NotImplementedError

    def stream_generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> Iterator[str]:
        """Progressively yield generated text tokens/chunks from provider."""
        raise NotImplementedError


class NVIDIAProvider(AIProvider):
    """NVIDIA Build inference provider (OpenAI-compatible endpoints)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else settings.nvidia_api_key).strip()
        self.model = (model if model is not None else settings.ai_model).strip()
        self.base_url = (base_url if base_url is not None else settings.ai_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.ai_timeout
        self.max_tokens = max_tokens if max_tokens is not None else settings.ai_max_tokens
        self.enable_thinking = enable_thinking if enable_thinking is not None else settings.ai_enable_thinking

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        if not self.api_key:
            raise AIProviderConfigurationError("PROVIDER_NOT_INITIALIZED: NVIDIA_API_KEY is not configured.")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.7),
        }

        # Thinking configuration: only inject for models that explicitly support thinking/reasoning templates
        supports_thinking = any(k in self.model.lower() for k in ("nemotron-3-ultra", "reasoning", "deepseek-r1"))
        if supports_thinking:
            payload["chat_template_kwargs"] = {"thinking": bool(self.enable_thinking)}

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        stage_start = time.perf_counter()
        logger.info(
            "STAGE6 nvidia call: model=%s max_tokens=%d timeout=%.1fs thinking=%s prompt_chars=%d",
            self.model, payload["max_tokens"], self.timeout, supports_thinking and self.enable_thinking, len(prompt),
        )
        try:
            response = _post_with_ipv4_fallback(endpoint, payload, headers, self.timeout)
            logger.info("STAGE6 nvidia HTTP done: status=%d elapsed=%.3fs", response.status_code, time.perf_counter() - stage_start)
        except httpx.TimeoutException as exc:
            logger.warning("NVIDIA API request timed out after %.1f seconds on model %s", self.timeout, self.model)
            raise AIProviderUnavailableError(f"NVIDIA AI provider timed out on model {self.model}.") from None
        except httpx.RequestError as exc:
            logger.warning("NVIDIA API connection failed: %s", type(exc).__name__)
            raise AIProviderUnavailableError("Could not connect to NVIDIA AI provider.") from None

        if response.status_code in (401, 403):
            logger.warning("NVIDIA API returned authentication error HTTP %d", response.status_code)
            raise AIProviderAuthError("NVIDIA AI authentication failed.")

        if response.status_code == 429:
            logger.warning("NVIDIA API rate limit exceeded HTTP 429")
            raise AIProviderUnavailableError("NVIDIA AI rate limit exceeded.")

        if response.status_code == 404:
            logger.warning("NVIDIA model %s not found (HTTP 404)", self.model)
            raise AIProviderError(f"Model {self.model} not found.")

        if response.status_code >= 500:
            err_msg = ""
            try:
                err_data = response.json()
                if isinstance(err_data, dict):
                    err_msg = str(err_data.get("error", {}).get("message", "") or "")
            except Exception:
                err_msg = response.text[:200]
            logger.warning("NVIDIA API server error HTTP %d on model %s: %s", response.status_code, self.model, err_msg)
            err_lower = err_msg.lower()
            # Model-specific overload/exhaustion -> AIProviderError so router can cooldown this candidate and try fallback
            if any(term in err_lower for term in ("overload", "resourceexhausted", "request limit", "capacity", "concurrency")):
                raise AIProviderError(f"Model {self.model} is temporarily unavailable/exhausted (HTTP {response.status_code}): {err_msg}")
            # Provider-wide outage
            raise AIProviderUnavailableError("NVIDIA AI provider is temporarily unavailable.")

        if response.status_code != 200:
            logger.warning("NVIDIA API error HTTP %d", response.status_code)
            raise AIProviderError(f"NVIDIA AI request failed with HTTP {response.status_code}.")

        try:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise AIProviderError("NVIDIA API returned empty choices.")

            message = choices[0].get("message", {})
            # Explicitly extract user-visible content ONLY. Ignore reasoning_content.
            raw_content = message.get("content") or ""
            reasoning_present = bool(message.get("reasoning_content"))
            final_content = strip_internal_reasoning(raw_content)
            logger.info(
                "STAGE6 nvidia response parsed: content_chars=%d reasoning_content_present=%s finish_reason=%s",
                len(final_content), reasoning_present, choices[0].get("finish_reason"),
            )
            if reasoning_present:
                logger.info("STAGE6 nvidia reasoning_content PREVIEW (not shown to user): %s", str(message.get("reasoning_content"))[:300])
            if not final_content:
                # If content was empty but reasoning_content existed, never leak reasoning_content
                raise AIProviderError("NVIDIA API returned no visible content.")
            return final_content

        except (ValueError, KeyError) as exc:
            logger.warning("Malformed response from NVIDIA API: %s", type(exc).__name__)
            raise AIProviderError("Malformed response received from NVIDIA AI provider.") from None

    def stream_generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> Iterator[str]:
        if not self.api_key:
            raise AIProviderConfigurationError("PROVIDER_NOT_INITIALIZED: NVIDIA_API_KEY is not configured.")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.7),
            "stream": True,
        }

        supports_thinking = any(k in self.model.lower() for k in ("nemotron-3-ultra", "reasoning", "deepseek-r1"))
        if supports_thinking:
            payload["chat_template_kwargs"] = {"thinking": bool(self.enable_thinking)}

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        filter_obj = StreamingReasoningFilter()
        try:
            with _stream_with_ipv4_fallback(endpoint, payload, headers, self.timeout) as response:
                if response.status_code in (401, 403):
                    logger.warning("NVIDIA API returned authentication error HTTP %d", response.status_code)
                    raise AIProviderAuthError("NVIDIA AI authentication failed.")
                if response.status_code == 429:
                    logger.warning("NVIDIA API rate limit exceeded HTTP 429")
                    raise AIProviderUnavailableError("NVIDIA AI rate limit exceeded.")
                if response.status_code == 404:
                    logger.warning("NVIDIA model %s not found (HTTP 404)", self.model)
                    raise AIProviderError(f"Model {self.model} not found.")
                if response.status_code >= 500:
                    raise AIProviderUnavailableError(f"NVIDIA AI provider server error HTTP {response.status_code}.")
                if response.status_code != 200:
                    raise AIProviderError(f"NVIDIA AI request failed with HTTP {response.status_code}.")

                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line_data = line[6:].strip()
                    elif line.startswith("data:"):
                        line_data = line[5:].strip()
                    else:
                        continue
                    if line_data == "[DONE]":
                        break
                    try:
                        chunk_obj = json.loads(line_data)
                        choices = chunk_obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                cleaned = filter_obj.process(content)
                                if cleaned:
                                    yield cleaned
                    except Exception:
                        continue
        except httpx.TimeoutException:
            logger.warning("NVIDIA API stream timed out on model %s", self.model)
            raise AIProviderUnavailableError(f"NVIDIA AI provider timed out on model {self.model}.") from None
        except httpx.RequestError as exc:
            logger.warning("NVIDIA API stream connection failed: %s", type(exc).__name__)
            raise AIProviderUnavailableError("Could not connect to NVIDIA AI provider.") from None


class GeminiProvider(AIProvider):
    """Google Gemini inference provider using the official REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else settings.gemini_api_key).strip()
        self.model = (model if model is not None else settings.gemini_model).strip()
        # Strip any "models/" prefix if passed
        if self.model.startswith("models/"):
            self.model = self.model[len("models/"):]
        self.base_url = (base_url if base_url is not None else settings.gemini_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.ai_timeout
        self.max_tokens = max_tokens if max_tokens is not None else settings.ai_max_tokens

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        if not self.api_key:
            raise AIProviderConfigurationError("PROVIDER_NOT_INITIALIZED: GEMINI_API_KEY is not configured.")

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.2),
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
                "topP": kwargs.get("top_p", 0.8),
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        model_candidates = [self.model]
        for fallback_m in ("gemini-3.1-flash-lite", "gemini-3.8-flash"):
            if fallback_m not in model_candidates:
                model_candidates.append(fallback_m)

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        stage_start = time.perf_counter()
        response: httpx.Response | None = None
        last_status = 0

        for candidate_model in model_candidates:
            now_loop = time.perf_counter()
            rem_total = self.timeout - (now_loop - stage_start)
            if rem_total < 2.0:
                break

            endpoint = f"{self.base_url}/models/{candidate_model}:generateContent"
            logger.info(
                "STAGE6 gemini call: model=%s max_tokens=%d timeout=%.1fs prompt_chars=%d",
                candidate_model, payload["generationConfig"]["maxOutputTokens"], rem_total, len(prompt),
            )
            try:
                resp = _post_with_ipv4_fallback(endpoint, payload, headers, rem_total)
                logger.info("STAGE6 gemini HTTP done: model=%s status=%d elapsed=%.3fs", candidate_model, resp.status_code, time.perf_counter() - now_loop)
            except httpx.TimeoutException:
                logger.warning("Gemini API request timed out on model %s", candidate_model)
                continue
            except httpx.RequestError as exc:
                logger.warning("Gemini API connection failed on %s: %s", candidate_model, type(exc).__name__)
                continue

            if resp.status_code == 200:
                response = resp
                self.model = candidate_model
                break
            elif resp.status_code in (401, 403):
                logger.warning("Gemini API returned authentication error HTTP %d", resp.status_code)
                raise AIProviderAuthError("Gemini AI authentication failed.")
            elif resp.status_code in (429, 404, 503):
                last_status = resp.status_code
                logger.warning("Gemini model %s returned HTTP %d; trying next Gemini candidate", candidate_model, resp.status_code)
                time.sleep(0.5)
                continue
            else:
                last_status = resp.status_code
                continue

        if response is None or response.status_code != 200:
            if last_status == 429:
                raise AIProviderUnavailableError("Gemini AI rate limit exceeded.")
            raise AIProviderUnavailableError(f"Gemini AI provider unavailable or latency budget exhausted (HTTP {last_status}).")

        try:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise AIProviderError("Gemini API returned empty candidates.")

            parts = candidates[0].get("content", {}).get("parts", [])
            raw_content = "".join(p.get("text", "") for p in parts)
            final_content = strip_internal_reasoning(raw_content)
            logger.info(
                "STAGE6 gemini response parsed: content_chars=%d finish_reason=%s",
                len(final_content), candidates[0].get("finishReason"),
            )
            if not final_content:
                raise AIProviderError("Gemini API returned no visible content.")
            return final_content

        except (ValueError, KeyError) as exc:
            logger.warning("Malformed response from Gemini API: %s", type(exc).__name__)
            raise AIProviderError("Malformed response received from Gemini AI provider.") from None

    def stream_generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> Iterator[str]:
        if not self.api_key:
            raise AIProviderConfigurationError("PROVIDER_NOT_INITIALIZED: GEMINI_API_KEY is not configured.")

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.2),
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens),
                "topP": kwargs.get("top_p", 0.8),
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        filter_obj = StreamingReasoningFilter()
        endpoint = f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse"
        try:
            with _stream_with_ipv4_fallback(endpoint, payload, headers, self.timeout) as response:
                if response.status_code in (401, 403):
                    raise AIProviderAuthError("Gemini AI authentication failed.")
                if response.status_code == 429:
                    raise AIProviderUnavailableError("Gemini AI rate limit exceeded.")
                if response.status_code != 200:
                    raise AIProviderUnavailableError(f"Gemini API error HTTP {response.status_code}.")

                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line_data = line[6:].strip()
                    elif line.startswith("data:"):
                        line_data = line[5:].strip()
                    else:
                        continue
                    try:
                        chunk_obj = json.loads(line_data)
                        candidates = chunk_obj.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                text = p.get("text", "")
                                if text:
                                    cleaned = filter_obj.process(text)
                                    if cleaned:
                                        yield cleaned
                    except Exception:
                        continue
        except httpx.TimeoutException:
            raise AIProviderUnavailableError("Gemini AI provider timed out.") from None
        except httpx.RequestError:
            raise AIProviderUnavailableError("Could not connect to Gemini AI provider.") from None


class MockAIProvider(AIProvider):
    """Deterministic mock provider for automated tests with zero external dependencies."""

    def __init__(
        self,
        default_answer: str | None = None,
        simulate_error: str | None = None,
        handler: Callable[[str, str | None], str] | None = None,
    ) -> None:
        self.default_answer = default_answer
        self.simulate_error = simulate_error
        self.handler = handler
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None
        # Optional per-test token generator override (see stream_generate).
        # Declared here so the attribute is part of the class contract; test
        # overrides use (prompt, system_prompt=None, **kwargs) signatures.
        self.stream_generator_override: Callable[..., Iterator[str]] | None = None

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        if self.simulate_error == "auth":
            raise AIProviderAuthError("Simulated AI authentication failure.")
        elif self.simulate_error == "timeout":
            raise AIProviderUnavailableError("Simulated AI timeout.")
        elif self.simulate_error == "unavailable":
            raise AIProviderUnavailableError("Simulated AI provider unavailable.")
        elif self.simulate_error == "generic":
            raise AIProviderError("Simulated generic AI error.")

        if self.handler:
            return self.handler(prompt, system_prompt)

        # Structured mocks for quiz and flashcard generation
        sp_lower = (system_prompt or "").lower()
        p_lower = prompt.lower()
        if "quiz" in p_lower or "quiz" in sp_lower or "question objects" in sp_lower:
            return (
                '[\n'
                '  {\n'
                '    "question": "What does double buffering mean in computer graphics?",\n'
                '    "options": [\n'
                '      "Use of an offscreen image to reduce flicker",\n'
                '      "Doubling screen resolution",\n'
                '      "Running two GPUs in parallel",\n'
                '      "Buffering network packets twice"\n'
                '    ],\n'
                '    "correct_answer": "Use of an offscreen image to reduce flicker",\n'
                '    "correct_index": 0,\n'
                '    "explanation": "Double buffering utilizes an offscreen buffer where pixels are prepared before display.",\n'
                '    "source_page": 833\n'
                '  }\n'
                ']'
            )
        if "flashcard" in p_lower or "flashcard" in sp_lower:
            return (
                '[\n'
                '  {\n'
                '    "front": "What is Double Buffering?",\n'
                '    "back": "Double buffering uses an offscreen image to prepare pixels before display to reduce flicker.",\n'
                '    "source_page": 833\n'
                '  }\n'
                ']'
            )

        # Heuristic deterministic mock response reflecting grounded generation
        if "double buffering" in prompt.lower():
            return (
                "## Definition\n\n"
                "Double buffering is a technique that uses an offscreen image to reduce flicker during display rendering.\n\n"
                "## How it works\n\n"
                "The screen acts as the primary buffer for visible pixels, while an offscreen image serves as a secondary buffer "
                "where pixels can be prepared before being displayed seamlessly."
            )
        elif "slip" in prompt.lower() or "question" in prompt.lower():
            return (
                "## Explanation\n\n"
                "Based on the verified document excerpt, the requested question details the specified assignment tasks and requirements."
            )
        else:
            return "## Summary\n\nSynthesized explanation based strictly on the supplied verified evidence."

    def stream_generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> Iterator[str]:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        if self.simulate_error == "auth":
            raise AIProviderAuthError("Simulated AI authentication failure.")
        elif self.simulate_error == "timeout":
            raise AIProviderUnavailableError("Simulated AI timeout.")
        elif self.simulate_error == "unavailable":
            raise AIProviderUnavailableError("Simulated AI provider unavailable.")
        elif self.simulate_error == "generic":
            raise AIProviderError("Simulated generic AI error.")

        if self.stream_generator_override is not None:
            yield from self.stream_generator_override(prompt, system_prompt, **kwargs)
            return

        # True progressive token generator: yields tokens directly from generation sequence without post-splitting
        yield from self._stream_tokens(prompt, system_prompt, **kwargs)

    def _stream_tokens(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> Iterator[str]:
        """Progressively yield generated tokens directly from the generation pipeline without post-splitting."""
        p_lower = prompt.lower()
        if "summary" in p_lower or "profile request" in p_lower:
            yield "## Executive Summary\n\n"
            yield "Synthesized explanation based strictly on the supplied verified evidence."
        elif "explain" in p_lower or "how" in p_lower or "what" in p_lower:
            yield "## Explanation\n\n"
            yield "Based on the verified document excerpt, "
            yield "the requested question details the specified assignment tasks "
            yield "and requirements."
        elif "solution" in p_lower or "code" in p_lower:
            yield "### Problem from document\n\n"
            yield "Grounded programming specification identified in evidence.\n\n"
            yield "### Derived solution\n\n"
            yield "```python\n# Verified solution\ndef solve():\n    return 42\n```"
        elif "comparison" in p_lower or "table" in p_lower:
            yield "| Parameter | Specification |\n|---|---|\n| Architecture | Distributed |\n| Verification | Passed |\n"
        else:
            yield "## Research Briefing\n\n"
            yield "Verified findings synthesized directly from the provided source evidence."


_active_mock_provider: MockAIProvider | None = None


def set_active_mock_provider(mock: MockAIProvider | None) -> None:
    """Utility to inject a specific mock provider during tests."""
    global _active_mock_provider
    _active_mock_provider = mock


def get_ai_provider() -> AIProvider:
    """Provider factory respecting configuration and active mocks."""
    provider_name = settings.ai_provider
    if provider_name == "mock":
        global _active_mock_provider
        if _active_mock_provider is None:
            _active_mock_provider = MockAIProvider()
        return _active_mock_provider
    elif provider_name == "nvidia":
        return NVIDIAProvider()
    elif provider_name == "gemini":
        return GeminiProvider()
    else:
        raise AIProviderError(f"Unsupported AI provider: {provider_name}")
