"""Model Router for CogniCrew (Task 8).

Features:
1. Fast Capability-Aware Routing using verified pools from Task 7.
2. Small production shortlist per request (1 Primary + up to 2 Fallbacks, max 3 attempts).
3. Dynamic promotion of healthy reserve models when active candidates enter cooldown.
4. Per-model health tracking & Circuit Breaker (failure threshold + cooldown).
5. Strict total latency budget enforcement (ROUTING_TOTAL_BUDGET_MS).
6. Deterministic test hooks for forced failure verification.
7. Comprehensive structured routing logs.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from .ai_provider import (
    AIProviderAuthError,
    AIProviderConfigurationError,
    AIProviderError,
    AIProviderUnavailableError,
    GeminiProvider,
    NVIDIAProvider,
)
from .settings import settings

logger = logging.getLogger("cognicrew.router")

ROUTING_TABLE_PATH = Path("scripts/task8_routing_table.json")


class TaskType(str, Enum):
    DOCUMENT_QA = "DOCUMENT_QA"
    EXPLANATION = "EXPLANATION"
    MATH_NARRATION = "MATH_NARRATION"
    MARKDOWN_TABLE = "MARKDOWN_TABLE"
    TOPIC_SYNTHESIS = "TOPIC_SYNTHESIS"


@dataclass
class ModelHealth:
    model_id: str
    success_count: int = 0
    consecutive_failures: int = 0
    timeout_count: int = 0
    recent_latencies: list[float] = field(default_factory=list)
    last_success: float | None = None
    last_failure: float | None = None
    cooldown_until: float | None = None

    def is_healthy(self, now: float, failure_threshold: int) -> bool:
        if self.cooldown_until is not None:
            if now < self.cooldown_until:
                return False
            # Cooldown period elapsed -> allow probing/retry
            return True
        return self.consecutive_failures < failure_threshold


@dataclass
class RoutingAttemptRecord:
    task_type: TaskType
    attempt_number: int
    model_id: str
    timeout_allocated: float
    actual_latency: float
    status: str  # "SUCCESS", "FAIL", "TIMEOUT", "FORCED_FAILURE"
    error_detail: str | None = None
    remaining_budget: float = 0.0


@dataclass
class RoutingResult:
    content: str
    answering_model: str
    attempt_count: int
    total_elapsed: float
    attempts: list[RoutingAttemptRecord]
    answering_model_final: str | None = None
    gemini_fallback_ms: float = 0.0
    timeout_reason: str | None = None
    failure_classification: str | None = None

    def __post_init__(self) -> None:
        if self.answering_model_final is None:
            self.answering_model_final = self.answering_model


class ModelRouter:
    """Manages resilient multi-model routing across verified candidate pools."""

    def __init__(
        self,
        routing_table_path: Path | None = None,
        total_budget_ms: float = 28000.0,
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
        max_attempts: int = 3,
        min_attempt_budget_s: float = 2.0,
        default_attempt_timeout_s: float = 16.0,
    ) -> None:
        self.routing_table_path = routing_table_path or ROUTING_TABLE_PATH
        self.total_budget_ms = total_budget_ms
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_attempts = max_attempts
        self.min_attempt_budget_s = min_attempt_budget_s
        self.default_attempt_timeout_s = default_attempt_timeout_s

        self._health: dict[str, ModelHealth] = {}
        self._forced_failures: dict[str, str] = {}  # model_id -> "error" | "timeout" | "generic"
        self._task_pools: dict[str, list[str]] = {}
        self._load_routing_table()

    def _load_routing_table(self) -> None:
        """Load verified pools and policy from Task 8 routing table, or initialize defaults."""
        if self.routing_table_path.exists():
            try:
                with open(self.routing_table_path, "r") as f:
                    data = json.load(f)
                tasks = data.get("tasks", {})
                for tname, tinfo in tasks.items():
                    pool = tinfo.get("verified_pool", [])
                    if pool:
                        self._task_pools[tname] = list(pool)
                policy = data.get("policy", {})
                if policy:
                    if "routing_total_budget_ms" in policy:
                        self.total_budget_ms = float(policy["routing_total_budget_ms"])
                    if "model_failure_threshold" in policy:
                        self.failure_threshold = int(policy["model_failure_threshold"])
                    if "model_cooldown_seconds" in policy:
                        self.cooldown_seconds = float(policy["model_cooldown_seconds"])
                    if "max_active_attempts_per_request" in policy:
                        self.max_attempts = int(policy["max_active_attempts_per_request"])
                    if "model_preflight_timeout_ms" in policy:
                        preflight_s = float(policy["model_preflight_timeout_ms"]) / 1000.0
                        self.default_attempt_timeout_s = max(15.0, preflight_s * 1.5)
                logger.info(
                    "Loaded verified pools for %d tasks and policy (budget=%.1fs, attempt_to=%.1fs) from %s",
                    len(self._task_pools), self.total_budget_ms / 1000.0, self.default_attempt_timeout_s, self.routing_table_path,
                )
                return
            except Exception as exc:
                logger.warning("Failed to load %s: %s", self.routing_table_path, exc)

        # Built-in fallback verified models based on Task 7 benchmark results
        default_reliable = [
            "nvidia/ising-calibration-1.5-31b",
            "meta/llama-3.2-11b-vision-instruct",
            "poolside/laguna-xs-2.1",
            "nvidia/nemotron-3-super-120b-a12b",
        ]
        for t in TaskType:
            self._task_pools[t.value] = list(default_reliable)

    def get_health(self, model_id: str) -> ModelHealth:
        if model_id not in self._health:
            self._health[model_id] = ModelHealth(model_id=model_id)
        return self._health[model_id]

    def set_forced_failure(self, model_id: str, failure_mode: str = "error") -> None:
        """Deterministic test hook to simulate model failures."""
        self._forced_failures[model_id] = failure_mode
        logger.info("[ROUTER HOOK] Forced failure registered for %s (mode=%s)", model_id, failure_mode)

    def clear_forced_failures(self) -> None:
        self._forced_failures.clear()
        logger.info("[ROUTER HOOK] Forced failures cleared")

    def reset_health(self, model_id: str | None = None) -> None:
        if model_id:
            self._health.pop(model_id, None)
        else:
            self._health.clear()

    def select_candidates(self, task_type: TaskType) -> list[str]:
        """Select healthy models from verified pool, prioritizing healthy active models

        and promoting healthy reserves when active models enter cooldown.
        """
        now = time.time()
        pool = self._task_pools.get(task_type.value, [])
        if not pool:
            # Fallback to configured settings model if pool is somehow empty
            pool = [settings.ai_model]

        healthy_candidates: list[str] = []
        cooled_down_candidates: list[str] = []

        for mid in pool:
            h = self.get_health(mid)
            if h.is_healthy(now, self.failure_threshold):
                healthy_candidates.append(mid)
            else:
                cooled_down_candidates.append(mid)

        # Return up to max_attempts healthy candidates; if none healthy, allow retry of expired/first
        if healthy_candidates:
            return healthy_candidates[: self.max_attempts]
        if cooled_down_candidates:
            # If all are in cooldown, pick the one closest to expiry
            logger.warning("[ROUTER] All models in cooldown for %s, probing top candidate", task_type.value)
            return cooled_down_candidates[:1]
        return [settings.ai_model]

    def execute_with_routing(
        self,
        task_type: TaskType,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        """Execute request with fast capability-aware routing.

        Flow:
        1. Query task shortlist (max 3 attempts).
        2. Attempt primary; return immediately on success.
        3. On failure, check remaining budget; if sufficient, try next healthy model.
        4. If budget exhausted or all candidates fail, raise AIProviderUnavailableError.
        """
        req_start = time.perf_counter()
        total_budget_s = self.total_budget_ms / 1000.0
        now_ts = time.time()

        candidates = self.select_candidates(task_type)
        logger.info(
            "[ROUTER START] task=%s candidate_count=%d candidates=%s total_budget=%.1fs",
            task_type.value, len(candidates), candidates, total_budget_s,
        )

        attempts: list[RoutingAttemptRecord] = []
        consecutive_provider_errors = 0

        for attempt_idx, model_id in enumerate(candidates, start=1):
            now_loop = time.perf_counter()
            elapsed_so_far = now_loop - req_start
            remaining_budget_s = total_budget_s - elapsed_so_far

            if remaining_budget_s < self.min_attempt_budget_s:
                logger.warning(
                    "[ROUTER BUDGET CUT-OFF] task=%s attempt=%d remaining_budget=%.2fs < min=%.2fs -> aborting",
                    task_type.value, attempt_idx, remaining_budget_s, self.min_attempt_budget_s,
                )
                break

            # Adaptive attempt timeout based on task type and generation size:
            req_tokens = kwargs.get("max_tokens")
            if req_tokens and req_tokens <= 300:
                base_timeout = 6.0
            elif req_tokens and req_tokens <= 600:
                base_timeout = 8.0
            elif req_tokens and req_tokens >= 2000:
                # Longer 2200-3500 token responses need a reasonable larger attempt budget
                base_timeout = 16.0 if settings.gemini_api_key else 20.0
            elif task_type in (TaskType.EXPLANATION, TaskType.DOCUMENT_QA):
                base_timeout = 9.0 if settings.gemini_api_key else 11.0
            else:
                base_timeout = 8.5 if settings.gemini_api_key else 11.0

            if settings.gemini_api_key:
                # If 1 primary failure/timeout occurred, switch to Gemini fallback promptly
                # to give Gemini at least 12-14s within the 28s total budget.
                if consecutive_provider_errors >= 1:
                    logger.info(
                        "[ROUTER FAST-FALLBACK] %d consecutive failure(s) reached; switching to Gemini fallback promptly",
                        consecutive_provider_errors,
                    )
                    break
                min_gemini_reserve = 12.0
                if remaining_budget_s <= min_gemini_reserve:
                    logger.info(
                        "[ROUTER FAST-FALLBACK] Preserving %.2fs budget for Gemini fallback; breaking candidate loop at attempt %d",
                        remaining_budget_s, attempt_idx,
                    )
                    break
                attempt_timeout = min(base_timeout, max(2.0, remaining_budget_s - min_gemini_reserve))
            else:
                attempt_timeout = min(base_timeout, remaining_budget_s)
            attempt_start = time.perf_counter()

            # Check deterministic test hook
            forced_mode = self._forced_failures.get(model_id)
            if forced_mode:
                forced_elapsed = min(0.05, attempt_timeout)
                logger.warning(
                    "[ROUTER FORCED FAIL] model=%s forced_mode=%s on attempt=%d",
                    model_id, forced_mode, attempt_idx,
                )
                self._record_failure(model_id, is_timeout=(forced_mode == "timeout"))
                record = RoutingAttemptRecord(
                    task_type=task_type,
                    attempt_number=attempt_idx,
                    model_id=model_id,
                    timeout_allocated=attempt_timeout,
                    actual_latency=forced_elapsed,
                    status="FORCED_FAILURE",
                    error_detail=f"Forced test failure mode={forced_mode}",
                    remaining_budget=remaining_budget_s - forced_elapsed,
                )
                attempts.append(record)
                continue

            # Real generation call
            provider = NVIDIAProvider(
                model=model_id,
                timeout=attempt_timeout,
                max_tokens=kwargs.get("max_tokens"),
            )

            try:
                content = provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs,
                )
                attempt_elapsed = time.perf_counter() - attempt_start
                self._record_success(model_id, attempt_elapsed)

                record = RoutingAttemptRecord(
                    task_type=task_type,
                    attempt_number=attempt_idx,
                    model_id=model_id,
                    timeout_allocated=attempt_timeout,
                    actual_latency=attempt_elapsed,
                    status="SUCCESS",
                    remaining_budget=total_budget_s - (time.perf_counter() - req_start),
                )
                attempts.append(record)

                total_elapsed = time.perf_counter() - req_start
                logger.info(
                    "[ROUTER SUCCESS] task=%s answered_by=%s attempt=%d/%d attempt_lat=%.2fs total_lat=%.2fs",
                    task_type.value, model_id, attempt_idx, len(candidates), attempt_elapsed, total_elapsed,
                )
                res = RoutingResult(
                    content=content,
                    answering_model=model_id,
                    attempt_count=attempt_idx,
                    total_elapsed=total_elapsed,
                    attempts=attempts,
                )
                self.last_result = res
                return res

            except (AIProviderAuthError, AIProviderConfigurationError) as exc:
                # Configuration / Auth error is non-recoverable across NVIDIA models
                logger.error("[ROUTER CONFIG/AUTH ERROR] Configuration failure on model=%s: %s", model_id, exc)
                record = RoutingAttemptRecord(
                    task_type=task_type,
                    attempt_number=attempt_idx,
                    model_id=model_id,
                    timeout_allocated=attempt_timeout,
                    actual_latency=time.perf_counter() - attempt_start,
                    status="PROVIDER_NOT_INITIALIZED" if "not configured" in str(exc).lower() else "CONFIGURATION_FAILURE",
                    error_detail=str(exc),
                    remaining_budget=total_budget_s - (time.perf_counter() - req_start),
                )
                attempts.append(record)
                if not settings.gemini_api_key:
                    raise
                break

            except AIProviderUnavailableError as exc:
                attempt_elapsed = time.perf_counter() - attempt_start
                is_timeout = "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                is_rate_limit = "rate limit" in str(exc).lower()
                is_conn_error = "could not connect" in str(exc).lower() or "connection" in str(exc).lower()
                is_503 = "503" in str(exc) or "service unavailable" in str(exc).lower() or "temporarily unavailable" in str(exc).lower()

                self._record_failure(model_id, is_timeout=is_timeout)
                consecutive_provider_errors += 1

                if is_timeout:
                    status_str = "TIMEOUT"
                elif is_503 or is_conn_error:
                    status_str = "PROVIDER_DOWN"
                elif is_rate_limit:
                    status_str = "RATE_LIMITED"
                else:
                    status_str = "FAIL"

                record = RoutingAttemptRecord(
                    task_type=task_type,
                    attempt_number=attempt_idx,
                    model_id=model_id,
                    timeout_allocated=attempt_timeout,
                    actual_latency=attempt_elapsed,
                    status=status_str,
                    error_detail=str(exc),
                    remaining_budget=total_budget_s - (time.perf_counter() - req_start),
                )
                attempts.append(record)
                logger.warning(
                    "[ROUTER ATTEMPT FAILED] task=%s model=%s attempt=%d/%d lat=%.2fs status=%s reason=%s",
                    task_type.value, model_id, attempt_idx, len(candidates), attempt_elapsed, status_str, exc,
                )
                if is_rate_limit:
                    logger.warning(
                        "[ROUTER RATE LIMIT] Provider-wide rate limit encountered on %s; stopping fallback loop immediately",
                        model_id,
                    )
                    break
                is_provider_down = "provider is temporarily unavailable" in str(exc).lower() or is_conn_error
                if is_provider_down:
                    logger.warning(
                        "[ROUTER PROVIDER_DOWN FAIL-FAST] Provider-wide outage on %s (%s); stopping fallback loop immediately",
                        model_id, exc,
                    )
                    break

            except AIProviderError as exc:
                attempt_elapsed = time.perf_counter() - attempt_start
                self._record_failure(model_id, is_timeout=False)
                consecutive_provider_errors += 1

                record = RoutingAttemptRecord(
                    task_type=task_type,
                    attempt_number=attempt_idx,
                    model_id=model_id,
                    timeout_allocated=attempt_timeout,
                    actual_latency=attempt_elapsed,
                    status="FAIL",
                    error_detail=str(exc),
                    remaining_budget=total_budget_s - (time.perf_counter() - req_start),
                )
                attempts.append(record)
                logger.warning(
                    "[ROUTER ATTEMPT FAILED] task=%s model=%s attempt=%d/%d lat=%.2fs reason=%s",
                    task_type.value, model_id, attempt_idx, len(candidates), attempt_elapsed, exc,
                )

        # Cross-provider Gemini fallback if candidates did not succeed and Gemini key is configured
        if settings.gemini_api_key:
            gemini_res = self._execute_gemini_fallback(
                task_type=task_type,
                prompt=prompt,
                system_prompt=system_prompt,
                req_start=req_start,
                total_budget_s=total_budget_s,
                attempts=attempts,
                **kwargs,
            )
            if gemini_res is not None:
                self.last_result = gemini_res
                return gemini_res

        total_elapsed = time.perf_counter() - req_start
        timeout_reason = None
        failure_classification = "UNKNOWN"
        for a in attempts:
            if a.status == "TIMEOUT":
                timeout_reason = a.error_detail or f"Timeout on {a.model_id} after {a.timeout_allocated}s"
                failure_classification = "TIMEOUT"
            elif a.status == "PROVIDER_DOWN" and failure_classification != "TIMEOUT":
                failure_classification = "PROVIDER_DOWN"
            elif a.status in ("FAIL", "CONFIGURATION_FAILURE") and failure_classification not in ("TIMEOUT", "PROVIDER_DOWN"):
                failure_classification = a.status

        self.last_result = RoutingResult(
            content="",
            answering_model="none",
            attempt_count=len(attempts),
            total_elapsed=total_elapsed,
            attempts=attempts,
            gemini_fallback_ms=0.0,
            timeout_reason=timeout_reason,
            failure_classification=failure_classification,
        )
        logger.error(
            "[ROUTER EXHAUSTED] task=%s attempts=%d total_elapsed=%.2fs -> honest unavailable",
            task_type.value, len(attempts), total_elapsed,
        )
        raise AIProviderUnavailableError(
            f"Honest synthesis unavailable: all active verified candidates for {task_type.value} failed or latency budget exhausted."
        )

    def _execute_gemini_fallback(
        self,
        task_type: TaskType,
        prompt: str,
        system_prompt: str | None,
        req_start: float,
        total_budget_s: float,
        attempts: list[RoutingAttemptRecord],
        **kwargs: Any,
    ) -> RoutingResult | None:
        if not settings.gemini_api_key:
            return None

        elapsed_so_far = time.perf_counter() - req_start
        remaining_budget_s = total_budget_s - elapsed_so_far
        if remaining_budget_s < self.min_attempt_budget_s:
            logger.warning("[ROUTER GEMINI FALLBACK] Insufficient budget remaining: %.2fs < %.2fs", remaining_budget_s, self.min_attempt_budget_s)
            return None

        gemini_model = settings.gemini_model
        model_tag = f"gemini/{gemini_model}"
        attempt_idx = len(attempts) + 1
        attempt_timeout = min(30.0, remaining_budget_s)

        # Check deterministic test hook for gemini
        forced_mode = self._forced_failures.get(model_tag) or self._forced_failures.get(gemini_model) or self._forced_failures.get("gemini")
        if forced_mode:
            forced_elapsed = min(0.05, attempt_timeout)
            logger.warning("[ROUTER FORCED FAIL] gemini forced_mode=%s on attempt=%d", forced_mode, attempt_idx)
            self._record_failure(model_tag, is_timeout=(forced_mode == "timeout"))
            attempts.append(RoutingAttemptRecord(
                task_type=task_type,
                attempt_number=attempt_idx,
                model_id=model_tag,
                timeout_allocated=attempt_timeout,
                actual_latency=forced_elapsed,
                status="FORCED_FAILURE",
                error_detail=f"Forced test failure mode={forced_mode}",
                remaining_budget=remaining_budget_s - forced_elapsed,
            ))
            return None

        attempt_start = time.perf_counter()
        logger.info(
            "[ROUTER GEMINI FALLBACK] task=%s attempt=%d model=%s timeout=%.1fs remaining_budget=%.1fs",
            task_type.value, attempt_idx, gemini_model, attempt_timeout, remaining_budget_s,
        )

        provider = GeminiProvider(
            model=gemini_model,
            timeout=attempt_timeout,
            max_tokens=kwargs.get("max_tokens"),
        )
        try:
            content = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )
            attempt_elapsed = time.perf_counter() - attempt_start
            self._record_success(model_tag, attempt_elapsed)

            record = RoutingAttemptRecord(
                task_type=task_type,
                attempt_number=attempt_idx,
                model_id=model_tag,
                timeout_allocated=attempt_timeout,
                actual_latency=attempt_elapsed,
                status="SUCCESS",
                remaining_budget=total_budget_s - (time.perf_counter() - req_start),
            )
            attempts.append(record)
            total_elapsed = time.perf_counter() - req_start
            logger.info(
                "[ROUTER SUCCESS] task=%s answered_by=%s attempt=%d attempt_lat=%.2fs total_lat=%.2fs",
                task_type.value, model_tag, attempt_idx, attempt_elapsed, total_elapsed,
            )
            return RoutingResult(
                content=content,
                answering_model=model_tag,
                attempt_count=attempt_idx,
                total_elapsed=total_elapsed,
                attempts=attempts,
                gemini_fallback_ms=round(attempt_elapsed * 1000, 1),
            )
        except (AIProviderAuthError, AIProviderConfigurationError) as exc:
            attempt_elapsed = time.perf_counter() - attempt_start
            logger.error("[ROUTER GEMINI AUTH/CONFIG ERROR] %s", exc)
            attempts.append(RoutingAttemptRecord(
                task_type=task_type,
                attempt_number=attempt_idx,
                model_id=model_tag,
                timeout_allocated=attempt_timeout,
                actual_latency=attempt_elapsed,
                status="CONFIGURATION_FAILURE",
                error_detail=str(exc),
                remaining_budget=total_budget_s - (time.perf_counter() - req_start),
            ))
            return None
        except AIProviderUnavailableError as exc:
            attempt_elapsed = time.perf_counter() - attempt_start
            is_timeout = "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            self._record_failure(model_tag, is_timeout=is_timeout)
            attempts.append(RoutingAttemptRecord(
                task_type=task_type,
                attempt_number=attempt_idx,
                model_id=model_tag,
                timeout_allocated=attempt_timeout,
                actual_latency=attempt_elapsed,
                status="TIMEOUT" if is_timeout else "PROVIDER_DOWN",
                error_detail=str(exc),
                remaining_budget=total_budget_s - (time.perf_counter() - req_start),
            ))
            logger.warning("[ROUTER GEMINI ATTEMPT FAILED] status=%s reason=%s", "TIMEOUT" if is_timeout else "PROVIDER_DOWN", exc)
            return None
        except AIProviderError as exc:
            attempt_elapsed = time.perf_counter() - attempt_start
            self._record_failure(model_tag, is_timeout=False)
            attempts.append(RoutingAttemptRecord(
                task_type=task_type,
                attempt_number=attempt_idx,
                model_id=model_tag,
                timeout_allocated=attempt_timeout,
                actual_latency=attempt_elapsed,
                status="FAIL",
                error_detail=str(exc),
                remaining_budget=total_budget_s - (time.perf_counter() - req_start),
            ))
            logger.warning("[ROUTER GEMINI ATTEMPT FAILED] reason=%s", exc)
            return None

    def stream_with_routing(
        self,
        task_type: TaskType,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream tokens progressively from the primary healthy routed model or fallback."""
        candidates = self.select_candidates(task_type)
        total_budget_s = self.total_budget_ms / 1000.0
        req_start = time.perf_counter()

        last_error: Exception | None = None
        for attempt_idx, model_id in enumerate(candidates, start=1):
            elapsed = time.perf_counter() - req_start
            rem_budget = total_budget_s - elapsed
            if rem_budget < self.min_attempt_budget_s:
                break

            provider = NVIDIAProvider(
                model=model_id,
                timeout=min(self.default_attempt_timeout_s, rem_budget),
                max_tokens=kwargs.get("max_tokens"),
            )
            try:
                gen = provider.stream_generate(prompt=prompt, system_prompt=system_prompt, **kwargs)
                first_chunk = next(gen, None)
                if first_chunk is not None:
                    self._record_success(model_id, time.perf_counter() - req_start)
                    self.last_result = RoutingResult(
                        content="",
                        answering_model=model_id,
                        attempt_count=attempt_idx,
                        total_elapsed=time.perf_counter() - req_start,
                        attempts=[],
                    )
                    yield first_chunk
                    for chunk in gen:
                        yield chunk
                    return
            except (AIProviderAuthError, AIProviderConfigurationError) as exc:
                self._record_failure(model_id, is_timeout=False)
                last_error = exc
                if not settings.gemini_api_key:
                    raise
                break
            except Exception as exc:
                is_to = "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                self._record_failure(model_id, is_timeout=is_to)
                last_error = exc
                continue

        # Gemini fallback if NVIDIA candidates failed or not configured
        if settings.gemini_api_key:
            try:
                g_provider = GeminiProvider(
                    timeout=max(2.0, total_budget_s - (time.perf_counter() - req_start)),
                    max_tokens=kwargs.get("max_tokens"),
                )
                gen = g_provider.stream_generate(prompt=prompt, system_prompt=system_prompt, **kwargs)
                first_chunk = next(gen, None)
                if first_chunk is not None:
                    self.last_result = RoutingResult(
                        content="",
                        answering_model=f"gemini/{g_provider.model}",
                        attempt_count=len(candidates) + 1,
                        total_elapsed=time.perf_counter() - req_start,
                        attempts=[],
                    )
                    yield first_chunk
                    for chunk in gen:
                        yield chunk
                    return
            except Exception as exc:
                last_error = exc

        if last_error:
            raise last_error
        raise AIProviderUnavailableError("All routed models unavailable for streaming.")

    def _record_success(self, model_id: str, latency: float) -> None:
        h = self.get_health(model_id)
        h.success_count += 1
        h.consecutive_failures = 0
        h.cooldown_until = None
        h.last_success = time.time()
        h.recent_latencies.append(latency)
        if len(h.recent_latencies) > 20:
            h.recent_latencies.pop(0)

    def _record_failure(self, model_id: str, is_timeout: bool) -> None:
        h = self.get_health(model_id)
        h.consecutive_failures += 1
        h.last_failure = time.time()
        if is_timeout:
            h.timeout_count += 1
        if is_timeout or h.consecutive_failures >= self.failure_threshold:
            h.cooldown_until = time.time() + self.cooldown_seconds
            logger.warning(
                "[CIRCUIT BREAKER] Model %s placed in cooldown for %.0fs (failures=%d timeout=%s)",
                model_id, self.cooldown_seconds, h.consecutive_failures, is_timeout,
            )


# Global singleton
_global_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    global _global_router
    if _global_router is None:
        _global_router = ModelRouter()
    return _global_router
