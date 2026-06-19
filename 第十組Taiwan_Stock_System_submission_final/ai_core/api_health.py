from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time
from typing import Any

import requests


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
GEMINI_KEY_ENV_NAMES = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_STUDIO_API_KEY",
    "GOOGLE_AI_API_KEY",
)


@dataclass
class ApiHealthResult:
    provider: str
    status: str
    model: str
    key_env: str
    message: str
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_gemini_api(model: str | None = None, timeout: int = 20) -> ApiHealthResult:
    api_key, key_env = get_gemini_api_key()
    model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if not api_key:
        return ApiHealthResult(
            provider="Gemini",
            status="missing_key",
            model=model,
            key_env="/".join(GEMINI_KEY_ENV_NAMES),
            message="找不到 Google AI Studio API key，請設定 GOOGLE_API_KEY 或 GEMINI_API_KEY。",
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    start = time.perf_counter()
    try:
        response = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "請只回覆 OK，用來測試 API 是否正常。"}],
                    }
                ],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 8},
            },
            timeout=timeout,
        )
        latency_ms = round((time.perf_counter() - start) * 1000)
        if response.status_code >= 400:
            return ApiHealthResult(
                provider="Gemini",
                status="http_error",
                model=model,
                key_env=key_env,
                message=_google_error_message(response),
                latency_ms=latency_ms,
            )
        payload = response.json()
        parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = " ".join(part.get("text", "") for part in parts).strip()
        if not text:
            return ApiHealthResult(
                provider="Gemini",
                status="response_error",
                model=model,
                key_env=key_env,
                message="API 有回應，但沒有產生文字內容。",
                latency_ms=latency_ms,
            )
        return ApiHealthResult(
            provider="Gemini",
            status="ok",
            model=model,
            key_env=key_env,
            message=f"API 可正常呼叫，回覆：{text[:40]}",
            latency_ms=latency_ms,
        )
    except requests.Timeout:
        return ApiHealthResult(
            provider="Gemini",
            status="network_error",
            model=model,
            key_env=key_env,
            message=f"連線逾時，超過 {timeout} 秒沒有回應。",
        )
    except requests.RequestException as exc:
        return ApiHealthResult(
            provider="Gemini",
            status="network_error",
            model=model,
            key_env=key_env,
            message=f"網路或連線錯誤：{type(exc).__name__}",
        )
    except Exception as exc:
        return ApiHealthResult(
            provider="Gemini",
            status="response_error",
            model=model,
            key_env=key_env,
            message=f"回應解析失敗：{type(exc).__name__}",
        )


def check_openai_api(model: str | None = None, timeout: int = 20) -> ApiHealthResult:
    api_key = os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    if not api_key:
        return ApiHealthResult(
            provider="OpenAI",
            status="missing_key",
            model=model,
            key_env="OPENAI_API_KEY",
            message="找不到 OPENAI_API_KEY。",
        )

    start = time.perf_counter()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "請只回覆 OK，用來測試 API 是否正常。"}],
            temperature=0,
            max_tokens=8,
        )
        latency_ms = round((time.perf_counter() - start) * 1000)
        text = response.choices[0].message.content or ""
        return ApiHealthResult(
            provider="OpenAI",
            status="ok",
            model=model,
            key_env="OPENAI_API_KEY",
            message=f"API 可正常呼叫，回覆：{text[:40]}",
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return ApiHealthResult(
            provider="OpenAI",
            status="http_error",
            model=model,
            key_env="OPENAI_API_KEY",
            message=f"API 呼叫失敗：{type(exc).__name__}",
        )


def check_all_apis(
    provider: str = "all",
    gemini_model: str | None = None,
    openai_model: str | None = None,
    timeout: int = 20,
) -> list[ApiHealthResult]:
    provider = provider.lower().strip()
    results: list[ApiHealthResult] = []
    if provider in {"all", "gemini", "google"}:
        results.append(check_gemini_api(model=gemini_model, timeout=timeout))
    if provider in {"all", "openai"}:
        results.append(check_openai_api(model=openai_model, timeout=timeout))
    return results


def get_gemini_api_key() -> tuple[str | None, str]:
    for name in GEMINI_KEY_ENV_NAMES:
        value = os.getenv(name)
        if value:
            return value, name
    return None, "/".join(GEMINI_KEY_ENV_NAMES)


def _google_error_message(response: requests.Response) -> str:
    try:
        error = response.json().get("error", {})
        code = error.get("code", response.status_code)
        status = error.get("status", "")
        message = error.get("message", response.text)
        return f"HTTP {code} {status}: {_clean_message(message)}"
    except Exception:
        return f"HTTP {response.status_code}: {_clean_message(response.text)}"


def _clean_message(message: str, max_chars: int = 280) -> str:
    message = " ".join(str(message).split())
    return message[:max_chars] + ("..." if len(message) > max_chars else "")
