from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, request


@dataclass(slots=True)
class ResponsesClient:
    model: str
    api_key: str | None = None
    api_base: str = "https://api.openai.com/v1"
    reasoning_effort: str = "low"

    def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
    ) -> dict[str, Any]:
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the LLM worker.")

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": tools,
            "reasoning": {"effort": self.reasoning_effort},
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.api_base.rstrip('/')}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed: {exc.code} {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc


def extract_output_text(response_payload: dict[str, Any]) -> str:
    output = response_payload.get("output")
    if not isinstance(output, list):
        return ""
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text = part.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
    return "".join(text_parts).strip()
