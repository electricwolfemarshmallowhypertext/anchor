from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate_patch(self, model: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc

        try:
            decoded = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError("ollama response was not valid JSON") from exc

        response_text = decoded.get("response")
        if not isinstance(response_text, str):
            raise ValueError("ollama response missing 'response' string")

        try:
            patch_dict = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError("ollama 'response' was not valid JSON patch") from exc

        if not isinstance(patch_dict, dict):
            raise ValueError("ollama patch payload must be a JSON object")
        return patch_dict
