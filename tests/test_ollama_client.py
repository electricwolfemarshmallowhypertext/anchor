from __future__ import annotations

import io
import json
import urllib.error

import pytest

from anchor.ollama_client import OllamaClient


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_generate_patch_uses_json_format_mode(monkeypatch) -> None:
    captured_payload: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        return _FakeResponse('{"response":"{\\"agent_id\\":\\"demo\\",\\"from_version\\":1}"}')

    monkeypatch.setattr("anchor.ollama_client.urllib.request.urlopen", fake_urlopen)
    client = OllamaClient()
    result = client.generate_patch(model="anchor", prompt="test", schema={"type": "object"})

    assert result == {"agent_id": "demo", "from_version": 1}
    assert captured_payload["format"] == "json"
    assert captured_payload["stream"] is False


def test_generate_patch_includes_http_error_body_in_runtime_error(monkeypatch) -> None:
    error_body = b'{"error":"invalid JSON schema in format"}'

    def fake_urlopen(request, timeout):
        del request, timeout
        raise urllib.error.HTTPError(
            url="http://localhost:11434/api/generate",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(error_body),
        )

    monkeypatch.setattr("anchor.ollama_client.urllib.request.urlopen", fake_urlopen)
    client = OllamaClient()

    with pytest.raises(RuntimeError) as exc_info:
        client.generate_patch(model="anchor", prompt="test")

    message = str(exc_info.value)
    assert "status=500" in message
    assert "invalid JSON schema in format" in message
    assert exc_info.value.__cause__ is not None

