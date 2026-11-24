"""The one place model calls happen. `pr report` asks an analyst agent for a short read of the
results; everything else in the pipeline is local numpy / MLX.

Backends:
  claude  `claude -p --output-format json --json-schema ...` via the locally logged-in Claude
          Code CLI (the user's Claude subscription). ANTHROPIC_API_KEY is stripped from the
          child env so it never silently switches to API billing.
  ollama  local Ollama /api/chat with a JSON-schema `format` (optional fallback).
  fake    canned answer, for tests and offline runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from typing import Callable, Dict, List, Optional


class Backend:
    name = "base"

    def complete_json(self, prompt: str, schema: Dict) -> Dict:
        raise NotImplementedError


class ClaudeCLIBackend(Backend):
    name = "claude"

    def __init__(self, model: Optional[str] = None, timeout: int = 180,
                 run: Callable = subprocess.run, binary: str = "claude"):
        self.model, self.timeout, self.run, self.binary = model, timeout, run, binary

    def command(self, prompt: str, schema: Dict) -> List[str]:
        cmd = [self.binary, "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(schema),
               "--tools", "", "--no-session-persistence", "--strict-mcp-config"]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def complete_json(self, prompt: str, schema: Dict) -> Dict:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        p = self.run(self.command(prompt, schema), capture_output=True, text=True,
                     timeout=self.timeout, env=env)
        if p.returncode != 0:
            raise RuntimeError("claude CLI failed (%d): %s" % (p.returncode, (p.stderr or p.stdout)[-500:]))
        res = json.loads(p.stdout)
        if res.get("is_error"):
            raise RuntimeError("claude CLI error: %s" % res.get("result"))
        out = res.get("structured_output")
        if out is None:
            out = json.loads(res["result"])
        out["_meta"] = {"backend": "claude", "models": list(res.get("modelUsage", {}).keys()),
                        "duration_ms": res.get("duration_ms"), "num_turns": res.get("num_turns")}
        return out


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: str = "http://localhost:11434", timeout: int = 300):
        self.model, self.host, self.timeout = model or "qwen3:4b-instruct", host, timeout

    def complete_json(self, prompt: str, schema: Dict) -> Dict:
        body = json.dumps({"model": self.model, "stream": False, "format": schema,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.host + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(json.loads(r.read())["message"]["content"])
        out["_meta"] = {"backend": "ollama", "models": [self.model]}
        return out


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, answer: Optional[Dict] = None):
        self.calls: List[Dict] = []
        self.answer = answer or {
            "headline": "Fake analyst: prediction-as-routing removes wasted reads.",
            "findings": ["finding one", "finding two", "finding three"],
            "caveats": ["offline fake backend"],
        }

    def complete_json(self, prompt: str, schema: Dict) -> Dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        return dict(self.answer, _meta={"backend": "fake", "models": []})


def get_backend(name: str, model: Optional[str] = None) -> Optional[Backend]:
    if name == "none":
        return None
    return {"claude": ClaudeCLIBackend, "ollama": OllamaBackend, "fake": FakeBackend}[name](
        **({"model": model} if name != "fake" else {}))


ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
        "caveats": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
    },
    "required": ["headline", "findings", "caveats"],
}


def analyst_prompt(summary: Dict) -> str:
    return (
        "You are a systems analyst reviewing a small experiment that reproduces the Edge0 idea "
        "(serve MoE experts from SSD; a trained per-layer prerouter predicts the next layer's experts one "
        "token ahead and the prediction is used as the routing). Read the JSON results and write: a one-"
        "sentence headline, 2-4 findings that cite specific numbers, and 1-3 caveats about what the "
        "simulation does not capture. Be concrete and do not invent numbers.\n\nRESULTS:\n"
        + json.dumps(summary, separators=(",", ":"))
    )
