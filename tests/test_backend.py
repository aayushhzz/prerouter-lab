import json
import subprocess

from prerouter_lab.backend import ANALYST_SCHEMA, ClaudeCLIBackend, FakeBackend, analyst_prompt, get_backend


def test_claude_backend_builds_cli_call_and_parses_structured_output(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        out = {"type": "result", "is_error": False, "result": "",
               "structured_output": {"headline": "h", "findings": ["a", "b"], "caveats": ["c"]},
               "modelUsage": {"some-model": {}}, "duration_ms": 5, "num_turns": 1}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")

    b = ClaudeCLIBackend(model="haiku", run=fake_run)
    res = b.complete_json("hi", ANALYST_SCHEMA)
    cmd = seen["cmd"]
    assert cmd[:3] == ["claude", "-p", "hi"]
    assert "--json-schema" in cmd and json.loads(cmd[cmd.index("--json-schema") + 1]) == ANALYST_SCHEMA
    assert cmd[cmd.index("--model") + 1] == "haiku" and cmd[cmd.index("--tools") + 1] == ""
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert res["headline"] == "h" and res["_meta"]["models"] == ["some-model"]


def test_claude_backend_raises_on_cli_error():
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "not logged in")

    try:
        ClaudeCLIBackend(run=fake_run).complete_json("x", ANALYST_SCHEMA)
    except RuntimeError as e:
        assert "not logged in" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_fake_backend_and_factory():
    b = get_backend("fake")
    assert isinstance(b, FakeBackend)
    assert b.complete_json(analyst_prompt({"x": 1}), ANALYST_SCHEMA)["findings"]
    assert '"x":1' in b.calls[0]["prompt"]
    assert get_backend("none") is None
