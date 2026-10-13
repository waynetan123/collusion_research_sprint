from pathlib import Path

import pytest

from collusion.harness import config as C

CONFIGS = sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_configs_load(path):
    cfg = C.load(path)
    assert cfg.run_id and len(cfg.hash) == 16


def test_unknown_keys_rejected():
    with pytest.raises(ValueError):
        C.from_dict({"run_id": "x", "game": {"audit": 0.2}})
    with pytest.raises(ValueError):
        C.from_dict({"run_id": "x", "arms": ["arm3"]})


def test_hash_tracks_settings():
    a = C.from_dict({"run_id": "x"})
    b = C.from_dict({"run_id": "x", "game": {"audit_rate": 0.1}})
    assert a.hash != b.hash and a.hash == C.from_dict({"run_id": "x"}).hash


def test_openrouter_backend_validation():
    ok = {"kind": "openrouter", "base_url": "https://openrouter.ai/api", "provider": ["open-inference"],
          "tokenizer": "deepseek-v4"}
    cfg = C.from_dict({"run_id": "x", "backend": ok})
    assert cfg.backend.provider == ("open-inference",)
    for bad in ({**ok, "provider": []}, {**ok, "base_url": "http://localhost:8000"}, {**ok, "tokenizer": "gpt"},
                {**ok, "kind": "openai"}):
        with pytest.raises(ValueError):
            C.from_dict({"run_id": "x", "backend": bad})


def test_operational_settings_are_not_hashed():
    base = C.from_dict({"run_id": "x"})
    same = C.from_dict({"run_id": "x", "fix_workers": 9, "backend": {
        "max_cost_usd": 2.5, "concurrency": 48, "timeout_s": 60.0, "max_http_retries": 6}})
    assert base.hash == same.hash
    assert base.hash != C.from_dict({"run_id": "x", "backend": {"model": "other"}}).hash


def test_prompt_text_is_hashed(monkeypatch, tmp_path):
    base = C.from_dict({"run_id": "x"}).hash  # `hash` is a property: take it before the swap
    edited = tmp_path / "prompts.py"
    edited.write_bytes(C.PROMPTS_FILE.read_bytes() + b"# edit\n")
    monkeypatch.setattr(C, "PROMPTS_FILE", edited)
    assert C.from_dict({"run_id": "x"}).hash != base
