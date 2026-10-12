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
