"""Keep LiteLLM offline during tests and expose the exported pool fixtures."""

import os

import pytest

from collusion.pool.io import POOL_FILE
from collusion.v2 import HARNESS_ROOT

# Must be set before litellm is first imported: no price-map fetch from GitHub.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(scope="session")
def pool_records():
    if not POOL_FILE.exists():
        pytest.skip("data/pool/pool.jsonl not built")
    from collusion.pool.io import load_pool

    return load_pool()


@pytest.fixture(scope="session")
def manifest_entries():
    import json

    from collusion.v2 import POOL_MANIFEST

    path = HARNESS_ROOT / POOL_MANIFEST
    if not path.exists():
        pytest.skip("pool tasks not exported (python -m collusion.v2.export_tasks)")
    return json.loads(path.read_text(encoding="utf-8"))
