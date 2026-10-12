import pytest

from collusion.harness.tokens import local_tokenizer_available
from collusion.pool import io as pool_io


@pytest.fixture(scope="session")
def pool_list():
    if not pool_io.POOL_FILE.exists():
        pytest.skip("data/pool/pool.jsonl not built")
    return pool_io.load_pool()


@pytest.fixture(scope="session")
def pool(pool_list):
    return {r["id"]: r for r in pool_list}


@pytest.fixture(scope="session")
def tok():
    if not local_tokenizer_available():
        pytest.skip("Qwen tokenizer not downloaded (python -m collusion.harness fetch-tokenizer)")
    from collusion.harness.tokens import LocalTokenizer

    return LocalTokenizer()
