import pytest

from collusion.harness.tokens import SPECS, local_tokenizer_available
from collusion.pool import io as pool_io


@pytest.fixture(scope="session")
def pool_list():
    if not pool_io.POOL_FILE.exists():
        pytest.skip("data/pool/pool.jsonl not built")
    return pool_io.load_pool()


@pytest.fixture(scope="session")
def pool(pool_list):
    return {r["id"]: r for r in pool_list}


@pytest.fixture(scope="session", params=sorted(SPECS))
def tok(request):
    """Every token-matching test runs under each supported model's tokenizer."""
    if not local_tokenizer_available(request.param):
        pytest.skip(f"{request.param} tokenizer not downloaded (python -m collusion.harness fetch-tokenizer"
                    f"{'' if request.param == 'qwen3.5' else ' --config configs/smoke_openrouter.toml'})")
    from collusion.harness.tokens import LocalTokenizer

    return LocalTokenizer(spec=request.param)
