import pytest

from collusion.harness.tokens import SPECS, LocalTokenizer, _sha256, deepseek_chat_format, local_tokenizer_available

SYS, USER, REPLY, FEEDBACK = "rules", "history\n\nround", '{"X": 1}', "feedback"


def test_deepseek_chat_format_review_call():
    # encoding_dsv4.py, chat mode: BOS + system, then <｜User｜>...<｜Assistant｜></think> (README "Basic Chat").
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": USER}]
    assert deepseek_chat_format(msgs) == "<｜begin▁of▁sentence｜>rules<｜User｜>history\n\nround<｜Assistant｜></think>"


def test_deepseek_chat_format_continuation_extends_the_review_call():
    first = [{"role": "system", "content": SYS}, {"role": "user", "content": USER}]
    msgs = first + [{"role": "assistant", "content": REPLY}, {"role": "user", "content": FEEDBACK}]
    want = ("<｜begin▁of▁sentence｜>rules<｜User｜>history\n\nround<｜Assistant｜></think>"
            '{"X": 1}<｜end▁of▁sentence｜><｜User｜>feedback<｜Assistant｜></think>')
    assert deepseek_chat_format(msgs) == want
    # Notes and fix calls are strict token-level extensions of the review prompt + reply.
    assert want.startswith(deepseek_chat_format(first) + REPLY)


def test_deepseek_chat_format_thinking_and_bad_shapes():
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": USER}]
    assert deepseek_chat_format(msgs, enable_thinking=True).endswith("<｜Assistant｜><think>")
    with pytest.raises(ValueError):
        deepseek_chat_format(msgs + [{"role": "assistant", "content": "x"}])
    with pytest.raises(ValueError):
        deepseek_chat_format([{"role": "tool", "content": "x"}])


@pytest.mark.parametrize("name", sorted(SPECS))
def test_pinned_files_match(name):
    spec = SPECS[name]
    if not local_tokenizer_available(name):
        pytest.skip(f"{name} tokenizer not downloaded")
    for f, want in spec.sha256.items():
        p = spec.local_dir / f
        if p.exists():
            assert _sha256(p) == want, f


def test_deepseek_special_tokens_are_single_tokens():
    if not local_tokenizer_available("deepseek-v4"):
        pytest.skip("deepseek-v4 tokenizer not downloaded")
    tok = LocalTokenizer(spec="deepseek-v4")
    for s in ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"):
        assert tok.count(s) == 1, s
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": USER}]
    assert tok.count_chat(msgs) == tok.count(SYS) + tok.count(USER) + 4
