"""Token counting for redaction padding and length matching.

`LocalTokenizer` uses the `tokenizers` package with a pinned tokenizer.json and
a replica of the model's chat format for the message shapes the harness sends
(system, user, assistant without reasoning or tools, generation prompt with
thinking disabled). Two models are supported (`SPECS`):

- qwen3.5: Qwen3.5's tokenizer.json (identical in Qwen/Qwen3.5-27B and
  Qwen/Qwen3.5-27B-FP8 at the pinned revisions) and its Jinja chat template.
  `VLLMBackend.tokenize_chat` asks the server's /tokenize endpoint, which applies
  the real template; the vLLM smoke test checks that both agree.
- deepseek-v4: DeepSeek V4 Flash 0731's tokenizer.json. DeepSeek ships a Python
  reference encoder (encoding/encoding_dsv4.py) instead of a Jinja template; it
  is downloaded and pinned as the reference for `deepseek_chat_format` but never
  imported or run. The OpenRouter smoke test compares local counts with the
  provider's reported prompt tokens.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parents[3]


def chat_format(messages: list[dict], enable_thinking: bool = False, add_generation_prompt: bool = True) -> str:
    """Qwen3.5 chat template for plain-text system/user/assistant messages.

    The template trims every message's content. Assistant messages before the
    last user message are rendered without a reasoning block.
    """
    out = []
    for i, m in enumerate(messages):
        content = m["content"].strip()
        if m["role"] == "system":
            if i != 0:
                raise ValueError("system message must be first")
            out.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif m["role"] == "user":
            out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif m["role"] == "assistant":
            if i == len(messages) - 1:
                raise ValueError("chat_format supports assistant messages only before a user turn")
            if "</think>" in content:
                content = content.split("</think>")[-1].lstrip("\n")
            out.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        else:
            raise ValueError(f"unsupported role {m['role']!r}")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))
    return "".join(out)


DS_BOS, DS_EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
DS_USER, DS_ASSISTANT = "<｜User｜>", "<｜Assistant｜>"
DS_THINK_START, DS_THINK_END = "<think>", "</think>"


def deepseek_chat_format(messages: list[dict], enable_thinking: bool = False, add_generation_prompt: bool = True) -> str:
    """DeepSeek V4 encoding (encoding_dsv4.py) for plain-text system/user/assistant messages.

    Content is not trimmed. Every user turn is followed by the assistant prefix
    and "</think>" in chat mode (thinking off), or "<think>" on the last user
    turn in thinking mode; an earlier assistant reply follows directly and ends
    with EOS (its reasoning is dropped, as in the reference with drop_thinking).
    """
    out = [DS_BOS]
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    for i, m in enumerate(messages):
        if m["role"] == "system":
            if i != 0:
                raise ValueError("system message must be first")
            out.append(m["content"])
        elif m["role"] == "user":
            out.append(DS_USER + m["content"])
            if i + 1 < len(messages) or add_generation_prompt:
                think = DS_THINK_START if enable_thinking and i == last_user else DS_THINK_END
                out.append(DS_ASSISTANT + think)
        elif m["role"] == "assistant":
            if i == len(messages) - 1:
                raise ValueError("deepseek_chat_format supports assistant messages only before a user turn")
            out.append(m["content"] + DS_EOS)
        else:
            raise ValueError(f"unsupported role {m['role']!r}")
    return "".join(out)


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    repo: str
    revision: str
    sha256: dict  # file (repo path) -> sha256; tokenizer.json is required, the rest are checked if present
    local_dir: Path
    chat_format: Callable[..., str]


SPECS = {
    "qwen3.5": TokenizerSpec(
        "qwen3.5", "Qwen/Qwen3.5-27B-FP8", "97f5941bf617e31c5e237364a8602ce3f03a551a",
        {"tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
         "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"},
        ROOT / "data" / "raw" / "qwen" / "Qwen3.5-27B-FP8", chat_format),
    "deepseek-v4": TokenizerSpec(
        "deepseek-v4", "deepseek-ai/DeepSeek-V4-Flash-0731", "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        {"tokenizer.json": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
         "tokenizer_config.json": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
         "encoding/encoding_dsv4.py": "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6"},
        ROOT / "data" / "raw" / "deepseek" / "DeepSeek-V4-Flash-0731", deepseek_chat_format),
}
DEFAULT = "qwen3.5"

# Backwards-compatible names for the Qwen pin.
QWEN_REPO = SPECS["qwen3.5"].repo
QWEN_REVISION = SPECS["qwen3.5"].revision
TOKENIZER_SHA256 = SPECS["qwen3.5"].sha256["tokenizer.json"]
CHAT_TEMPLATE_SHA256 = SPECS["qwen3.5"].sha256["chat_template.jinja"]
LOCAL_DIR = SPECS["qwen3.5"].local_dir


def get_spec(name: str = DEFAULT) -> TokenizerSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(f"unknown tokenizer {name!r}; known: {sorted(SPECS)}") from None


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...

    def count_chat(self, messages: list[dict]) -> int: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_tokenizer(spec: TokenizerSpec | str = DEFAULT) -> Path:
    """Download the pinned tokenizer files if missing; verify hashes. Returns tokenizer.json's path.

    Xet transfers are disabled: inside some sandboxes the Xet client cannot
    verify TLS through the system trust store.
    """
    spec = get_spec(spec) if isinstance(spec, str) else spec
    path = spec.local_dir / "tokenizer.json"
    if not path.exists():
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import hf_hub_download

        for f in spec.sha256:
            hf_hub_download(spec.repo, f, revision=spec.revision, local_dir=str(spec.local_dir))
    for f, want in spec.sha256.items():
        p = spec.local_dir / f
        if p.exists() and _sha256(p) != want:
            raise RuntimeError(f"{p} does not match the pinned sha256 ({spec.name})")
    return path


class LocalTokenizer:
    def __init__(self, path: Path | None = None, enable_thinking: bool = False, spec: TokenizerSpec | str = DEFAULT):
        from tokenizers import Tokenizer as HFTokenizer

        self.spec = get_spec(spec) if isinstance(spec, str) else spec
        self._tok = HFTokenizer.from_file(str(path or fetch_tokenizer(self.spec)))
        self.enable_thinking = enable_thinking

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_chat(self, messages: list[dict]) -> int:
        return self.count(self.spec.chat_format(messages, self.enable_thinking))


def local_tokenizer_available(spec: TokenizerSpec | str = DEFAULT) -> bool:
    spec = get_spec(spec) if isinstance(spec, str) else spec
    return (spec.local_dir / "tokenizer.json").exists()
