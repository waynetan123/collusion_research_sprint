"""Token counting for redaction padding and length matching.

`LocalTokenizer` uses the `tokenizers` package with Qwen3.5's tokenizer.json
(identical in Qwen/Qwen3.5-27B and Qwen/Qwen3.5-27B-FP8 at the pinned
revisions). `chat_format` replicates the Qwen3.5 chat template for the message
shapes the harness sends (system, user, assistant without reasoning or tools,
generation prompt with thinking disabled). `VLLMTokenizer` asks the server's
/tokenize endpoint, which applies the real template; the smoke test checks
that both agree.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[3]
QWEN_REPO = "Qwen/Qwen3.5-27B-FP8"
QWEN_REVISION = "97f5941bf617e31c5e237364a8602ce3f03a551a"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
CHAT_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
LOCAL_DIR = ROOT / "data" / "raw" / "qwen" / "Qwen3.5-27B-FP8"


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


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...

    def count_chat(self, messages: list[dict]) -> int: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_tokenizer(local_dir: Path = LOCAL_DIR) -> Path:
    """Download the pinned tokenizer.json (and chat template) if missing; verify hashes.

    Xet transfers are disabled: inside some sandboxes the Xet client cannot
    verify TLS through the system trust store.
    """
    path = local_dir / "tokenizer.json"
    if not path.exists():
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import hf_hub_download

        for f in ("tokenizer.json", "chat_template.jinja"):
            hf_hub_download(QWEN_REPO, f, revision=QWEN_REVISION, local_dir=str(local_dir))
    if _sha256(path) != TOKENIZER_SHA256:
        raise RuntimeError(f"{path} does not match the pinned sha256")
    tmpl = local_dir / "chat_template.jinja"
    if tmpl.exists() and _sha256(tmpl) != CHAT_TEMPLATE_SHA256:
        raise RuntimeError(f"{tmpl} differs from the template chat_format() replicates")
    return path


class LocalTokenizer:
    def __init__(self, path: Path | None = None, enable_thinking: bool = False):
        from tokenizers import Tokenizer as HFTokenizer

        self._tok = HFTokenizer.from_file(str(path or fetch_tokenizer()))
        self.enable_thinking = enable_thinking

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_chat(self, messages: list[dict]) -> int:
        return self.count(chat_format(messages, self.enable_thinking))


def local_tokenizer_available() -> bool:
    return (LOCAL_DIR / "tokenizer.json").exists()
