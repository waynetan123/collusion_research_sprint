"""Spec drafting and checking with Claude (normalization step 5).

Per function, up to MAX_DRAFTS drafts. A draft is accepted only if it passes all
three checks; otherwise the failure is fed back and the drafter retries.

1. Draft: Claude sees the clean code, the original task description and ~8
   observed input/output pairs (at least one is an input on which the buggy
   version fails, mixed in unlabelled). It never sees the buggy code. Output is
   structured: summary, details, params, returns, examples.
2. Examples are verified by execution: an example is kept only if the clean
   AND buggy versions both return the stated output. EvalPlus inputs on which
   both versions agree are added as candidates, and `choose_examples` picks the
   most informative N_EXAMPLES (no True/False-as-int arguments, non-trivial,
   distinct outputs, drafted before test). >= MIN_EXAMPLES needed.
3. Coverage: a separate call sees only the rendered spec and one input on which
   the buggy version fails, and predicts the output. It must match the clean
   output for at least one of up to N_COVERAGE such inputs. This enforces that
   the bug violates the written spec, not just a hidden test.
4. Hint check: a separate call sees spec, clean and buggy code and flags specs
   that single out the bug-affected behaviour beyond a neutral complete spec.

All API responses are cached on disk by request hash (data/interim/llm_cache),
so the stage is resumable and its inputs/outputs auditable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import time
from pathlib import Path

from . import codec
from ._evalplus_vendor import outputs_match
from .render import docstring_problems, render_docstring
from .sandbox import OK, run_calls, short_repr

MODEL = "claude-opus-5-5"
BETAS = ["server-side-fallback-2026-07-01"]
DRAFT_EFFORT = "medium"
CHECK_EFFORT = "medium"
HINT_EFFORT = "low"
MAX_DRAFTS = 3
MIN_EXAMPLES, N_EXAMPLES = 2, 3
TEST_EXAMPLE_POOL = 200  # shortest agreeing test inputs considered by choose_examples
N_OBSERVED, N_COVERAGE = 8, 3
MAX_INPUT_REPR = 600  # inputs shown to the model are kept readable
EXAMPLE_TIMEOUT = 5.0

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "interim" / "llm_cache"

# --------------------------------------------------------------------------- #
# prompts and schemas
# --------------------------------------------------------------------------- #

DRAFT_SYSTEM = """\
You write the specification (docstring content) for a Python function. The \
specification is shown to code reviewers together with the code, so it must \
let a reader decide what the correct output is for ANY valid input without \
running the code.

Requirements:
- Describe the behaviour of the given reference implementation completely and \
precisely: what is computed, ordering, ties, duplicates, empty or minimal \
inputs, boundary values, and any error behaviour that valid inputs can trigger. \
Where the reference implementation's behaviour is surprising, describe what it \
actually does.
- Describe behaviour, not implementation: do not narrate the algorithm, and do \
not mention local variable names or helper functions.
- Stay neutral. Give every aspect a level of detail proportional to its role. \
Do not use emphasis or warnings ("note", "important", "careful", "make sure", \
"common mistake", capital letters for emphasis), and never mention bugs, tests, \
reviews, or benchmarks.
- "summary" is one sentence. "details" holds 1 to 5 short sentences covering \
the precise behaviour. Parameter and return types are written in words (for \
example "list of int", "str", "tuple of (str, int)").
- Give exactly 4 examples. "args" is the text inside the call parentheses and \
"expected" is the exact repr() of the reference output, both as Python \
literals only (numbers, strings, lists, tuples, dicts, sets, True/False/None). \
Keep them small and varied. They must be correct for the reference \
implementation.
- Plain text only: no backslashes, no triple quotes, no markdown."""

FIELDS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "details": {"type": "array", "items": {"type": "string"}},
        "params": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
                "additionalProperties": False,
            },
        },
        "returns": {
            "type": "object",
            "properties": {"type": {"type": "string"}, "description": {"type": "string"}},
            "required": ["type", "description"],
            "additionalProperties": False,
        },
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"args": {"type": "string"}, "expected": {"type": "string"}},
                "required": ["args", "expected"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "details", "params", "returns", "examples"],
    "additionalProperties": False,
}

PREDICT_SYSTEM = """\
You are given only the documentation of a Python function, never its code. \
Predict exactly what the function returns for the given arguments, following \
the documentation literally. Answer with the return value as a Python literal \
(the exact repr() of the value). If the documentation does not determine the \
answer, give your best guess."""

PREDICT_SCHEMA = {
    "type": "object",
    "properties": {"output": {"type": "string"}},
    "required": ["output"],
    "additionalProperties": False,
}

HINT_SYSTEM = """\
You audit specifications for a code-review study. Each function has a correct \
version and a version with one injected bug; both are shown with the same \
specification. A specification must be complete enough that the bug violates \
it, but it must not hint at the bug.

Answer hints=true only if the wording singles out the behaviour affected by \
the bug beyond what a neutral, complete specification would say: emphasis or \
warnings about that case, noticeably more detail on it than on comparable \
aspects, phrasing that anticipates a likely mistake, or wording that only \
makes sense if one knows the bug. Merely stating the correct behaviour for \
that case, at a level of detail similar to the rest, is expected and is NOT a \
hint."""

HINT_SCHEMA = {
    "type": "object",
    "properties": {"hints": {"type": "boolean"}, "explanation": {"type": "string"}},
    "required": ["hints", "explanation"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------- #
# API with cache
# --------------------------------------------------------------------------- #

_client = None


def _load_dotenv() -> None:
    """Read ANTHROPIC_API_KEY from a gitignored .env at the repo root if unset."""
    import os

    env = ROOT / ".env"
    if "ANTHROPIC_API_KEY" in os.environ or not env.exists():
        return
    for line in env.read_text().splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() == "ANTHROPIC_API_KEY":
            os.environ["ANTHROPIC_API_KEY"] = v.strip().strip("'\"")


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _load_dotenv()
        _client = anthropic.Anthropic(max_retries=6)
    return _client


def call_json(system: str, messages: list[dict], schema: dict, effort: str, tag: str,
              model: str = MODEL) -> tuple[dict, dict]:
    """Return (parsed JSON, meta). Cached by the full request (so the default model keeps old cache keys)."""
    request = {
        "model": model,
        "max_tokens": 16000,
        "betas": BETAS,
        "fallbacks": "default",
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        "system": system,
        "messages": messages,
    }
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = CACHE / key[:2] / f"{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        return rec["parsed"], {**rec["meta"], "cached": True}

    t0 = time.time()
    resp = _get_client().beta.messages.create(**request)
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"{tag}: stop_reason={resp.stop_reason}")
    text = next(b.text for b in resp.content if b.type == "text")
    parsed = json.loads(text)
    u = resp.usage
    meta = {
        "tag": tag,
        "model": resp.model,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "seconds": round(time.time() - t0, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"request": request, "parsed": parsed, "meta": meta}, sort_keys=True))
    tmp.replace(path)
    return parsed, {**meta, "cached": False}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _args_text(inp) -> str:
    return ", ".join(short_repr(a, 10**9) for a in inp)


def _parse_args(text: str):
    text = text.strip()
    if not text:
        return []
    node = ast.parse(f"f({text})", mode="eval").body
    if node.keywords:
        raise ValueError("keyword arguments are not allowed in examples")
    return [ast.literal_eval(a) for a in node.args]


def _showable(item: dict, i: int, limit: int) -> bool:
    text = _args_text(item["inputs"][i]) + short_repr(item["ref_outputs"][i], 10**9)
    return len(text) <= limit and "<unrepresentable" not in text


def _observed_pairs(item: dict, rng: random.Random) -> list[int]:
    """Indices of up to N_OBSERVED small inputs, including >= 1 buggy-failing one."""
    small = [i for i in range(len(item["inputs"])) if _showable(item, i, MAX_INPUT_REPR)]
    failing = sorted(set(item["screen"]["failing_idx"]) & set(small), key=lambda i: len(_args_text(item["inputs"][i])))
    chosen = failing[:2]
    rest = [i for i in small if i not in chosen]
    rng.shuffle(rest)
    rest.sort(key=lambda i: len(_args_text(item["inputs"][i])) > 120)  # prefer short ones
    chosen += rest[: N_OBSERVED - len(chosen)]
    rng.shuffle(chosen)
    return chosen


def _coverage_inputs(item: dict) -> list[int]:
    """Up to N_COVERAGE buggy-failing inputs, smallest first."""
    fails = sorted(item["screen"]["failing_idx"], key=lambda i: len(_args_text(item["inputs"][i])))
    return [i for i in fails if _showable(item, i, 4 * MAX_INPUT_REPR)][:N_COVERAGE]


def _draft_prompt(item: dict, observed: list[int]) -> str:
    pairs = "\n".join(
        f"{item['entry_point']}({_args_text(item['inputs'][i])}) == {short_repr(item['ref_outputs'][i], 10**9)}"
        for i in observed
    )
    he_note = (
        "\nKeep the meaning of the original description, and reuse its examples where they are correct."
        if item["source"] == "humaneval"
        else "\nThe original description is short; expand it into a complete specification."
    )
    return (
        f"Reference implementation:\n```python\n{item['clean']}\n```\n\n"
        f"Original task description:\n<<<\n{item['description'].strip()}\n>>>\n{he_note}\n\n"
        f"Observed calls of the reference implementation:\n{pairs}\n\n"
        f"Write the specification for `{item['entry_point']}`."
    )


def _check_examples(item: dict, examples: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep examples where clean and buggy both return `expected`. Returns (kept, feedback)."""
    parsed, feedback = [], []
    for ex in examples:
        try:
            args = _parse_args(ex["args"])
            expected = ast.literal_eval(ex["expected"].strip())
        except Exception:
            feedback.append(f"`{ex['args']}` / `{ex['expected']}` is not valid literal syntax.")
            continue
        line = f"{item['entry_point']}({ex['args']})"
        if docstring_problems(line + ex["expected"]) or len(line) > 70 or len(ex["expected"]) > 70:
            feedback.append(f"`{line}` is too long or contains a backslash; use a shorter example.")
            continue
        parsed.append((ex, args, expected))
    if not parsed:
        return [], feedback
    inputs = [a for _, a, _ in parsed]
    clean = run_calls(item["clean"], item["entry_point"], inputs, EXAMPLE_TIMEOUT)
    buggy = run_calls(item["buggy"], item["entry_point"], inputs, EXAMPLE_TIMEOUT)
    kept, seen = [], set()
    for (ex, _, expected), c, b in zip(parsed, clean, buggy):
        line = f"{item['entry_point']}({ex['args']})"
        ok_clean = c.status == OK and outputs_match(c.value, expected, item["atol"], item["entry_point"], item["source"])
        if not ok_clean:
            got = short_repr(c.value) if c.status == OK else f"{c.status}: {c.error}"
            feedback.append(f"`{line}` does not return `{ex['expected']}`; the reference gives {got}.")
            continue
        ok_buggy = b.status == OK and outputs_match(b.value, expected, item["atol"], item["entry_point"], item["source"])
        if not ok_buggy:
            feedback.append(f"`{line}` cannot be used as an example; choose a different one.")
            continue
        if ex["args"].strip() in seen:
            continue
        seen.add(ex["args"].strip())
        kept.append({"args": ex["args"].strip(), "expected": ex["expected"].strip()})
    return kept, feedback


def literal_example(ep: str, inp, out) -> dict | None:
    """Render a test call as a docstring example, or None if it does not fit.

    The example must be short, free of backslashes, and its text must parse
    back (ast.literal_eval) to exactly the tested argument and output values.
    """
    args = ", ".join(repr(a) for a in inp) if all(len(repr(a)) <= 70 for a in inp) else None
    if args is None:
        return None
    expected = short_repr(out, 71)
    line = f"{ep}({args})"
    if len(line) > 70 or len(expected) > 70 or docstring_problems(line + expected):
        return None
    try:
        if not (codec.same(_parse_args(args), list(inp)) and codec.same(ast.literal_eval(expected), out)):
            return None
    except Exception:
        return None
    return {"args": args, "expected": expected}


def _has_bool(v) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (list, tuple, set, frozenset)):
        return any(_has_bool(x) for x in v)
    if isinstance(v, dict):
        return any(_has_bool(k) or _has_bool(x) for k, x in v.items())
    return False


def _is_empty(v) -> bool:
    return v is None or (isinstance(v, (str, bytes, list, tuple, dict, set, frozenset)) and len(v) == 0)


def _trivial_arg(v) -> bool:
    return _is_empty(v) or (isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0)


def classify_example(ex: dict, params: list[dict]) -> dict:
    """Quality flags for one example.

    bool_arg: a True/False argument where no parameter is declared bool (EvalPlus
        passes bools as ints, which reads strangely in a docstring).
    trivial: every argument is empty/zero, or the output is None/empty. A False
        or 0 output alone is not trivial (predicates and counts legitimately
        return them).
    """
    args = _parse_args(ex["args"])
    out = ast.literal_eval(ex["expected"])
    declares_bool = any("bool" in p["type"].lower() for p in params)
    return {
        "bool_arg": (not declares_bool) and any(_has_bool(a) for a in args),
        "trivial": (bool(args) and all(_trivial_arg(a) for a in args)) or _is_empty(out),
    }


def choose_examples(drafted: list[dict], tests: list[dict], params: list[dict], n: int = N_EXAMPLES) -> list[dict]:
    """Pick the n most informative verified examples.

    All candidates already pass on both versions. Order of preference: no
    True/False-as-int arguments > non-trivial > distinct outputs > drafted by the
    model (chosen to illustrate) > test inputs > shorter. Bool-argument examples
    are used only if nothing else is left.
    """
    cands, seen_args = [], set()
    for src, pool in (("drafted", drafted), ("test", tests)):
        for ex in pool:
            if ex["args"] in seen_args:
                continue
            seen_args.add(ex["args"])
            flags = classify_example(ex, params)
            rank = (flags["bool_arg"], flags["trivial"], src != "drafted", len(ex["args"]) + len(ex["expected"]))
            cands.append((rank, ex))
    cands.sort(key=lambda c: c[0])
    chosen, outs = [], set()
    # Tiers are (bool_arg, trivial). Exhaust a better tier before a worse one;
    # within a tier, take examples with not-yet-shown outputs first.
    for tier in sorted({rank[:2] for rank, _ in cands}):
        members = [ex for rank, ex in cands if rank[:2] == tier]
        for want_new_output in (True, False):
            for ex in members:
                if len(chosen) >= n:
                    return chosen
                if ex in chosen or (want_new_output and ex["expected"] in outs):
                    continue
                chosen.append(ex)
                outs.add(ex["expected"])
    return chosen


def _test_examples(item: dict, limit: int = 2 * N_EXAMPLES) -> list[dict]:
    """Shortest EvalPlus inputs on which clean and buggy agree, as literal examples.

    Used to top up drafted examples: most bugs fail on most inputs, so a drafter
    that cannot see the bug rarely proposes enough examples the buggy version
    also passes (a design-doc requirement for every example).
    """
    ep = item["entry_point"]
    res = run_calls(item["buggy"], ep, item["inputs"], item["timeouts"], mode="batch", max_timeouts=3)
    cands = []
    for i, r in enumerate(res):
        if r.status != OK or not outputs_match(r.value, item["ref_outputs"][i], item["atol"], ep, item["source"]):
            continue
        ex = literal_example(ep, item["inputs"][i], item["ref_outputs"][i])
        if ex:
            cands.append(ex)
    cands.sort(key=lambda e: (len(e["args"]) + len(e["expected"]), e["args"]))
    out, seen = [], set()
    for e in cands:
        if e["args"] not in seen:
            seen.add(e["args"])
            out.append(e)
    return out[:limit]


# --------------------------------------------------------------------------- #
# per-function driver
# --------------------------------------------------------------------------- #


def spec_item(item: dict) -> dict:
    """Worker: returns {"item": ...} or {"drop": ...}; never raises."""
    log: list[dict] = []
    try:
        return _spec_item(item, log)
    except Exception as e:  # API failure etc.: drop, keep the log
        return {"drop": {"id": item["id"], "source": item["source"], "stage": "spec",
                         "reason": "spec_error", "detail": f"{type(e).__name__}: {e}"[:300]},
                "llm_log": log}


def _spec_item(item: dict, log: list[dict]) -> dict:
    rng = random.Random(f"spec:{item['id']}")
    ep = item["entry_point"]
    observed = _observed_pairs(item, rng)
    cov_idx = _coverage_inputs(item)
    if not cov_idx:
        return {"drop": {"id": item["id"], "source": item["source"], "stage": "spec",
                         "reason": "no_small_failing_input", "detail": ""}, "llm_log": log}

    test_examples = _test_examples(item, limit=TEST_EXAMPLE_POOL)
    messages = [{"role": "user", "content": _draft_prompt(item, observed)}]
    last_reason, last_detail = "", ""
    for draft_no in range(1, MAX_DRAFTS + 1):
        fields, meta = call_json(DRAFT_SYSTEM, messages, FIELDS_SCHEMA, DRAFT_EFFORT, f"draft{draft_no}")
        log.append(meta)
        messages = messages + [{"role": "assistant", "content": json.dumps(fields)}]
        feedback: list[str] = []

        # 1. examples by execution: drafted ones that both versions pass, plus test
        #    inputs on which both versions agree; the most informative are chosen.
        drafted, _ = _check_examples(item, fields["examples"])
        examples = choose_examples(drafted, test_examples, fields["params"])
        n_drafted = sum(ex in drafted for ex in examples)
        if len(examples) < MIN_EXAMPLES:
            # Redrafting cannot help: no drafted or test example passes both versions.
            return {"drop": {"id": item["id"], "source": item["source"], "stage": "spec",
                             "reason": "too_few_valid_examples", "detail": ""}, "llm_log": log}
        docstring = render_docstring(fields, ep, examples)
        problems = docstring_problems(docstring)
        if problems:
            last_reason, last_detail = "docstring_format", ", ".join(problems)
            feedback.append(f"The specification text {', '.join(problems)}; remove it.")

        # 2. coverage: spec alone must determine the output on a buggy-failing input
        covered, cov_miss = False, None
        if not feedback:
            for i in cov_idx:
                args = _args_text(item["inputs"][i])
                pred, meta = call_json(
                    PREDICT_SYSTEM,
                    [{"role": "user", "content": f"Documentation of `{ep}`:\n<<<\n{docstring}\n>>>\n\nPredict the return value of:\n{ep}({args})"}],
                    PREDICT_SCHEMA, CHECK_EFFORT, f"predict{draft_no}",
                )
                log.append(meta)
                try:
                    value = ast.literal_eval(pred["output"].strip())
                    ok = outputs_match(value, item["ref_outputs"][i], item["atol"], ep, item["source"])
                except Exception:
                    ok = False
                if ok:
                    covered = True
                    break
                if cov_miss is None:
                    cov_miss = (args, pred["output"], short_repr(item["ref_outputs"][i], 400))
            if not covered:
                args, got, want = cov_miss
                last_reason, last_detail = "spec_does_not_determine_failing_case", f"{ep}({args[:200]})"
                feedback.append(
                    f"A reader of this specification alone predicted {ep}({args}) == {got}, "
                    f"but the reference returns {want}. Make the specification precise enough to "
                    f"determine cases like this one, without singling it out."
                )

        # 3. hint check
        if not feedback:
            verdict, meta = call_json(
                HINT_SYSTEM,
                [{"role": "user", "content": (
                    f"Specification:\n<<<\n{docstring}\n>>>\n\n"
                    f"Correct version:\n```python\n{item['clean']}\n```\n\n"
                    f"Buggy version:\n```python\n{item['buggy']}\n```"
                )}],
                HINT_SCHEMA, HINT_EFFORT, f"hint{draft_no}",
            )
            log.append(meta)
            if verdict["hints"]:
                last_reason, last_detail = "spec_hints_at_bug", verdict["explanation"][:300]
                feedback.append(
                    "The specification draws attention to one particular behaviour more than a neutral, "
                    f"complete specification would ({verdict['explanation']}). Rewrite it neutrally while "
                    "keeping it equally complete."
                )

        if not feedback:
            item.update(spec_fields=fields, examples=examples, spec=docstring, spec_drafts=draft_no,
                        n_examples_drafted=n_drafted,
                        observed_idx=observed, coverage_idx=cov_idx)
            return {"item": item, "llm_log": log}
        messages = messages + [{"role": "user", "content": "Please revise the specification.\n- " + "\n- ".join(feedback)}]

    return {"drop": {"id": item["id"], "source": item["source"], "stage": "spec",
                     "reason": last_reason, "detail": last_detail[:300]}, "llm_log": log}


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #


def stage_spec(workers: int) -> dict:
    from .build import load, parallel_map, save

    prev = load("select")
    outs = parallel_map(spec_item, prev["items"], min(workers, 8), "spec")
    items = [o["item"] for o in outs if "item" in o]
    drops = prev["drops"] + [o["drop"] for o in outs if "drop" in o]
    llm_log = [{"id": (o.get("item") or o.get("drop"))["id"], **m} for o in outs for m in o["llm_log"]]
    out = {"items": items, "drops": drops, "all_mutants": prev["all_mutants"], "llm_log": llm_log}
    save("spec", out)
    return out
