"""Checkpoint branches (design doc Step 6 retaliation test and Step 7, arm 3).

A branch copies the parent's state after round k and applies one transform:

- continue: nothing changes;
- truncate: for every round <= k, the verdicts given and received and the
  per-round score are redacted (round numbers and the current score stay);
- wipe: every note of rounds <= k is redacted;
- truncate_wipe: both;
- blind: from round k+1 on, both reviewed patches carry the same author tag
  ("author hidden"); received verdicts are still attributed to Agent B;
- sham: everything is kept and marker lines are inserted as extra lines, as
  many as truncate_wipe redacts (two per round <= k);
- retaliate: B's verdict on A's patch is forced to REJECT in the first round
  after k where A's patch is clean.

Redaction is in place: each removed span becomes "[record unavailable]" plus
content-free filler (" ." per token), sized to the span's token count. A span
shorter than the marker (e.g. a one-word note) cannot shrink to fit, so its
deficit is taken from the filler of the longest spans; the invariant asserted
is that the full prompt's token count is unchanged. The padded text is stored
in the state, so later rendering needs no tokenizer.
"""

from __future__ import annotations

from . import prompts as P
from . import render
from .schedule import AGENTS, Schedule
from .state import Forced, SessionState

REDACTING = {"truncate": ("record",), "wipe": ("note",), "truncate_wipe": ("record", "note")}


def _span_original(entry, field: str) -> str:
    return render.record_text(entry) if field == "record" else render.note_text(entry)


def _set_span(entry, field: str, text: str) -> None:
    if field == "record":
        entry.record_text = text
    else:
        entry.note_text = text


def _redacted(n_filler: int) -> str:
    return P.MARKER + P.FILLER * n_filler


def _prompt_tokens(tokenizer, state, agent, rd, pool, game) -> int:
    return tokenizer.count_chat(render.review_messages(state, agent, rd, pool, game))


def pad_redactions(state: SessionState, parent: SessionState, fields: tuple[str, ...], k: int, tokenizer,
                   rd_next, pool, game) -> dict:
    """Redact `fields` in every ledger entry with round <= k, token-exact per agent."""
    report = {}
    m = tokenizer.count(" " + P.MARKER)
    for agent in AGENTS:
        spans = []  # (entry, field, original token count)
        for e in state.agents[agent].ledger:
            if e.round > k:
                continue
            for f in fields:
                spans.append((e, f, tokenizer.count(" " + _span_original(e, f))))
        target = _prompt_tokens(tokenizer, parent, agent, rd_next, pool, game)
        fill = [t - m for _, _, t in spans]
        deficit = sum(-x for x in fill if x < 0)
        fill = [max(0, x) for x in fill]
        while deficit > 0:  # take the shortfall from the longest spans
            i = max(range(len(fill)), key=lambda j: fill[j]) if fill else None
            if i is None or fill[i] == 0:
                raise RuntimeError(f"{agent}: redacted spans cannot fit the original token count")
            fill[i] -= 1
            deficit -= 1
        for (e, f, _), n in zip(spans, fill):
            _set_span(e, f, _redacted(n))
        # Token boundaries can merge differently in context; correct on the longest span.
        got = _prompt_tokens(tokenizer, state, agent, rd_next, pool, game)
        tries = 0
        while got != target and spans and tries < 10:
            i = max(range(len(fill)), key=lambda j: fill[j])
            fill[i] = max(0, fill[i] + target - got)
            _set_span(spans[i][0], spans[i][1], _redacted(fill[i]))
            got = _prompt_tokens(tokenizer, state, agent, rd_next, pool, game)
            tries += 1
        if got != target:
            raise AssertionError(f"{agent}: prompt tokens {got} != {target} after redaction")
        report[agent] = {"prompt_tokens": got, "n_spans": len(spans), "marker_tokens": m}
    return report


def make_branch(parent: SessionState, transform: str, new_sid: str, schedule: Schedule, pool: dict, cfg,
                tokenizer=None, max_round: int | None = None) -> tuple[SessionState, dict]:
    k = parent.round_done
    max_round = min(max_round or schedule.n_rounds, schedule.n_rounds)
    if k >= max_round:
        raise ValueError(f"cannot branch after the last round ({k})")
    st = SessionState.from_json(parent.to_json())
    st.session_id, st.parent, st.branch_round, st.transform = new_sid, parent.session_id, k, transform
    st.status = "running"
    n = cfg.retaliation_rounds if transform == "retaliate" else cfg.arm3_rounds
    st.last_round = min(k + n, max_round)
    rd_next = schedule[k + 1]
    report: dict = {"transform": transform, "branch_round": k, "last_round": st.last_round}

    if transform in REDACTING:
        if tokenizer is None:
            raise ValueError("redacting branches need a tokenizer")
        report["tokens"] = pad_redactions(st, parent, REDACTING[transform], k, tokenizer, rd_next, pool, cfg.game)
    elif transform == "blind":
        st.blind_from = k + 1
    elif transform == "sham":
        for a in AGENTS:
            for e in st.agents[a].ledger:
                if e.round <= k:
                    e.extra_lines = [f"    | {P.MARKER}"] * 2
    elif transform == "retaliate":
        r = next((rd.round for rd in schedule.rounds[k:] if not rd.deals["A"].buggy), None)
        if r is None or r > st.last_round:
            raise ValueError("no clean A patch after the checkpoint for the retaliation test")
        st.forced = [Forced(r, "B", "A", "REJECT")]
        report["forced_round"] = r
    elif transform != "continue":
        raise ValueError(f"unknown transform {transform!r}")

    if tokenizer is not None:
        report["prompt_tokens_next"] = {
            a: {"parent": _prompt_tokens(tokenizer, parent, a, rd_next, pool, cfg.game),
                "branch": _prompt_tokens(tokenizer, st, a, rd_next, pool, cfg.game)} for a in AGENTS}
        if transform in ("continue", "blind", "retaliate", *REDACTING):
            for a, v in report["prompt_tokens_next"].items():
                assert v["parent"] == v["branch"], f"{transform}: {a} prompt tokens changed {v}"
    return st, report
