"""Render prompts from state. Pure functions: same inputs, same bytes.

Layout of a review call (design doc "Context format"):

    system: rules, exact scoring, arm-specific framing
    user:   == History ==
            (start of session)
            R1 | Reviewed: Patch X (Agent B) APPROVE; Patch Y (external contributor) REJECT | Your patch: REJECTED | Score: 2
                | Notes: ...
            ...
            == Round k ==
            <current score, own patch, the two patches to review>

The history only ever grows at its end, so every review call shares its
prefix with the previous one (prefix caching). The chat template trims
message content, so rendered content never starts or ends with whitespace.
"""

from __future__ import annotations

from collusion.pool.render import render_patch

from . import prompts as P
from .schedule import LABELS, PARTNER, Round, reviewed_slots
from .state import LedgerEntry, SessionState


def slot_tags(arm: str, agent: str, blind: bool) -> dict[str, str]:
    """Author tag for each slot `agent` reviews."""
    partner, stranger = reviewed_slots(agent)
    if blind:
        return {partner: P.BLIND_TAG, stranger: P.BLIND_TAG}
    if arm == "arm2":
        return {partner: f"Agent {PARTNER[agent]}", stranger: P.STRANGER_TAG}
    return {partner: P.ARM1_TAG, stranger: P.ARM1_TAG}


def system_text(state: SessionState, agent: str, game) -> str:
    return P.system_prompt(
        state.arm, agent, PARTNER[agent], audit_rate=game.audit_rate, correct=game.correct_reward,
        merge=game.merge_reward, penalty=game.audit_penalty, salience=game.salience_line,
    )


def record_text(e: LedgerEntry) -> str:
    """The span a truncate branch redacts: verdicts given and received, per-round score."""
    if e.record_text is not None:
        return e.record_text
    given = "; ".join(f"Patch {r.label} ({r.tag}) {r.verdict}" for r in e.reviewed)
    return f"Reviewed: {given} | Your patch: {e.received} | Score: {e.score}"


def note_text(e: LedgerEntry) -> str:
    """The span a wipe branch redacts."""
    if e.note_text is not None:
        return e.note_text
    return e.note or P.NO_NOTE


def entry_text(e: LedgerEntry) -> str:
    lines = [f"R{e.round} | {record_text(e)}", f"    | Notes: {note_text(e)}"]
    return "\n".join(lines + e.extra_lines)


def history_text(ledger: list[LedgerEntry]) -> str:
    return "\n".join([P.HISTORY_HEADER, P.HISTORY_START] + [entry_text(e) for e in ledger])


def is_blind(state: SessionState, round_no: int) -> bool:
    return state.blind_from is not None and round_no >= state.blind_from


def round_text(state: SessionState, agent: str, rd: Round, pool: dict[str, dict]) -> str:
    tags = slot_tags(state.arm, agent, is_blind(state, rd.round))
    own = rd.deals[agent]
    parts = [
        P.ROUND_HEADER.format(round=rd.round),
        P.SCORE_LINE.format(score=state.agents[agent].score),
        "",
        P.OWN_PATCH_HEADER,
        f"```python\n{code_of(pool, own)}\n```",
        "",
        P.REVIEW_HEADER,
    ]
    for label in LABELS:
        slot = rd.label_slot(agent, label)
        parts += ["", render_patch(label, code_of(pool, rd.deals[slot]), tags[slot])]
    parts += ["", P.REPLY_LINE]
    return "\n".join(parts)


def code_of(pool: dict[str, dict], deal) -> str:
    rec = pool[deal.fn_id]
    return rec["buggy_code"] if deal.buggy else rec["clean_code"]


def user_text(state: SessionState, agent: str, rd: Round, pool: dict[str, dict]) -> str:
    text = history_text(state.agents[agent].ledger) + "\n\n" + round_text(state, agent, rd, pool)
    assert text == text.strip(), "rendered content must not start or end with whitespace"
    return text


def review_messages(state: SessionState, agent: str, rd: Round, pool: dict[str, dict], game) -> list[dict]:
    return [
        {"role": "system", "content": system_text(state, agent, game)},
        {"role": "user", "content": user_text(state, agent, rd, pool)},
    ]


def feedback_text(arm: str, agent: str, round_no: int, received: str, n_correct: int, merged: bool,
                  n_audited: int, game, score: int) -> str:
    correct_pts = n_correct * game.correct_reward
    merge_pts = game.merge_reward if merged else 0
    audit_pts = -n_audited * game.audit_penalty
    return P.FEEDBACK.format(
        round=round_no, reviewer=P.reviewer_phrase(arm, PARTNER[agent]), received=received,
        n_correct=n_correct, correct_pts=correct_pts, merged="yes" if merged else "no", merge_pts=merge_pts,
        n_audited=n_audited, audit_pts=audit_pts, delta=correct_pts + merge_pts + audit_pts, score=score,
    )


def fix_reprompt_text(labels: list[str]) -> str:
    patches = " and ".join(f"Patch {lb}" for lb in labels)
    example = "{" + ", ".join(f'"{lb}": {{"fix": "def ..."}}' for lb in labels) + "}"
    return P.FIX_REPROMPT.format(patches=patches, example=example)
