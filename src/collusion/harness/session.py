"""One session: two agents playing rounds against a fixed schedule.

A round (design doc "Round structure"):
1. Both agents get a review call (concurrently): history + own patch + two
   patches to review. Replies are parsed, re-sampled if unparseable, and
   rejections without a usable fix get one fix re-prompt.
2. Forced verdicts (retaliation branch) are applied.
3. Scoring: correct verdicts, merges, audits (pre-drawn uniforms).
4. Notes call (concurrently): the review exchange continued with verdict-only
   feedback; the agent may write a private note of at most 60 words.
5. A ledger entry is appended to each agent's history; events and a state
   snapshot are written. Fixes are tested in the background.
"""

from __future__ import annotations

import asyncio
import json
import re

from . import render
from .backend import Backend, CallContext
from .config import Config
from .logio import RunDir
from .review import ReviewOutcome, derive_seed, prompt_sha, run_review
from .schedule import AGENTS, LABELS, PARTNER, Schedule, reviewed_slots
from .scoring import score_round
from .state import LedgerEntry, Reviewed, SessionState


def clean_note(text: str, max_words: int) -> tuple[str, bool]:
    """Collapse whitespace (a note must stay on its ledger line) and cap the word count."""
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    words = [w for w in words if w]
    return " ".join(words[:max_words]), len(words) > max_words


class Session:
    def __init__(self, state: SessionState, schedule: Schedule, pool: dict[str, dict], cfg: Config,
                 backend: Backend, run: RunDir, fixtester=None):
        self.state = state
        self.schedule = schedule
        self.pool = pool
        self.cfg = cfg
        self.backend = backend
        self.run = run
        self.fixtester = fixtester
        self.pending: set[asyncio.Future] = set()
        self._prev_history: dict[str, str] = {}

    @property
    def done(self) -> bool:
        return self.state.status != "running" or self.state.round_done >= self.state.last_round

    def _truth(self, rd, agent: str) -> dict:
        """Ground truth for the fake backend (never sent to a real model)."""
        labels = {}
        for lb in LABELS:
            slot = rd.label_slot(agent, lb)
            rec = self.pool[rd.deals[slot].fn_id]
            labels[lb] = {"slot": slot, "partner": slot == PARTNER[agent], "buggy": rd.deals[slot].buggy,
                          "clean_code": rec["clean_code"], "entry_point": rec["entry_point"]}
        ledger = self.state.agents[agent].ledger
        return {"labels": labels, "last_received": ledger[-1].received if ledger else None}

    async def _review(self, agent: str, rd) -> tuple[list[dict], ReviewOutcome]:
        messages = render.review_messages(self.state, agent, rd, self.pool, self.cfg.game)
        hist = render.history_text(self.state.agents[agent].ledger)
        prev = self._prev_history.get(agent)
        if prev is not None and not hist.startswith(prev):
            raise AssertionError(f"{self.state.session_id}: history of {agent} is not append-only at round {rd.round}")
        self._prev_history[agent] = hist
        eps = {lb: self.pool[rd.deals[rd.label_slot(agent, lb)].fn_id]["entry_point"] for lb in LABELS}
        ctx = CallContext(self.state.session_id, self.state.arm, rd.round, agent, "review", truth=self._truth(rd, agent))
        out = await run_review(self.backend, messages, list(LABELS), eps, ctx, self.cfg.limits,
                               (self.state.session_id, rd.round, agent))
        return messages, out

    async def _notes(self, agent: str, rd, messages: list[dict], outcome: ReviewOutcome, feedback: str):
        reply_text = outcome.reply_text or json.dumps(
            {lb: {k: outcome.reviews[lb][k] for k in ("verdict", "reason", "fix")} for lb in LABELS})
        nmsgs = messages + [{"role": "assistant", "content": reply_text}, {"role": "user", "content": feedback}]
        seed = derive_seed(self.state.session_id, rd.round, agent, "notes", 0)
        ctx = CallContext(self.state.session_id, self.state.arm, rd.round, agent, "notes")
        reply = await self.backend.chat(nmsgs, max_tokens=self.cfg.limits.notes_max_tokens, seed=seed, schema=None, ctx=ctx)
        return nmsgs, seed, reply

    async def play_round(self) -> dict:
        """Play the next round. Returns {"round": r, "partner": {...}} outcomes for the detector."""
        st = self.state
        r = st.round_done + 1
        rd = self.schedule[r]
        blind = render.is_blind(st, r)

        results = await asyncio.gather(*(self._review(a, rd) for a in AGENTS))
        msgs = {a: m for a, (m, _) in zip(AGENTS, results)}
        outs = {a: o for a, (_, o) in zip(AGENTS, results)}

        model_verdicts = {a: {rd.label_slot(a, lb): outs[a].reviews[lb]["verdict"] for lb in LABELS} for a in AGENTS}
        verdicts = {a: dict(v) for a, v in model_verdicts.items()}
        forced = [f for f in st.forced if f.round == r]
        for f in forced:
            verdicts[f.reviewer][f.slot] = f.verdict
        scores = score_round(rd, verdicts, self.cfg.game)

        feedback = {}
        for a in AGENTS:
            s = scores[a]
            feedback[a] = render.feedback_text(st.arm, a, r, s.received, s.n_correct, s.merged, s.n_audited,
                                               self.cfg.game, st.agents[a].score + s.delta)
        notes = await asyncio.gather(*(self._notes(a, rd, msgs[a], outs[a], feedback[a]) for a in AGENTS))

        events = []
        for a, (nmsgs, nseed, nreply) in zip(AGENTS, notes):
            ag = st.agents[a]
            s = scores[a]
            note, truncated = clean_note(nreply.text, self.cfg.limits.note_max_words)
            tags = render.slot_tags(st.arm, a, blind)
            reviews = {}
            for lb in LABELS:
                slot = rd.label_slot(a, lb)
                deal = rd.deals[slot]
                rv = outs[a].reviews[lb]
                was_forced = any(f.reviewer == a and f.slot == slot for f in forced)
                reviews[lb] = {
                    "slot": slot, "fn_id": deal.fn_id, "source": deal.source, "buggy": deal.buggy,
                    "partner": slot == PARTNER[a], "tag": tags[slot],
                    "verdict": verdicts[a][slot], "model_verdict": model_verdicts[a][slot],
                    "forced": was_forced,
                    "reason": rv["reason"], "fix": rv["fix"], "fix_reprompted": rv["fix_reprompted"],
                    # A forced rejection needs no fix (design doc, retaliation test).
                    "fix_missing": rv["fix_missing"] and verdicts[a][slot] == "REJECT" and not was_forced,
                    "correct": s.correct[slot], "audited": s.audited[slot], "audit_u": rd.audit_u[a][slot],
                }
            ag.score += s.delta
            ag.ledger.append(LedgerEntry(
                round=r, reviewed=[Reviewed(lb, tags[rd.label_slot(a, lb)], verdicts[a][rd.label_slot(a, lb)]) for lb in LABELS],
                received=s.received, score=ag.score, note=note,
            ))
            own = rd.deals[a]
            accepted = next((c for c in outs[a].calls if c.kind == "review" and c.ok), None)
            events += [{"type": "call", "round": r, "agent": a, **c.as_event()} for c in outs[a].calls]
            events.append({"type": "call", "round": r, "agent": a, "kind": "notes", "attempt": 0, "seed": nseed,
                           "prompt_sha256": prompt_sha(nmsgs), "prompt_tokens": nreply.prompt_tokens,
                           "completion_tokens": nreply.completion_tokens, "cached_tokens": nreply.cached_tokens,
                           "finish_reason": nreply.finish_reason, "latency": round(nreply.latency, 3), "ok": True,
                           "error": None, "text": nreply.text,
                           **({"reasoning": nreply.reasoning} if nreply.reasoning else {})})
            events.append({
                "type": "round", "round": r, "agent": a, "arm": st.arm, "blind": blind, "reviews": reviews,
                "own": {"slot": a, "fn_id": own.fn_id, "source": own.source, "buggy": own.buggy,
                        "received": s.received, "merged": s.merged},
                "n_correct": s.n_correct, "n_audited": s.n_audited, "delta": s.delta, "score": ag.score,
                "note": note, "note_truncated": truncated, "verdict_imputed": outs[a].verdict_imputed,
                "n_resamples": outs[a].n_resamples,
                "review_prompt_sha256": prompt_sha(msgs[a]),
                "review_prompt_tokens": accepted.reply.prompt_tokens if accepted else None,
            })

        st.round_done = r
        if st.round_done >= st.last_round:
            st.status = "done"
        self.run.append(st.session_id, events)
        self.run.save_state(st)
        self._submit_fixes(r, events)

        return {"round": r, "partner": {
            f"{a}>{PARTNER[a]}": {"buggy": rd.deals[PARTNER[a]].buggy,
                                  "approved": verdicts[a][PARTNER[a]] == "APPROVE",
                                  "valid": not outs[a].verdict_imputed}
            for a in AGENTS}}

    def _submit_fixes(self, r: int, events: list[dict]) -> None:
        if self.fixtester is None:
            return
        for e in events:
            if e["type"] != "round":
                continue
            for lb, rv in e["reviews"].items():
                if rv["verdict"] != "REJECT" or rv["fix_missing"] or rv["forced"]:
                    continue
                rec = self.pool[rv["fn_id"]]
                fut = self.fixtester.submit(rv["fn_id"], rec["entry_point"], rec["source"], rec["atol"], rv["fix"])
                meta = {"type": "fixtest", "round": r, "agent": e["agent"], "label": lb, "slot": rv["slot"],
                        "fn_id": rv["fn_id"], "buggy": rv["buggy"]}
                fut.add_done_callback(lambda f, meta=meta: self._fix_done(f, meta))
                self.pending.add(fut)

    def _fix_done(self, fut: asyncio.Future, meta: dict) -> None:
        self.pending.discard(fut)
        try:
            res = fut.result()
        except Exception as e:  # worker crash etc.: logged, can be backfilled
            res = {"error": f"{type(e).__name__}: {e}"[:300]}
        self.run.append(self.state.session_id, [{**meta, **res}])

    async def drain(self) -> None:
        if self.pending:
            await asyncio.gather(*list(self.pending), return_exceptions=True)
            await asyncio.sleep(0)
