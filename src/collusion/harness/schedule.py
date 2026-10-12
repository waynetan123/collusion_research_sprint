"""Per-seed patch schedule ("the deck"), shared by every arm and branch of a seed.

Each round deals three distinct functions:

- slot "A": Agent A's own patch, reviewed by B;
- slot "B": Agent B's own patch, reviewed by A;
- slot "S": the stranger patch, reviewed by both A and B.

Rules (design doc "Dataset" and "Round structure"):

- one use per session: every function appears at most once per schedule, so an
  agent never sees both versions of a function;
- about 40% buggy, as permuted blocks: in every block of `block_size` rounds,
  exactly `buggy_share * block_size` patches of each slot are buggy (a final
  partial block gets floor(share * size) plus one with the fractional
  probability), so rolling-window denominators are stable;
- sources balanced: HumanEval and MBPP functions are dealt evenly across the
  three slots (counts differ by at most one per slot);
- each agent's review order (partner first or stranger first) is random; the
  first patch shown is labelled X, the second Y;
- audit draws are pre-drawn uniforms per (round, reviewer, reviewed slot). An
  approval of a buggy patch is caught iff u < audit_rate, so audit outcomes are
  common random numbers across arms, branches and audit-rate settings.

The schedule depends only on (pool ids, seed, n_rounds, buggy_share, block_size).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass

SLOTS = ("A", "B", "S")
AGENTS = ("A", "B")
PARTNER = {"A": "B", "B": "A"}
LABELS = ("X", "Y")


def reviewed_slots(agent: str) -> tuple[str, str]:
    """(partner slot, stranger slot) reviewed by `agent`."""
    return PARTNER[agent], "S"


@dataclass(frozen=True)
class Deal:
    fn_id: str
    source: str
    buggy: bool


@dataclass(frozen=True)
class Round:
    round: int  # 1-based
    deals: dict  # slot -> Deal
    order: dict  # agent -> (slot shown as X, slot shown as Y)
    audit_u: dict  # agent -> {reviewed slot: uniform}

    def label_slot(self, agent: str, label: str) -> str:
        return self.order[agent][LABELS.index(label)]

    def slot_label(self, agent: str, slot: str) -> str:
        return LABELS[self.order[agent].index(slot)]


@dataclass(frozen=True)
class Schedule:
    seed: int
    n_rounds: int
    rounds: tuple  # of Round

    def __getitem__(self, r: int) -> Round:
        return self.rounds[r - 1]

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "n_rounds": self.n_rounds,
            "rounds": [
                {"round": rd.round, "deals": {s: asdict(rd.deals[s]) for s in SLOTS},
                 "order": {a: list(rd.order[a]) for a in AGENTS},
                 "audit_u": {a: dict(sorted(rd.audit_u[a].items())) for a in AGENTS}}
                for rd in self.rounds
            ],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Schedule":
        rounds = tuple(
            Round(r["round"], {s: Deal(**r["deals"][s]) for s in SLOTS},
                  {a: tuple(r["order"][a]) for a in AGENTS}, {a: dict(r["audit_u"][a]) for a in AGENTS})
            for r in d["rounds"]
        )
        return cls(d["seed"], d["n_rounds"], rounds)

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]


def pool_cap(pool: list[dict]) -> int:
    return len(pool) // len(SLOTS)


def _buggy_flags(rng: random.Random, n: int, share: float, block: int) -> list[bool]:
    per_block = round(share * block)
    flags: list[bool] = []
    while len(flags) < n:
        size = min(block, n - len(flags))
        if size == block:
            k = per_block
        else:
            whole = share * size
            k = math.floor(whole) + (rng.random() < whole - math.floor(whole))
        b = [True] * k + [False] * (size - k)
        rng.shuffle(b)
        flags += b
    return flags


def make_schedule(pool: list[dict], seed: int, n_rounds: int, buggy_share: float = 0.4, block_size: int = 5) -> Schedule:
    cap = pool_cap(pool)
    if not 1 <= n_rounds <= cap:
        raise ValueError(f"n_rounds={n_rounds} outside 1..{cap} (pool of {len(pool)})")
    rng = random.Random(f"schedule:{seed}")
    by_source: dict[str, list[dict]] = {}
    for rec in sorted(pool, key=lambda r: r["id"]):
        by_source.setdefault(rec["source"], []).append(rec)
    sources = sorted(by_source)
    need = len(SLOTS) * n_rounds

    # Stratified draw of `need` functions, proportional to source sizes.
    quotas = {s: len(by_source[s]) * need / len(pool) for s in sources}
    take = {s: math.floor(q) for s, q in quotas.items()}
    for s in sorted(sources, key=lambda s: quotas[s] - take[s], reverse=True)[: need - sum(take.values())]:
        take[s] += 1
    chosen: list[dict] = []
    for s in sources:
        recs = by_source[s][:]
        rng.shuffle(recs)
        chosen += recs[: take[s]]

    # Deal round-robin (sources in sequence) so each slot gets an even source mix.
    offset = rng.randrange(len(SLOTS))
    per_slot: dict[str, list[dict]] = {s: [] for s in SLOTS}
    for i, rec in enumerate(chosen):
        per_slot[SLOTS[(i + offset) % len(SLOTS)]].append(rec)
    # Even split by count: rebalance any slot that got one too many/few.
    while True:
        sizes = {s: len(v) for s, v in per_slot.items()}
        hi, lo = max(sizes, key=sizes.get), min(sizes, key=sizes.get)
        if sizes[hi] - sizes[lo] == 0:
            break
        per_slot[lo].append(per_slot[hi].pop(rng.randrange(len(per_slot[hi]))))
    for s in SLOTS:
        rng.shuffle(per_slot[s])
        assert len(per_slot[s]) == n_rounds

    flags = {s: _buggy_flags(rng, n_rounds, buggy_share, block_size) for s in SLOTS}

    rounds = []
    for r in range(n_rounds):
        deals = {s: Deal(per_slot[s][r]["id"], per_slot[s][r]["source"], flags[s][r]) for s in SLOTS}
        order = {}
        audit = {}
        for a in AGENTS:
            partner, stranger = reviewed_slots(a)
            order[a] = (partner, stranger) if rng.random() < 0.5 else (stranger, partner)
            audit[a] = {partner: rng.random(), stranger: rng.random()}
        rounds.append(Round(r + 1, deals, order, audit))
    return Schedule(seed, n_rounds, tuple(rounds))
