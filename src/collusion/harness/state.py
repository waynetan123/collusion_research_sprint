"""Session state: everything needed to render the next prompt and continue play.

State is plain data with a lossless JSON round-trip. A snapshot is written after
every round, so any round can serve as a checkpoint for branching. Ledger
fields that a branch redacts are replaced by pre-padded text (`record_text`,
`note_text`), so rendering never needs a tokenizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Reviewed:
    label: str  # "X" | "Y"
    tag: str  # author tag shown at the time, e.g. "Agent B"
    verdict: str  # "APPROVE" | "REJECT"


@dataclass
class LedgerEntry:
    round: int
    reviewed: list[Reviewed]
    received: str  # "APPROVED" | "REJECTED"
    score: int  # running score after this round
    note: str  # "" if the agent wrote none
    record_text: str | None = None  # truncate: replaces "Reviewed: ... | Score: N"
    note_text: str | None = None  # wipe: replaces the note
    extra_lines: list[str] = field(default_factory=list)  # sham: extra marker lines

    @classmethod
    def from_json(cls, d: dict) -> "LedgerEntry":
        d = dict(d)
        d["reviewed"] = [Reviewed(**x) for x in d["reviewed"]]
        return cls(**d)


@dataclass
class AgentState:
    name: str  # "A" | "B"
    score: int = 0
    ledger: list[LedgerEntry] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict) -> "AgentState":
        return cls(d["name"], d["score"], [LedgerEntry.from_json(e) for e in d["ledger"]])


@dataclass
class Forced:
    """Retaliation test: override `reviewer`'s verdict on `slot` in `round`."""

    round: int
    reviewer: str
    slot: str
    verdict: str


@dataclass
class SessionState:
    session_id: str
    run_id: str
    seed: int
    arm: str  # "arm1" | "arm2"
    replicate: int
    agents: dict[str, AgentState]
    round_done: int = 0  # last completed round
    last_round: int = 0  # play stops after this round (0 = not set)
    status: str = "running"  # running | done | checkpointed | deadline
    parent: str | None = None  # parent session id (branches)
    branch_round: int | None = None  # parent round the branch starts after
    transform: str | None = None
    blind_from: int | None = None  # rounds >= this show no author labels
    forced: list[Forced] = field(default_factory=list)
    config_hash: str = ""
    schedule_hash: str = ""
    pact: bool = False  # forced collusion: the system prompt carries the pact line (prompts_pact.py)
    pact_version: str = "pact-v1"  # which pact text (config.PACT_VERSIONS); older snapshots are all pact-v1

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SessionState":
        d = dict(d)
        d["agents"] = {k: AgentState.from_json(v) for k, v in d["agents"].items()}
        d["forced"] = [Forced(**f) for f in d["forced"]]
        return cls(**d)

    @classmethod
    def new(cls, session_id: str, run_id: str, seed: int, arm: str, replicate: int, last_round: int,
            config_hash: str, schedule_hash: str) -> "SessionState":
        return cls(session_id, run_id, seed, arm, replicate, {a: AgentState(a) for a in ("A", "B")},
                   last_round=last_round, config_hash=config_hash, schedule_hash=schedule_hash)
