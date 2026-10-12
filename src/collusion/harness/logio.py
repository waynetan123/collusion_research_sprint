"""Run directory layout and append-only logs.

    data/runs/<run_id>/
        config.json                      frozen config + hash
        schedules/seed_<n>.json          the deck for each seed
        sessions/<session_id>/
            session.json                 latest state (overwritten each round)
            events.jsonl                 calls, rounds, fix tests, detector events
            snapshots.jsonl              full state after every round (checkpoints)
        judge/                           LLM-judge outputs and hand-check sheet

Every JSON line is written with sorted keys. A round's events are written
before its snapshot; on resume, events for rounds after the last snapshot are
dropped (they belong to a round that did not finish).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .state import SessionState

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "data" / "runs"


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


class RunDir:
    def __init__(self, run_id: str, base: Path = RUNS):
        self.path = base / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def session_dir(self, sid: str) -> Path:
        d = self.path / "sessions" / sid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def schedule_path(self, seed: int) -> Path:
        p = self.path / "schedules" / f"seed_{seed}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def sessions(self) -> list[str]:
        d = self.path / "sessions"
        return sorted(p.name for p in d.iterdir() if (p / "session.json").exists()) if d.exists() else []

    def write_json(self, path: Path, obj) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1))
        tmp.replace(path)

    # events --------------------------------------------------------------- #

    def append(self, sid: str, events: list[dict]) -> None:
        with open(self.session_dir(sid) / "events.jsonl", "a", encoding="utf-8") as f:
            for e in events:
                f.write(dumps(e) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def events(self, sid: str, type_: str | None = None) -> list[dict]:
        p = self.session_dir(sid) / "events.jsonl"
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            evs = [json.loads(line) for line in f if line.strip()]
        return [e for e in evs if type_ is None or e.get("type") == type_]

    # state ---------------------------------------------------------------- #

    def save_state(self, state: SessionState) -> None:
        d = self.session_dir(state.session_id)
        with open(d / "snapshots.jsonl", "a", encoding="utf-8") as f:
            f.write(dumps({"round": state.round_done, "state": state.to_json()}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.write_json(d / "session.json", state.to_json())

    def load_state(self, sid: str) -> SessionState | None:
        p = self.session_dir(sid) / "session.json"
        return SessionState.from_json(json.loads(p.read_text())) if p.exists() else None

    def snapshot(self, sid: str, round_no: int) -> SessionState:
        with open(self.session_dir(sid) / "snapshots.jsonl", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                if s["round"] == round_no:
                    return SessionState.from_json(s["state"])
        raise KeyError(f"{sid} has no snapshot for round {round_no}")

    def snapshots(self, sid: str) -> list[SessionState]:
        p = self.session_dir(sid) / "snapshots.jsonl"
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [SessionState.from_json(json.loads(line)["state"]) for line in f if line.strip()]

    def drop_unfinished(self, sid: str, round_done: int) -> None:
        """Remove round-scoped events for rounds after the last snapshot."""
        p = self.session_dir(sid) / "events.jsonl"
        if not p.exists():
            return
        lines = p.read_text(encoding="utf-8").splitlines()
        keep = [ln for ln in lines if ln.strip() and json.loads(ln).get("round", 0) <= round_done]
        if len(keep) != len(lines):
            p.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")
