"""Terminal progress bars for runs and branches.

One overall bar ("all sessions") on top, then one bar per session with its own
total (the session's last round: 111 for arm 1, the deadline for arm 2, the
branch length for arm 3). When a session stops early (arm 2 at a confirmed
checkpoint), its bar's total shrinks to the round it stopped at, and so does
the overall total, so both end at 100%. The overall bar also shows fix tests
done / queued, since a run waits for queued fix tests after play ends.

The runner reports `{name: (round_done, last_round, status)}` per seed; a
resumed session starts its bar at the rounds already played, so the rate and
ETA count only this invocation's rounds.
"""

from __future__ import annotations

import asyncio

from tqdm import tqdm

FINISHED = ("done", "checkpointed", "deadline")


class Progress:
    def __init__(self, fixtester=None, file=None, refresh_s: float = 2.0):
        self.fixtester = fixtester
        self.file = file
        self.refresh_s = refresh_s
        self.bars: dict[tuple, tqdm] = {}
        self.status: dict[tuple, str] = {}
        self.overall = tqdm(total=0, desc="all sessions", unit="round", position=0, dynamic_ncols=True, file=file,
                            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} session-rounds "
                                       "[{elapsed}<{remaining}]{postfix}")

    def __call__(self, seed, sessions: dict) -> None:
        """sessions: name -> (round_done, last_round, status)."""
        for name, (r, last, status) in sessions.items():
            key = (seed, name)
            if key not in self.bars:
                self.bars[key] = tqdm(total=last, initial=r, desc=f"seed {seed:>2} {name}", unit="round",
                                      position=len(self.bars) + 1, dynamic_ncols=True, file=self.file)
                self.overall.update(r)
            b = self.bars[key]
            if status in FINISHED and r < b.total:  # stopped early (arm 2 checkpoint)
                b.total = r
            if r > b.n:
                delta = r - b.n
                b.update(delta)
                self.overall.update(delta)
            if status != self.status.get(key):
                self.status[key] = status
                b.set_postfix_str("" if status == "running" else status, refresh=False)
            b.refresh()
        self.refresh()

    def refresh(self) -> None:
        self.overall.total = sum(b.total for b in self.bars.values())
        ft = self.fixtester
        if ft is not None and ft.submitted:
            self.overall.set_postfix_str(f"fix tests {ft.completed}/{ft.submitted}", refresh=False)
        self.overall.refresh()

    async def ticker(self) -> None:
        """Refresh every few seconds so elapsed time, ETA and fix-test counts stay live between rounds."""
        while True:
            await asyncio.sleep(self.refresh_s)
            for b in self.bars.values():
                b.refresh()
            self.refresh()

    def close(self) -> None:
        for b in self.bars.values():
            b.close()
        self.overall.close()
