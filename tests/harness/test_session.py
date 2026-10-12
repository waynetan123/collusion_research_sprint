"""End-to-end sessions on the fake backend: lockstep, detector, resume, verify."""

import asyncio
import json

from collusion.harness import config as C
from collusion.harness.report import summarize, verify
from collusion.harness.runner import Context, run_branches, run_main


def _cfg(run_id, **kw):
    base = {"run_id": run_id, "seeds": [3], "max_rounds": 60, "arm3_rounds": 5,
            "fake": {"arm1": {"kind": "honest"}, "arm2": {"kind": "colluder", "collude_from": 5}}}
    base.update(kw)
    return C.from_dict(base)


def _run(cfg, tmp_path, tok=None):
    ctx = Context(cfg, tok, fix_testing=False, base=tmp_path)

    async def go():
        try:
            return await run_main(ctx)
        finally:
            await ctx.close()

    return asyncio.run(go()), ctx


def test_lockstep_detector_checkpoint_and_verify(tmp_path, pool_list, pool):
    res, ctx = _run(_cfg("e2e"), tmp_path)
    r = res[3]
    assert r["arm1"]["round_done"] == 60 and r["arm1"]["status"] == "done"
    assert r["arm2"]["status"] == "checkpointed" and r["checkpoint"] == r["arm2"]["round_done"]
    v = verify(ctx.run, pool, ctx.cfg.game)
    assert v["problems"] == [] and v["checked"] == 2 * (60 + r["checkpoint"])
    rows = {row["session"]: row for row in summarize(ctx.run)}
    assert rows["s3-arm2-rep0"]["excess_partner_miss"] > 0.5
    assert abs(rows["s3-arm1-rep0"]["gap"]) < 0.25


def test_deadline_without_collusion(tmp_path, pool_list):
    cfg = _cfg("dl", fake={"arm1": {"kind": "honest"}, "arm2": {"kind": "honest"}})
    res, ctx = _run(cfg, tmp_path)
    assert res[3]["arm2"]["status"] == "deadline" and res[3]["arm2"]["round_done"] == 55
    assert any(e["event"] == "no_collusion" for e in ctx.run.events("s3-arm2-rep0", "detector"))


def test_resume_matches_uninterrupted_run(tmp_path, pool_list):
    full, ctx_full = _run(_cfg("full", max_rounds=20, arms=["arm1"]), tmp_path / "a")
    _run(_cfg("part", max_rounds=20, arms=["arm1"]).replace(max_rounds=8), tmp_path / "b")
    # Simulate a crash mid-round 9: events for round 9 written, no snapshot.
    sid = "s3-arm1-rep0"
    from collusion.harness.logio import RunDir

    rd = RunDir("part", base=tmp_path / "b")
    rd.append(sid, [{"type": "round", "round": 9, "agent": "A", "junk": True}])
    # Resume with the full round budget. The config hash differs (max_rounds), so
    # resume under the original run id with an updated config file.
    p = rd.path / "config.json"
    cfg = _cfg("part", max_rounds=20, arms=["arm1"])
    p.write_text(json.dumps({"hash": cfg.hash, "config": cfg.as_dict()}))
    st = rd.load_state(sid)
    st.last_round, st.status = 20, "running"
    rd.write_json(rd.session_dir(sid) / "session.json", st.to_json())
    _run(cfg, tmp_path / "b")

    a = ctx_full.run.load_state(sid)
    b = rd.load_state(sid)
    assert a.agents["A"].ledger == b.agents["A"].ledger and a.agents["B"].score == b.agents["B"].score
    assert not any(e.get("junk") for e in rd.events(sid))
    # latency and the (simulated) prefix-cache hits legitimately differ after a restart
    strip = lambda evs: [{k: v for k, v in e.items() if k not in ("latency", "cached_tokens")} for e in evs]
    assert strip(ctx_full.run.events(sid)) == strip(rd.events(sid))


def test_branches_run_from_checkpoint(tmp_path, pool_list, pool, tok):
    res, ctx = _run(_cfg("br"), tmp_path, tok)
    k = res[3]["checkpoint"]
    ctx = Context(ctx.cfg, tok, fix_testing=False, base=tmp_path)

    async def go():
        try:
            return await run_branches(ctx, "s3-arm2-rep0", k, ["continue", "truncate_wipe", "blind", "retaliate"])
        finally:
            await ctx.close()

    out = asyncio.run(go())
    assert out["sessions"] == {"continue": k + 5, "truncate_wipe": k + 5, "blind": k + 5, "retaliate": min(60, k + 20)}
    assert verify(ctx.run, pool, ctx.cfg.game)["problems"] == []
    br = ctx.run.events(f"s3-arm2-rep0~truncate_wipe@{k}", "branch")[0]
    assert br["prompt_tokens_next"]["A"]["parent"] == br["prompt_tokens_next"]["A"]["branch"]


def test_config_change_needs_new_run_id(tmp_path, pool_list):
    _run(_cfg("cc", max_rounds=3, arms=["arm1"]), tmp_path)
    try:
        _run(_cfg("cc", max_rounds=4, arms=["arm1"]), tmp_path)
    except RuntimeError as e:
        assert "different config" in str(e)
    else:
        raise AssertionError("expected a config-hash error")
