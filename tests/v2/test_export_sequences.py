"""Pool export and sequence dealing."""

import json
from collections import Counter
from pathlib import Path

import pytest

from collusion.v2 import HARNESS_ROOT, POOL_MANIFEST, SEQUENCE_DIR
from collusion.v2.export_tasks import export, file_text, task_entries, tasks_sha256
from collusion.v2.sequences import EPISODES, deal, parse_seeds, sequence_path

LEAKS = ("humaneval", "mbpp", "evalplus", "buggy", "clean", "mutant")


def test_entries_are_neutral_and_complete(pool_records):
    entries = task_entries(pool_records)
    assert len(entries) == 2 * len(pool_records)
    assert len({e["task_id"] for e in entries}) == len(entries)
    assert len({e["code_path"] for e in entries}) == len(entries)
    for entry in entries:
        visible = (entry["task_id"] + entry["code_path"]).lower()
        assert not any(word in visible for word in LEAKS), entry["task_id"]
        assert entry["expected_verdict"] == ("fail" if entry["is_buggy"] else "pass")
    per_record = Counter(e["spec_key"] for e in entries)
    assert set(per_record.values()) == {2}


def test_task_numbers_do_not_follow_pool_order(pool_records):
    entries = task_entries(pool_records)
    by_pool = sorted(entries, key=lambda e: (e["pool_id"], e["version"]))
    numbers = [int(e["task_id"].split("_")[1]) for e in by_pool]
    assert numbers != sorted(numbers)
    # HumanEval tasks must not cluster in a numeric range that reveals the source.
    he = sorted(int(e["task_id"].split("_")[1]) for e in entries if e["source"] == "humaneval")
    assert he[-1] - he[0] > len(entries) // 2


def test_export_writes_stored_code_exactly(pool_records, tmp_path):
    entries = export(pool_records, root=tmp_path)
    by_id = {r["id"]: r for r in pool_records}
    for entry in entries:
        text = (tmp_path / entry["code_path"]).read_text(encoding="utf-8")
        record = by_id[entry["pool_id"]]
        assert text == record[f"{entry['version']}_code"] + "\n"
        assert "humaneval" not in text.lower() and "mbpp" not in text.lower()
    manifest = json.loads((tmp_path / POOL_MANIFEST).read_text())
    assert manifest == entries


def test_tasks_digest_covers_file_contents(pool_records, tmp_path):
    """Same pool gives the same digest; editing one task file changes it."""
    entries = export(pool_records, root=tmp_path / "a")
    export(pool_records, root=tmp_path / "b")
    assert tasks_sha256(tmp_path / "a") == tasks_sha256(tmp_path / "b")
    target = tmp_path / "b" / entries[0]["code_path"]
    target.write_text(target.read_text() + "# edited\n")
    assert tasks_sha256(tmp_path / "a") != tasks_sha256(tmp_path / "b")


def test_committed_export_is_current(pool_records):
    """The exported files in data/tasks/ match a fresh export of the pool."""
    manifest_path = HARNESS_ROOT / POOL_MANIFEST
    if not manifest_path.exists():
        pytest.skip("pool tasks not exported")
    entries = task_entries(pool_records)
    assert json.loads(manifest_path.read_text()) == entries
    by_id = {r["id"]: r for r in pool_records}
    for entry in entries:
        on_disk = (HARNESS_ROOT / entry["code_path"]).read_text(encoding="utf-8")
        assert on_disk == file_text(by_id[entry["pool_id"]], entry["version"])


def test_parse_seeds():
    assert parse_seeds("1-3,7, 5") == [1, 2, 3, 5, 7]
    with pytest.raises(ValueError):
        parse_seeds("0-2")


def test_deal_constraints(manifest_entries):
    by_id = {e["task_id"]: e for e in manifest_entries}
    for seed in range(1, 41):
        pairs = deal(manifest_entries, seed)
        assert len(pairs) == EPISODES
        tasks = [by_id[t] for pair in pairs for t in pair]
        assert len({t["pool_id"] for t in tasks}) == 2 * EPISODES  # no repeats, no twins
        assert sum(t["is_buggy"] for t in tasks) == EPISODES
        assert all(
            sorted(by_id[t]["source"] for t in pair) == ["humaneval", "mbpp"]
            for pair in pairs
        )
        buggy_by_source = Counter(t["source"] for t in tasks if t["is_buggy"])
        assert abs(buggy_by_source["humaneval"] - buggy_by_source["mbpp"]) <= 1
    assert deal(manifest_entries, 3) == deal(manifest_entries, 3)
    assert deal(manifest_entries, 3) != deal(manifest_entries, 4)


def test_written_sequences_load_in_upstream(manifest_entries):
    from collusion.v2.harness.tasks import read_task_pairs

    written = sorted((HARNESS_ROOT / SEQUENCE_DIR).glob("rep*_sampled_manifest.json"))
    if not written:
        pytest.skip("sequences not written")
    for path in written:
        seed = int(path.name[3:6])
        assert path == sequence_path(seed)
        stored = [entry["tasks"] for entry in json.loads(path.read_text())["pairs"]]
        assert stored == deal(manifest_entries, seed)
        pairs = read_task_pairs(path, HARNESS_ROOT, [POOL_MANIFEST])
        assert len(pairs) == EPISODES
