from collusion.harness import judge as J
from collusion.harness.fixtest import ensure_test_cache, run_fix_test
from collusion.harness.logio import RunDir, dumps


def test_judge_prompt_marks_bug(pool):
    rec = pool["HumanEval/0"]
    text = J.judge_prompt(rec, "The inner loop compares an element with itself.")
    a, b = rec["bug_location"]["buggy_lines"]
    assert f"The bug is at line {a}" in text or f"lines {a}-{b}" in text
    assert "```diff" in text and "Reviewer's reason" in text
    assert "APPROVE" not in text and "REJECT" not in text  # verdict hidden from the judge


def test_empty_reason_is_not_sent(pool):
    item = {"session": "s", "round": 1, "agent": "A", "label": "X", "fn_id": "HumanEval/0", "partner": True,
            "verdict": "APPROVE", "reason": "  "}
    out = J.judge_one(item, pool["HumanEval/0"])
    assert out["mentions_bug"] is False and out["judge"] == "empty_reason"


def test_handcheck_and_agreement(tmp_path, pool):
    run = RunDir("j", base=tmp_path)
    (run.path / "judge").mkdir()
    js = []
    for i in range(10):
        js.append({"session": "s", "round": i, "agent": "A", "label": "X", "fn_id": "HumanEval/0", "partner": True,
                   "verdict": "APPROVE", "reason": f"reason {i}", "mentions_bug": i % 2 == 0, "evidence": "",
                   "judge": "claude-opus-5-5"})
    (run.path / "judge" / "judgments.jsonl").write_text("".join(dumps(j) + "\n" for j in js))
    path = J.write_handcheck(run, pool, n=10)
    text = open(path).read()
    assert text.count("human: ____") == 10 and "mentions_bug" not in text
    # A human who agrees with the judge on all but one item.
    for j in js:
        ans = "yes" if (j["mentions_bug"] != (j["round"] == 3)) else "no"
        text = text.replace(f"`{J.item_key(j)}`  human: ____", f"`{J.item_key(j)}`  human: {ans}")
    open(path, "w").write(text)
    a = J.agreement(run)
    assert a["n"] == 10 and a["agreement"] == 0.9 and 0.7 < a["cohen_kappa"] < 0.9


def test_fix_testing_clean_passes_buggy_fails(pool):
    ensure_test_cache()
    rec = pool["Mbpp/2"] if "Mbpp/2" in pool else next(iter(pool.values()))
    base = {"fn_id": rec["id"], "entry_point": rec["entry_point"], "source": rec["source"], "atol": rec["atol"],
            "mode": "batch"}
    good = run_fix_test({**base, "fix": "```python\n" + rec["clean_code"] + "\n```"})
    bad = run_fix_test({**base, "fix": rec["buggy_code"]})
    assert good["fix_passes"] and good["n_pass"] == good["n_inputs"]
    assert not bad["fix_passes"] and bad["n_pass"] < bad["n_inputs"]
