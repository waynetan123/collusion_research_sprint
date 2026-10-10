from collusion.pool.spec import choose_examples, classify_example

INT = [{"name": "n", "type": "int", "description": ""}]
LST = [{"name": "xs", "type": "list of int", "description": ""}]


def ex(args, expected):
    return {"args": args, "expected": expected}


def test_classify_bool_and_trivial():
    assert classify_example(ex("True", "True"), INT)["bool_arg"]
    assert not classify_example(ex("True", "1"), [{"name": "b", "type": "bool", "description": ""}])["bool_arg"]
    assert classify_example(ex("[]", "[]"), LST)["trivial"]
    assert classify_example(ex("0", "0"), INT)["trivial"]
    assert classify_example(ex("[1, 2]", "[]"), LST)["trivial"]  # empty output
    assert not classify_example(ex("7", "False"), INT)["trivial"]  # predicate output is fine
    assert not classify_example(ex("[1, 2]", "0"), LST)["trivial"]


def test_prefers_non_trivial_over_short_trivial():
    tests = [ex("[]", "[]"), ex("[0]", "[]"), ex("[3, 1]", "[1, 3]"), ex("[5, 4, 2]", "[2, 4, 5]")]
    got = choose_examples([], tests, LST)
    assert got[:2] == [ex("[3, 1]", "[1, 3]"), ex("[5, 4, 2]", "[2, 4, 5]")]
    assert len(got) == 3  # trivial only fills the last slot


def test_bool_args_last_resort():
    tests = [ex("True", "True"), ex("False", "False"), ex("6", "8"), ex("10", "10")]
    assert choose_examples([], tests, INT) == [ex("6", "8"), ex("10", "10"), ex("True", "True")]
    only_bool = [ex("True", "True"), ex("False", "False")]
    assert choose_examples([], only_bool, INT) == only_bool


def test_distinct_outputs_within_tier_and_drafted_first():
    drafted = [ex("[9, 2]", "7")]
    tests = [ex("[1, 1]", "0"), ex("[2, 2]", "0"), ex("[4, 1]", "3")]
    got = choose_examples(drafted, tests, LST)
    assert got[0] == drafted[0]
    assert [g["expected"] for g in got] == ["7", "0", "3"]  # repeated '0' skipped for a new output


def test_dedupes_args():
    d = [ex("[1, 2]", "3")]
    assert choose_examples(d, d + [ex("[2, 2]", "4")], LST) == [ex("[1, 2]", "3"), ex("[2, 2]", "4")]
