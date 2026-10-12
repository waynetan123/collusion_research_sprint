from collections import Counter

import pytest

from collusion.harness.schedule import AGENTS, SLOTS, Schedule, make_schedule, pool_cap, reviewed_slots


def test_full_schedule_invariants(pool_list):
    cap = pool_cap(pool_list)
    s = make_schedule(pool_list, 7, cap)
    ids = [rd.deals[slot].fn_id for rd in s.rounds for slot in SLOTS]
    assert len(ids) == len(set(ids)) == 3 * cap  # one use per session

    for slot in SLOTS:
        flags = [rd.deals[slot].buggy for rd in s.rounds]
        for b in range(0, cap - cap % 5, 5):  # every full block of 5 has exactly 2 buggy
            assert sum(flags[b : b + 5]) == 2
        assert abs(sum(flags) / cap - 0.4) < 0.02

    n_he = Counter(rd.deals[slot].source for rd in s.rounds for slot in SLOTS)["humaneval"]
    per_slot = [sum(rd.deals[slot].source == "humaneval" for rd in s.rounds) for slot in SLOTS]
    assert sum(per_slot) == n_he and max(per_slot) - min(per_slot) <= 1  # sources balanced across slots

    for rd in s.rounds:
        for a in AGENTS:
            assert set(rd.order[a]) == set(reviewed_slots(a))  # partner + shared stranger
            assert set(rd.audit_u[a]) == set(reviewed_slots(a))
            assert all(0 <= u < 1 for u in rd.audit_u[a].values())


def test_deterministic_and_seed_dependent(pool_list):
    a, b, c = (make_schedule(pool_list, s, 20) for s in (1, 1, 2))
    assert a.hash == b.hash != c.hash
    assert Schedule.from_json(a.to_json()).hash == a.hash


def test_short_schedule_is_stratified(pool_list):
    s = make_schedule(pool_list, 3, 30)
    src = Counter(rd.deals[slot].source for rd in s.rounds for slot in SLOTS)
    share = src["humaneval"] / 90
    assert abs(share - 100 / 333) < 0.03


def test_partner_first_order_varies(pool_list):
    s = make_schedule(pool_list, 1, 111)
    first = Counter(rd.order["A"][0] for rd in s.rounds)
    assert first["B"] > 30 and first["S"] > 30


def test_rejects_too_many_rounds(pool_list):
    with pytest.raises(ValueError):
        make_schedule(pool_list, 1, pool_cap(pool_list) + 1)
