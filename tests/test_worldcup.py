"""Tournament-structure tests: draw integrity, bracket, thirds allocation."""

import itertools

import pytest

from src.simulation.worldcup import (
    FINAL,
    GROUPS,
    QF,
    R16,
    R32,
    SF,
    TEAM_TO_GROUP,
    THIRD_SLOTS,
    allocate_thirds,
)


def test_draw_has_48_unique_teams_in_12_groups():
    assert len(GROUPS) == 12
    all_teams = [t for ts in GROUPS.values() for t in ts]
    assert len(all_teams) == 48
    assert len(set(all_teams)) == 48
    assert all(len(ts) == 4 for ts in GROUPS.values())


def test_hosts_in_expected_groups():
    assert TEAM_TO_GROUP["Mexico"] == "A"
    assert TEAM_TO_GROUP["Canada"] == "B"
    assert TEAM_TO_GROUP["United States"] == "D"


def test_r32_uses_every_qualifier_slot_once():
    winners = [ref for slots in R32.values() for k, ref in slots if k == "W"]
    runners = [ref for slots in R32.values() for k, ref in slots if k == "R"]
    thirds = [ref for slots in R32.values() for k, ref in slots if k == "T"]
    assert sorted(winners) == sorted(GROUPS)      # all 12 winners
    assert sorted(runners) == sorted(GROUPS)      # all 12 runners-up
    assert len(thirds) == 8                       # 8 third-place slots


def test_bracket_feeds_are_consistent():
    r32_ids = set(R32)
    assert set(itertools.chain(*R16.values())) == r32_ids
    assert set(itertools.chain(*QF.values())) == set(R16)
    assert set(itertools.chain(*SF.values())) == set(QF)
    assert FINAL == 104


def test_thirds_allocation_valid_for_many_combos():
    """Every 8-of-12 combination must admit a constraint-respecting
    assignment or trigger the documented fallback. Check that the matcher
    finds a valid assignment for a large sample of combinations."""
    groups = sorted(GROUPS)
    solved = 0
    total = 0
    for combo in itertools.combinations(groups, 8):
        total += 1
        alloc = allocate_thirds(list(combo))
        if alloc is None:
            continue
        solved += 1
        assert sorted(alloc.values()) == sorted(combo)
        allowed = dict(THIRD_SLOTS)
        for match_no, g in alloc.items():
            assert g in allowed[match_no]
    assert total == 495
    assert solved / total > 0.95


def test_thirds_allocation_known_combo():
    alloc = allocate_thirds(["A", "B", "C", "D", "E", "F", "G", "H"])
    assert alloc is not None
    assert len(alloc) == 8
