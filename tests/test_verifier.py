"""Verifier tests: hand-checked verdicts and explicit-vs-nuXmv agreement."""

import random

import pytest

from logictune.environment import ACTIONS, PROPOSITIONS
from logictune.glm2fsa import Const, Controller, Rule, Var, mk_and, mk_not, mk_or
from logictune.specs import SPEC_IDS
from logictune.verifier import (
    _explicit_verdicts,
    controller_to_smv,
    find_nuxmv,
    run_nuxmv,
    verify_response,
)

IDEAL = """1. If pedestrian_in_front, stop.
2. If stop_sign, stop.
3. If not green_traffic_light, stop.
4. If opposite_car or car_from_left or car_from_right or pedestrian_at_left or pedestrian_at_right, stop.
5. Otherwise, go_straight."""


def test_ideal_controller_satisfies_all_specs():
    res = verify_response(IDEAL)
    assert res.score == 15, res.specs


def test_empty_controller_satisfies_nothing():
    res = verify_response("I cannot help with that.")
    assert res.score == 0


def test_always_stop_fails_exactly_the_liveness_specs():
    res = verify_response("1. Otherwise, stop.")
    failed = {k for k, v in res.specs.items() if not v}
    assert failed == {"phi_7", "phi_10", "phi_13"}


def test_always_go_straight():
    res = verify_response("1. Otherwise, go_straight.")
    failed = {k for k, v in res.specs.items() if not v}
    assert failed == {"phi_1", "phi_3", "phi_4", "phi_8", "phi_14"}


def test_rule_order_matters():
    good = verify_response("1. If pedestrian_in_front, stop.\n2. If green_traffic_light, go_straight.")
    bad = verify_response("1. If green_traffic_light, go_straight.\n2. If pedestrian_in_front, stop.")
    assert good.specs["phi_14"] and not bad.specs["phi_14"]


def _random_controller(rng):
    unk = {f"u{i}": f"phrase {i}" for i in range(rng.randint(0, 2))}
    atoms = PROPOSITIONS + list(unk)

    def expr(depth):
        r = rng.random()
        if depth == 0 or r < 0.35:
            v = Var(rng.choice(atoms))
            return mk_not(v) if rng.random() < 0.4 else v
        if r < 0.45:
            return Const(rng.random() < 0.7)
        sub = [expr(depth - 1) for _ in range(rng.randint(2, 3))]
        return mk_and(sub) if rng.random() < 0.6 else mk_or(sub)

    rules = [Rule(expr(rng.randint(0, 3)), rng.choice(ACTIONS), "") for _ in range(rng.randint(0, 6))]
    used = {u: v for u, v in unk.items() if any(u in r.guard.atoms() for r in rules)}
    return Controller(rules=rules, unknown_atoms=used)


@pytest.mark.skipif(find_nuxmv() is None, reason="nuXmv binary not available")
def test_explicit_checker_matches_nuxmv():
    rng = random.Random(7)
    for _ in range(60):
        ctrl = _random_controller(rng)
        assert _explicit_verdicts(ctrl) == run_nuxmv(controller_to_smv(ctrl)), ctrl.pretty()


@pytest.mark.skipif(find_nuxmv() is None, reason="nuXmv binary not available")
def test_engine_both_on_ideal():
    assert verify_response(IDEAL, engine="both").score == 15


def test_smv_contains_every_spec():
    smv = controller_to_smv(Controller())
    for s in SPEC_IDS:
        assert f"LTLSPEC NAME {s} :=" in smv
    assert "JUSTICE" in smv


@pytest.mark.skipif(find_nuxmv() is None, reason="nuXmv binary not available")
def test_many_free_atoms_fall_back_to_nuxmv():
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda omicron".split()
    text = "\n".join(f"{i}. If the {w} lamp is lit, stop." for i, w in enumerate(words, 1)) + "\n13. Otherwise, go_straight."
    res = verify_response(text, engine="explicit")
    assert len(res.controller.unknown_atoms) > 10 and 0 <= res.score <= 15
