"""Parser tests: natural-language and formal rules -> guarded actions."""

import pytest

from logictune.glm2fsa import Var, extract_rules, parse_action, parse_condition


@pytest.mark.parametrize(
    "text,expected",
    [
        ("stop", "stop"),
        ("come to a complete stop", "stop"),
        ("wait until the pedestrian has crossed", "stop"),
        ("proceed straight through the intersection", "go_straight"),
        ("go_straight", "go_straight"),
        ("proceed to turn left", "turn_left"),
        ("make a right turn", "turn_right"),
        ("do not turn left", "stop"),
        ("stop at the stop sign", "stop"),
        ("slow down and prepare to stop", None),
        ("the car checks its mirrors", None),
    ],
)
def test_parse_action(text, expected):
    assert parse_action(text) == expected


@pytest.mark.parametrize(
    "text,smv",
    [
        ("green_traffic_light", "green_traffic_light"),
        ("not green_traffic_light", "!green_traffic_light"),
        ("!green_traffic_light & !pedestrian_in_front", "!green_traffic_light & !pedestrian_in_front"),
        ("the light is red", "!green_traffic_light"),
        ("the light is green", "green_traffic_light"),
        ("there is a stop sign", "stop_sign"),
        ("a pedestrian is in front", "pedestrian_in_front"),
        ("car_from_left = false", "!car_from_left"),
        ("there is an oncoming car", "opposite_car"),
        ("cars are coming from the left or right", "car_from_left | car_from_right"),
        ("the green left-turn arrow is on", "green_left_turn_light"),
    ],
)
def test_parse_condition(text, smv):
    assert parse_condition(text).to_smv() == smv


def test_no_pedestrians_or_cars_is_a_negated_disjunction():
    e = parse_condition("there are no pedestrians or cars")
    assert e.to_smv().startswith("!(")
    for p in ("pedestrian_in_front", "opposite_car", "car_from_left"):
        assert p in e.atoms()


def test_unknown_phrase_becomes_free_atom():
    unknown = {}
    e = parse_condition("the road is wet", unknown)
    assert isinstance(e, Var) and e.name == "u0"
    assert unknown == {"road wet": "u0"}


def test_rule_order_and_otherwise():
    ctrl = extract_rules(
        "1. If there is a pedestrian in front, stop.\n"
        "2. If the light is green, go straight.\n"
        "3. Otherwise, stop."
    )
    assert [r.action for r in ctrl.rules] == ["stop", "go_straight", "stop"]
    assert ctrl.rules[0].guard.to_smv() == "pedestrian_in_front"
    assert ctrl.rules[2].guard.to_smv() == "TRUE"


def test_trailing_condition_and_unless():
    ctrl = extract_rules("Stop if the light is red. Go straight unless a pedestrian is in front.")
    assert ctrl.rules[0].action == "stop" and ctrl.rules[0].guard.to_smv() == "!green_traffic_light"
    assert ctrl.rules[1].action == "go_straight" and ctrl.rules[1].guard.to_smv() == "!pedestrian_in_front"


def test_prose_has_no_rules():
    assert extract_rules("I do not have access to a vehicle. The car is prepared to stop.").is_empty


def test_controller_json_round_trip():
    from logictune.glm2fsa import Controller

    ctrl = extract_rules("1. If not green_traffic_light, stop.\n2. Otherwise, go_straight.")
    again = Controller.from_json(ctrl.to_json())
    assert [r.guard for r in again.rules] == [r.guard for r in ctrl.rules]
