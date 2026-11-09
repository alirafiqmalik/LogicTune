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
    assert ctrl.rules[2].guard.to_smv() == "!green_traffic_light"  # else of the last if


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


def test_nested_conditions_by_indentation():
    ctrl = extract_rules(
        "If green_traffic_light and pedestrian_in_front,\n"
        "    If stop_sign, stop\n"
        "    Otherwise, go_straight\n"
        "If not green_traffic_light, stop"
    )
    assert [r.guard.to_smv() for r in ctrl.rules] == [
        "green_traffic_light & pedestrian_in_front & stop_sign",
        "green_traffic_light & pedestrian_in_front & !stop_sign",
        "!green_traffic_light",
    ]


def test_header_then_line_blocks():
    ctrl = extract_rules(
        "If green_left_turn_light is true,\nIf opposite_car is true,\nThen go_straight action.\n\n"
        "If opposite_car is false,\nThen stop action."
    )
    assert [(r.guard.to_smv(), r.action) for r in ctrl.rules] == [
        ("green_left_turn_light & opposite_car", "go_straight"),
        ("!opposite_car", "stop"),
    ]


def test_header_with_bullet_action():
    ctrl = extract_rules("If stop_sign:\n- stop\n\nOtherwise, go_straight")
    assert [(r.guard.to_smv(), r.action) for r in ctrl.rules] == [("stop_sign", "stop"), ("!stop_sign", "go_straight")]


def test_if_yes_is_not_always_true():
    ctrl = extract_rules("The controller checks the light. If yes, stop.")
    assert ctrl.rules[0].guard.to_smv() != "TRUE"


def test_comma_list_condition_then_action():
    ctrl = extract_rules("If green_traffic_light, pedestrian_in_front, stop_sign, then the car should stop.")
    assert ctrl.rules[0].guard.to_smv() == "green_traffic_light & pedestrian_in_front & stop_sign"


def test_comma_list_actions_after_condition():
    ctrl = extract_rules("If there is a pedestrian in front, stop and wait until the pedestrian has crossed, then proceed.")
    assert (ctrl.rules[0].guard.to_smv(), ctrl.rules[0].action) == ("pedestrian_in_front", "stop")


def test_bare_then_line_keeps_header():
    ctrl = extract_rules("If stop_sign and car_from_left,\nthen:\n    stop")
    assert (ctrl.rules[0].guard.to_smv(), ctrl.rules[0].action) == ("stop_sign & car_from_left", "stop")


def test_identifier_bullets_under_headers():
    ctrl = extract_rules(
        "1. If green_left_turn_light is true and a car is coming from the left, then:\n    - turn_left\n"
        "2. If a pedestrian is in front and the car wants to turn left, then:\n    - stop\n"
    )
    assert [r.action for r in ctrl.rules] == ["turn_left", "stop"]
    assert "pedestrian_in_front" in ctrl.rules[1].guard.atoms()


def test_rule_label_prefix():
    ctrl = extract_rules("Rule: If there is a red light, the car should stop.")
    assert (ctrl.rules[0].guard.to_smv(), ctrl.rules[0].action) == ("!green_traffic_light", "stop")


def _gs(ctrl):
    return [(r.guard.to_smv(), r.action) for r in ctrl.rules]


def test_if_then_else_blocks_without_punctuation():
    ctrl = extract_rules(
        "If (green_traffic_light AND opposite_car)\nThen:\n- Stop\n\nElse:\n- Go straight\n"
    )
    assert _gs(ctrl) == [
        ("green_traffic_light & opposite_car", "stop"),
        ("!(green_traffic_light & opposite_car)", "go_straight"),
    ]


def test_else_if_chain():
    ctrl = extract_rules(
        "If green_left_turn_light is true,\n  stop\n\n"
        "Else if opposite_car is true,\n  go_straight\n\n"
        "Else,\n  turn_right"
    )
    assert _gs(ctrl) == [
        ("green_left_turn_light", "stop"),
        ("!green_left_turn_light & opposite_car", "go_straight"),
        ("!(green_left_turn_light | opposite_car)", "turn_right"),
    ]


def test_if_colon_with_bullet_conditions():
    ctrl = extract_rules("If:\n- green_traffic_light\n- green_left_turn_light\n\nThen:\n- stop\n\nElse:\n- go_straight")
    assert _gs(ctrl) == [
        ("green_traffic_light & green_left_turn_light", "stop"),
        ("!(green_traffic_light & green_left_turn_light)", "go_straight"),
    ]


def test_code_style_braces():
    ctrl = extract_rules("if (pedestrian_in_front) {\n  stop();\n} else if (green_traffic_light) {\n  go_straight();\n} else {\n  stop();\n}")
    assert _gs(ctrl) == [
        ("pedestrian_in_front", "stop"),
        ("!pedestrian_in_front & green_traffic_light", "go_straight"),
        ("!(pedestrian_in_front | green_traffic_light)", "stop"),
    ]


def test_inline_rule_then_otherwise_line():
    ctrl = extract_rules("1. If stop_sign, stop.\n2. If green_traffic_light, go_straight.\n3. Otherwise, stop.")
    assert _gs(ctrl)[-1] == ("!green_traffic_light", "stop")


def test_example_blocks_are_ignored():
    ctrl = extract_rules("If pedestrian_in_front is true, the car should follow:\n\n- Stop\n\nExample:\nGreen light: true\n\n- Go straight\n")
    assert _gs(ctrl) == [("pedestrian_in_front", "stop")]


def test_chained_if_conditions_are_a_conjunction():
    ctrl = extract_rules("1. If not green_traffic_light or not green_left_turn_light, if opposite_car, if car_from_left, stop.\n"
                         "2. If not stop_sign, If green_traffic_light, go_straight.")
    assert _gs(ctrl) == [
        ("(!green_traffic_light | !green_left_turn_light) & opposite_car & car_from_left", "stop"),
        ("!stop_sign & green_traffic_light", "go_straight"),
    ]


def test_body_at_header_indentation_stays_in_the_block():
    # Answers like these were read as "always stop" before (reward hacking in round 2).
    ctrl = extract_rules("If\n\n- green_left_turn_light\n- opposite_car\n\nThen\n\n- stop")
    assert _gs(ctrl) == [("green_left_turn_light & opposite_car", "stop")]
    ctrl = extract_rules("When the light is red:\nStop at the line.\nWait for green.")
    assert [g for g, _ in _gs(ctrl)] == ["!green_traffic_light", "!green_traffic_light"]
    ctrl = extract_rules("If:\n- stop_sign\n\nThen:\n\nStop:\n- stop")
    assert all(g == "stop_sign" for g, _ in _gs(ctrl))


def test_numbered_siblings_and_deeper_bodies_close_blocks():
    ctrl = extract_rules("1. If stop_sign:\n   - stop\n2. Go straight.")
    assert _gs(ctrl) == [("stop_sign", "stop"), ("TRUE", "go_straight")]
    ctrl = extract_rules("If stop_sign:\n  - stop\nGo straight.")
    assert _gs(ctrl) == [("stop_sign", "stop"), ("TRUE", "go_straight")]


def test_condition_list_with_turn_light_nouns():
    ctrl = extract_rules("If\n\n1. Green traffic light\n2. Green left turn light\n3. Opposite car\n\nThen\n\n1. Stop")
    assert _gs(ctrl) == [("green_traffic_light & green_left_turn_light & opposite_car", "stop")]
    assert parse_action("check the left-turn arrow") is None


def test_action_vocabulary_lists_are_not_rules():
    ctrl = extract_rules("Actions:\n1. Stop\n2. Go straight ahead\n3. Turn left\n4. Turn right\n\nIf pedestrian_in_front, stop.")
    assert _gs(ctrl) == [("pedestrian_in_front", "stop")]
    ctrl = extract_rules("If pedestrian_in_front, stop.\nOtherwise, go_straight.\nStop.")  # one bare action is a rule
    assert _gs(ctrl)[-1] == ("TRUE", "stop")


def test_inline_rule_with_colon_opens_a_block():
    ctrl = extract_rules("If there is no cross-traffic present, stop:\n\n1. Stop\n2. Wait")
    assert all(g == "!(car_from_left | car_from_right)" for g, _ in _gs(ctrl))
