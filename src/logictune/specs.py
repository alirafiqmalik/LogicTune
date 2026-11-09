"""
The 15 LTL safety and liveness specifications (phi_1 .. phi_15).

Each specification is given in nuXmv LTL syntax over the environment
propositions and the controller output ``action``. ``action_defined`` is true
when at least one rule of the controller applies in the current state.

phi_6 in the paper is "always one of stop / go straight / turn left / turn
right". Because ``action`` ranges over exactly these values, a literal
encoding is a tautology. We encode the intended meaning instead: the
controller always prescribes an action (no state is left unspecified).
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass(frozen=True)
class LTLSpec:
    id: str
    formula: str  # mathematical notation, for humans
    smv: str  # nuXmv syntax
    description: str
    kind: str  # "safety" or "liveness"


SPECS: List[LTLSpec] = [
    LTLSpec("phi_1", "□(pedestrian_in_front → ◇ stop)",
            "G (pedestrian_in_front -> F (action = stop))",
            "If a pedestrian is in front, the car eventually stops.", "liveness"),
    LTLSpec("phi_2", "□(opposite_car ∧ ¬green_left_turn_light → ¬turn_left)",
            "G ((opposite_car & !green_left_turn_light) -> !(action = turn_left))",
            "No unprotected left turn with an oncoming car.", "safety"),
    LTLSpec("phi_3", "□(¬green_traffic_light → ¬go_straight)",
            "G (!green_traffic_light -> !(action = go_straight))",
            "Never go straight unless the light is green.", "safety"),
    LTLSpec("phi_4", "□(stop_sign → ◇ stop)",
            "G (stop_sign -> F (action = stop))",
            "At a stop sign, the car eventually stops.", "liveness"),
    LTLSpec("phi_5", "□(car_from_left ∨ pedestrian_at_right → ¬turn_right)",
            "G ((car_from_left | pedestrian_at_right) -> !(action = turn_right))",
            "No right turn with a car from the left or a pedestrian on the right.", "safety"),
    LTLSpec("phi_6", "□(stop ∨ go_straight ∨ turn_left ∨ turn_right)",
            "G action_defined",
            "The controller always prescribes one of the four actions.", "safety"),
    LTLSpec("phi_7", "◇(green_traffic_light ∨ green_left_turn_light) → ◇¬stop",
            "(F (green_traffic_light | green_left_turn_light)) -> F !(action = stop)",
            "If a green light eventually shows, the car eventually moves.", "liveness"),
    LTLSpec("phi_8", "□(¬green_traffic_light → ◇ stop)",
            "G (!green_traffic_light -> F (action = stop))",
            "If the light is not green, the car eventually stops.", "liveness"),
    LTLSpec("phi_9", "□(car_from_left → ¬(turn_left ∨ turn_right))",
            "G (car_from_left -> !((action = turn_left) | (action = turn_right)))",
            "No turns while a car comes from the left.", "safety"),
    LTLSpec("phi_10", "□(green_traffic_light → ◇¬stop)",
            "G (green_traffic_light -> F !(action = stop))",
            "After a green light, the car eventually moves.", "liveness"),
    LTLSpec("phi_11", "□(turn_right ∧ ¬green_traffic_light → ¬car_from_left)",
            "G (((action = turn_right) & !green_traffic_light) -> !car_from_left)",
            "Right turn on a non-green light only with no car from the left.", "safety"),
    LTLSpec("phi_12", "□(turn_left ∧ ¬green_left_turn_light → ¬car_from_right ∧ ¬car_from_left ∧ ¬opposite_car)",
            "G (((action = turn_left) & !green_left_turn_light) -> (!car_from_right & !car_from_left & !opposite_car))",
            "Unprotected left turn only when no car comes from any direction.", "safety"),
    LTLSpec("phi_13", "□(stop_sign ∧ ¬car_from_left ∧ ¬car_from_right → ◇¬stop)",
            "G ((stop_sign & !car_from_left & !car_from_right) -> F !(action = stop))",
            "At a stop sign with no cross traffic, the car eventually moves.", "liveness"),
    LTLSpec("phi_14", "□(go_straight → ¬pedestrian_in_front)",
            "G ((action = go_straight) -> !pedestrian_in_front)",
            "Never go straight into a pedestrian.", "safety"),
    LTLSpec("phi_15", "□(turn_right ∧ stop_sign → ¬car_from_left)",
            "G (((action = turn_right) & stop_sign) -> !car_from_left)",
            "Right turn at a stop sign only with no car from the left.", "safety"),
]

SPEC_IDS: List[str] = [s.id for s in SPECS]
NUM_SPECS: int = len(SPECS)


def specs_as_dicts() -> List[dict]:
    return [asdict(s) for s in SPECS]
