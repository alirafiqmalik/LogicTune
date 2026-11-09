"""
Environment model for the traffic-intersection verification task.

The environment is the set of Boolean atomic propositions that the
controller can observe, plus a fairness assumption. Every proposition is
free: at each time step the environment may choose any valuation. This is
the most adversarial environment that is still consistent with the
propositions in the paper.

Fairness (JUSTICE): infinitely often the intersection is clear and the
traffic light is green. Without this assumption no controller that stops for
pedestrians could satisfy the liveness specifications (phi_7, phi_10, phi_13),
because the environment could keep a pedestrian in front of the car forever.

Reference: Yang et al., "Fine-Tuning Language Models Using Formal Methods
Feedback: A Use Case in Autonomous Systems" (2023).
"""

from typing import Dict, List

# Boolean atomic propositions observed by the controller (order is fixed and
# shared with the SMV model, the explicit-state checker, and the simulator).
PROPOSITIONS: List[str] = [
    "green_traffic_light",
    "green_left_turn_light",
    "opposite_car",
    "car_from_left",
    "car_from_right",
    "pedestrian_at_left",
    "pedestrian_at_right",
    "pedestrian_in_front",
    "side_car",
    "stop_sign",
]

# Actions the controller can output.
ACTIONS: List[str] = ["stop", "go_straight", "turn_left", "turn_right"]

# Hazards that must be absent in a "clear" state.
HAZARDS: List[str] = [
    "opposite_car",
    "car_from_left",
    "car_from_right",
    "pedestrian_at_left",
    "pedestrian_at_right",
    "pedestrian_in_front",
    "side_car",
    "stop_sign",
]

# Fairness assumption: green light and no hazard, infinitely often.
FAIRNESS_ASSIGNMENT: Dict[str, bool] = {"green_traffic_light": True, **{h: False for h in HAZARDS}}


def fairness_smv() -> str:
    """Return the JUSTICE constraint as an SMV expression."""
    terms = []
    for name, value in FAIRNESS_ASSIGNMENT.items():
        terms.append(name if value else f"!{name}")
    return " & ".join(terms)


def describe_environment() -> Dict:
    """Machine-readable description of the environment (exported for the UI)."""
    return {
        "propositions": list(PROPOSITIONS),
        "actions": list(ACTIONS),
        "fairness": {
            "kind": "JUSTICE",
            "smv": fairness_smv(),
            "description": "Infinitely often the light is green and the intersection is clear.",
        },
        "transition_relation": "unconstrained (every valuation can follow every valuation)",
    }
