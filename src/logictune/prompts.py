"""
Prompts, scenarios, and the train / evaluation split.

All stages (data generation, DPO training, evaluation, the simulator) use the
same system prompt and the same chat template, so the model is trained on
exactly the text it sees at evaluation time.

Split:

* 17 driving scenarios (``SCENARIOS``) define the task family.
* Scenarios 4, 8, 12 and 16 are held out: no training prompt mentions them.
* Training prompts: ``prompts_200.json`` (12 instruction templates x scenario)
  and ``prompts.json`` (5 paraphrases x scenario), minus held-out scenarios.
* Evaluation prompts: two instruction templates that never appear in
  training, applied to all 17 scenarios.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .environment import ACTIONS, PROPOSITIONS

_HERE = Path(__file__).parent

SYSTEM_PROMPT = (
    "You write controllers for an autonomous car at a road intersection.\n"
    "Write the controller as a numbered list of rules. Each rule has the form:\n"
    "If <condition>, <action>.\n"
    "A condition may only use these Boolean propositions, combined with and, or, not: "
    + ", ".join(PROPOSITIONS)
    + ".\n"
    "Each action is exactly one of: " + ", ".join(ACTIONS) + ".\n"
    "The rules are checked from top to bottom, and the first rule whose condition is true decides the action."
)


@dataclass(frozen=True)
class Scenario:
    id: int
    text: str
    intent: str  # straight | left | right (used by the simulator)
    init: Dict[str, bool]  # initial proposition values for the simulator


SCENARIOS: List[Scenario] = [
    Scenario(1, "Green light but a pedestrian is in front while the car wants to go straight.", "straight",
             {"green_traffic_light": True, "pedestrian_in_front": True}),
    Scenario(2, "Green light but a pedestrian is on the right while the car wants to turn right.", "right",
             {"green_traffic_light": True, "pedestrian_at_right": True}),
    Scenario(3, "Green light but a pedestrian is on the left while the car wants to turn left.", "left",
             {"green_traffic_light": True, "pedestrian_at_left": True}),
    Scenario(4, "No green left-turn arrow and an opposite car is present while the car wants to turn left.", "left",
             {"green_traffic_light": True, "opposite_car": True}),
    Scenario(5, "No green left-turn arrow and a car is coming from the left or right while the car wants to turn left.", "left",
             {"green_traffic_light": True, "car_from_left": True}),
    Scenario(6, "No green left-turn arrow and the intersection is clear while the car wants to turn left (gap acceptance).", "left",
             {"green_traffic_light": True}),
    Scenario(7, "Green left-turn arrow is on and the car wants to turn left (protected left turn).", "left",
             {"green_traffic_light": True, "green_left_turn_light": True}),
    Scenario(8, "Red light and the car wants to turn right with no car from the left and no pedestrian on the right.", "right",
             {}),
    Scenario(9, "A car from the left or a pedestrian at the right blocks a right turn.", "right",
             {"green_traffic_light": True, "car_from_left": True}),
    Scenario(10, "Stop sign with cross-traffic present from left or right.", "straight",
             {"stop_sign": True, "car_from_left": True}),
    Scenario(11, "Stop sign with no cross-traffic present.", "straight",
             {"stop_sign": True}),
    Scenario(12, "Stop sign when turning right and cars/pedestrians may appear on the right or from the left.", "right",
             {"stop_sign": True, "car_from_left": True}),
    Scenario(13, "Red light while the car wants to go straight and must stop before proceeding.", "straight",
             {}),
    Scenario(14, "Green light but the car is currently stopped and must decide when to proceed.", "straight",
             {"green_traffic_light": True}),
    Scenario(15, "A side car is present during lane-keeping, merging, or turning near the intersection.", "straight",
             {"green_traffic_light": True, "side_car": True}),
    Scenario(16, "Uncontrolled intersection with no lights or stop sign but cross-traffic is present.", "straight",
             {"car_from_right": True}),
    Scenario(17, "Stop sign present simultaneously with an active traffic light signal (conflicting controls).", "straight",
             {"green_traffic_light": True, "stop_sign": True}),
]
SCENARIO_BY_ID = {s.id: s for s in SCENARIOS}
HELDOUT_SCENARIOS = (4, 8, 12, 16)

EVAL_TEMPLATES = [
    "Write the controller rules for this driving scenario: {scenario}",
    "Scenario: {scenario}\nGive the rule list the car should follow.",
]


@dataclass(frozen=True)
class PromptItem:
    prompt_id: str
    scenario_id: int
    text: str  # user message
    split: str  # train | eval

    def to_json(self) -> dict:
        return asdict(self)


def _scenario_of(text: str) -> Optional[int]:
    for s in SCENARIOS:
        if s.text in text:
            return s.id
    return None


def train_prompts(include_heldout: bool = False) -> List[PromptItem]:
    """Training prompts (held-out scenarios removed unless requested)."""
    items: List[PromptItem] = []
    p200 = json.loads((_HERE / "prompts_200.json").read_text())["prompts"]
    for i, text in enumerate(p200):
        sid = _scenario_of(text)
        if sid is None:
            raise ValueError(f"prompt without scenario: {text}")
        items.append(PromptItem(f"t200_{i:03d}", sid, text, "train"))
    p85 = json.loads((_HERE / "prompts.json").read_text())["prompts"]
    for i, text in enumerate(p85):
        sid = i // 5 + 1  # 5 paraphrases per scenario, in scenario order
        items.append(PromptItem(f"t85_{i:03d}", sid, text, "train"))
    if not include_heldout:
        items = [it for it in items if it.scenario_id not in HELDOUT_SCENARIOS]
    return items


def eval_prompts() -> List[PromptItem]:
    items = []
    for s in SCENARIOS:
        for t, tmpl in enumerate(EVAL_TEMPLATES):
            items.append(PromptItem(f"e{s.id:02d}_{t}", s.id, tmpl.format(scenario=s.text), "eval"))
    return items


def chat_prompt(tokenizer, user_message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Render the chat template exactly as used in generation and training."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Templates without a system role: fold the system prompt into the user turn.
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_message}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Callers tokenize with add_special_tokens=False, so the BOS token (if the
    # model uses one) must be part of the text.
    bos = getattr(tokenizer, "bos_token", None)
    if bos and not text.startswith(bos):
        text = bos + text
    return text


def scenarios_as_dicts() -> List[dict]:
    return [dict(asdict(s), heldout=s.id in HELDOUT_SCENARIOS) for s in SCENARIOS]
