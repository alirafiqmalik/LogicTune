"""
GLM2FSA-lite: parse a language-model response into a finite-state controller.

The model is asked for a numbered list of rules of the form
``If <condition>, <action>.``. This module turns such text into a
:class:`Controller`: an ordered list of guarded rules over the environment
propositions in :mod:`logictune.environment`.

Execution semantics (used by the model checker and by the simulator):

* The controller is a one-state automaton with prioritized guarded
  transitions. At every time step the rules are checked top to bottom and the
  first rule whose guard holds sets the action.
* A step without a condition ("Stop.") has guard ``TRUE``. ``Otherwise`` /
  ``else`` clauses also have guard ``TRUE``.
* If no rule applies, the action is undefined. The model checker treats an
  undefined action as a nondeterministic choice of any action, so an
  incomplete controller cannot satisfy specifications by omission.
* A condition fragment that maps to no known proposition (for example
  "the road is wet") becomes a free Boolean atom ``u<k>``. The model checker
  explores both of its values.

Reference: Yang et al., "Automaton-Based Representations of Task Knowledge
from Generative Language Models" (GLM2FSA), and "Fine-Tuning Language Models
Using Formal Methods Feedback" (2023).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from .environment import ACTIONS, PROPOSITIONS

# ---------------------------------------------------------------------------
# Boolean expression AST
# ---------------------------------------------------------------------------


class Expr:
    def to_smv(self) -> str:
        raise NotImplementedError

    def to_text(self) -> str:
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError

    def atoms(self) -> List[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Const(Expr):
    value: bool

    def to_smv(self) -> str:
        return "TRUE" if self.value else "FALSE"

    def to_text(self) -> str:
        return "true" if self.value else "false"

    def to_json(self) -> dict:
        return {"op": "const", "value": self.value}

    def atoms(self) -> List[str]:
        return []


@dataclass(frozen=True)
class Var(Expr):
    name: str  # proposition name or unknown-atom id (u0, u1, ...)

    def to_smv(self) -> str:
        return self.name

    def to_text(self) -> str:
        return self.name

    def to_json(self) -> dict:
        return {"op": "var", "name": self.name}

    def atoms(self) -> List[str]:
        return [self.name]


@dataclass(frozen=True)
class Not(Expr):
    arg: Expr

    def to_smv(self) -> str:
        return f"!{_wrap(self.arg)}"

    def to_text(self) -> str:
        return f"not {_wrap_text(self.arg)}"

    def to_json(self) -> dict:
        return {"op": "not", "arg": self.arg.to_json()}

    def atoms(self) -> List[str]:
        return self.arg.atoms()


@dataclass(frozen=True)
class And(Expr):
    args: Tuple[Expr, ...]

    def to_smv(self) -> str:
        return " & ".join(_wrap(a) for a in self.args)

    def to_text(self) -> str:
        return " and ".join(_wrap_text(a) for a in self.args)

    def to_json(self) -> dict:
        return {"op": "and", "args": [a.to_json() for a in self.args]}

    def atoms(self) -> List[str]:
        return [x for a in self.args for x in a.atoms()]


@dataclass(frozen=True)
class Or(Expr):
    args: Tuple[Expr, ...]

    def to_smv(self) -> str:
        return " | ".join(_wrap(a) for a in self.args)

    def to_text(self) -> str:
        return " or ".join(_wrap_text(a) for a in self.args)

    def to_json(self) -> dict:
        return {"op": "or", "args": [a.to_json() for a in self.args]}

    def atoms(self) -> List[str]:
        return [x for a in self.args for x in a.atoms()]


def _wrap(e: Expr) -> str:
    return f"({e.to_smv()})" if isinstance(e, (And, Or)) else e.to_smv()


def _wrap_text(e: Expr) -> str:
    return f"({e.to_text()})" if isinstance(e, (And, Or)) else e.to_text()


TRUE = Const(True)


def mk_and(items: Sequence[Expr]) -> Expr:
    flat: List[Expr] = []
    for it in items:
        if isinstance(it, Const):
            if not it.value:
                return Const(False)
            continue
        if isinstance(it, And):
            flat.extend(it.args)
        else:
            flat.append(it)
    flat = _dedupe(flat)
    if not flat:
        return TRUE
    return flat[0] if len(flat) == 1 else And(tuple(flat))


def mk_or(items: Sequence[Expr]) -> Expr:
    flat: List[Expr] = []
    for it in items:
        if isinstance(it, Const):
            if it.value:
                return TRUE
            continue
        if isinstance(it, Or):
            flat.extend(it.args)
        else:
            flat.append(it)
    flat = _dedupe(flat)
    if not flat:
        return Const(False)
    return flat[0] if len(flat) == 1 else Or(tuple(flat))


def mk_not(e: Expr) -> Expr:
    if isinstance(e, Const):
        return Const(not e.value)
    if isinstance(e, Not):
        return e.arg
    return Not(e)


def _dedupe(items: List[Expr]) -> List[Expr]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def expr_from_json(d: dict) -> Expr:
    op = d["op"]
    if op == "const":
        return Const(bool(d["value"]))
    if op == "var":
        return Var(d["name"])
    if op == "not":
        return Not(expr_from_json(d["arg"]))
    if op == "and":
        return And(tuple(expr_from_json(a) for a in d["args"]))
    if op == "or":
        return Or(tuple(expr_from_json(a) for a in d["args"]))
    raise ValueError(f"unknown op {op}")


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    guard: Expr
    action: str
    text: str  # source clause

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "guard": self.guard.to_json(),
            "guard_text": self.guard.to_text(),
            "guard_smv": self.guard.to_smv(),
            "action": self.action,
        }


@dataclass
class Controller:
    rules: List[Rule] = field(default_factory=list)
    unknown_atoms: Dict[str, str] = field(default_factory=dict)  # id -> source phrase

    @property
    def is_empty(self) -> bool:
        return not self.rules

    def to_json(self) -> dict:
        return {
            "rules": [r.to_json() for r in self.rules],
            "unknown_atoms": dict(self.unknown_atoms),
            "semantics": "first matching rule fires; undefined if none",
        }

    @classmethod
    def from_json(cls, d: dict) -> "Controller":
        rules = [Rule(expr_from_json(r["guard"]), r["action"], r.get("text", "")) for r in d["rules"]]
        return cls(rules=rules, unknown_atoms=dict(d.get("unknown_atoms", {})))

    def pretty(self) -> str:
        if not self.rules:
            return "(no rules: action undefined in every state)"
        lines = []
        for i, r in enumerate(self.rules, 1):
            lines.append(f"{i:>2}. if {r.guard.to_text()} -> {r.action}")
        for k, v in self.unknown_atoms.items():
            lines.append(f"    {k} = free atom for '{v}'")
        return "\n".join(lines)

    def to_networkx(self) -> nx.DiGraph:
        """One-state FSA with prioritized guarded self-loops (for plotting)."""
        g = nx.DiGraph()
        g.add_node("q0")
        for i, r in enumerate(self.rules, 1):
            g.add_edge("q0", "q0", key=i, guard=r.guard.to_text(), action=r.action, priority=i)
        return g


# ---------------------------------------------------------------------------
# Condition parsing
# ---------------------------------------------------------------------------

P = {name: name for name in PROPOSITIONS}

# Macro tokens expand to compound expressions.
_ANY_PED = mk_or([Var("pedestrian_in_front"), Var("pedestrian_at_left"), Var("pedestrian_at_right")])
_CROSS = mk_or([Var("car_from_left"), Var("car_from_right")])
_ANY_CAR = mk_or([Var("opposite_car"), Var("car_from_left"), Var("car_from_right")])
_CLEAR = mk_not(mk_or([_ANY_CAR, _ANY_PED]))
MACROS: Dict[str, Expr] = {
    "any_pedestrian": _ANY_PED,
    "cross_traffic": _CROSS,
    "any_car": _ANY_CAR,
    "clear": _CLEAR,
    "red_light": mk_not(Var("green_traffic_light")),
}

_LIGHT = r"(?:traffic\s+)?(?:light|signal)"
_IS = r"(?:is|turns|shows|becomes|changes\s+to|has\s+turned|is\s+now)"

# (pattern, replacement). Order matters: specific and negated forms first.
_PHRASES: List[Tuple[str, str]] = [
    # explicit identifiers
    (r"\bgreen_left_turn_light\b", " §green_left_turn_light§ "),
    (r"\bgreen_traffic_light\b", " §green_traffic_light§ "),
    (r"\bopposite_car\b", " §opposite_car§ "),
    (r"\bcar_from_left\b", " §car_from_left§ "),
    (r"\bcar_from_right\b", " §car_from_right§ "),
    (r"\bpedestrian_at_left\b", " §pedestrian_at_left§ "),
    (r"\bpedestrian_at_right\b", " §pedestrian_at_right§ "),
    (r"\bpedestrian_in_front\b", " §pedestrian_in_front§ "),
    (r"\bside_car\b", " §side_car§ "),
    (r"\bstop_sign\b", " §stop_sign§ "),
    # negated light forms
    (rf"\b(?:left[- ]turn\s+)?(?:arrow|left[- ]turn\s+{_LIGHT})\s+(?:is\s+not|isn't|is\s+off|is\s+not\s+on|is\s+not\s+green)\b", " not §green_left_turn_light§ "),
    (r"\bno\s+(?:green\s+|protected\s+)?(?:left[- ]turn\s+)?(?:arrow|left[- ]turn\s+(?:light|signal|arrow))\b", " not §green_left_turn_light§ "),
    (rf"\b{_LIGHT}\s+(?:is\s+not|isn't|is\s+no\s+longer)\s+green\b", " not §green_traffic_light§ "),
    (r"\b(?:no|not\s+a|non-?)\s*green\s+(?:traffic\s+)?(?:light|signal)\b", " not §green_traffic_light§ "),
    # positive light forms
    (r"\bgreen[ _-]left[ _-]turn[ _-](?:light|arrow|signal)\b", " §green_left_turn_light§ "),
    (r"\bgreen\s+(?:left\s+)?arrow\b", " §green_left_turn_light§ "),
    (rf"\bleft[- ]turn\s+(?:arrow|light|signal)\s+{_IS}\s+(?:green|on|lit)\b", " §green_left_turn_light§ "),
    (r"\bprotected\s+left(?:[- ]turn)?\b", " §green_left_turn_light§ "),
    (r"\bleft[- ]turn\s+(?:arrow|signal)\b|\bleft\s+arrow\b", " §green_left_turn_light§ "),
    (r"\bgreen\s+traffic\s+(?:light|signal)\b", " §green_traffic_light§ "),
    (r"\bgreen\s+(?:light|signal)\b", " §green_traffic_light§ "),
    (rf"\b{_LIGHT}\s+{_IS}\s+green\b", " §green_traffic_light§ "),
    (rf"\b{_LIGHT}\s+{_IS}\s+(?:red|yellow|amber)\b", " §red_light§ "),
    (rf"\b(?:red|yellow|amber)\s+{_LIGHT}\b", " §red_light§ "),
    (r"\b(?:red|yellow|amber)\b", " §red_light§ "),
    (r"\bgreen\b", " §green_traffic_light§ "),
    # cars
    (r"\b(?:cars?|vehicles?|traffic)\s+(?:is\s+|are\s+)?(?:approaching\s+|coming\s+)?from\s+(?:the\s+)?(?:left\s+or\s+(?:the\s+)?right|right\s+or\s+(?:the\s+)?left|either\s+(?:side|direction))\b", " §cross_traffic§ "),
    (r"\bcross[- ]?traffic\b|\bcrossing\s+(?:traffic|vehicles|cars)\b", " §cross_traffic§ "),
    (r"\b(?:oncoming|opposing|opposite)\s+(?:direction\s+)?(?:cars?|vehicles?|traffic)\b|\b(?:cars?|vehicles?)\s+(?:coming\s+)?from\s+the\s+opposite\s+direction\b", " §opposite_car§ "),
    (r"\b(?:cars?|vehicles?|traffic)\s+(?:is\s+|are\s+)?(?:approaching\s+|coming\s+)?from\s+(?:the\s+)?left\b|\b(?:car|vehicle)\s+(?:on|to)\s+the\s+left\b", " §car_from_left§ "),
    (r"\b(?:cars?|vehicles?|traffic)\s+(?:is\s+|are\s+)?(?:approaching\s+|coming\s+)?from\s+(?:the\s+)?right\b|\b(?:car|vehicle)\s+(?:on|to)\s+the\s+right\b", " §car_from_right§ "),
    (r"\bside\s+(?:car|vehicle)\b|\b(?:car|vehicle)\s+(?:beside|alongside|in\s+the\s+adjacent\s+lane)\b", " §side_car§ "),
    # pedestrians
    (r"\bpedestrians?\s+(?:is\s+|are\s+)?(?:directly\s+)?(?:in\s+front|ahead|in\s+the\s+(?:path|crosswalk(?:\s+ahead)?)|crossing\s+(?:in\s+front|ahead|the\s+(?:road|street)(?:\s+ahead)?))\b", " §pedestrian_in_front§ "),
    (r"\bpedestrians?\s+(?:is\s+|are\s+)?(?:on|at|to)\s+the\s+left\b", " §pedestrian_at_left§ "),
    (r"\bpedestrians?\s+(?:is\s+|are\s+)?(?:on|at|to)\s+the\s+right\b", " §pedestrian_at_right§ "),
    (r"\bpedestrians?\b|\bpeople\s+crossing\b|\bsomeone\s+(?:crossing|walking)\b", " §any_pedestrian§ "),
    (r"\bstop\s+sign\b", " §stop_sign§ "),
    # clear / safe
    (r"\b(?:intersection|road|path|way|crosswalk|lane)\s+is\s+(?:clear|free|empty)\b", " §clear§ "),
    (r"\b(?:it\s+is|it's)\s+safe(?:\s+to\s+\w+)?\b|\bsafe\s+to\s+(?:proceed|go|turn|cross|move)\b|\ball\s+clear\b|\bclear\b", " §clear§ "),
    (r"\bno\s+(?:other\s+)?(?:traffic|obstacles?|hazards?)\b", " not §any_car§ "),
    (r"\b(?:other|oncoming)\s+(?:cars?|vehicles?)\b|\banother\s+(?:car|vehicle)\b|\b(?:a|any)\s+(?:car|vehicle)\b|\bcars\b|\bvehicles\b|\btraffic\b", " §any_car§ "),
]
_PHRASES_C = [(re.compile(p, re.IGNORECASE), r) for p, r in _PHRASES]

_NEG_CUES = re.compile(
    r"\b(?:no|not|none|nobody|never|without|absent|isn't|aren't|there's\s+no|has\s+(?:crossed|passed|cleared|left)|have\s+(?:crossed|passed|cleared|left)|is\s+gone|are\s+gone|cleared|false)\b|n't\b"
)
_FILLER = set(
    """there is are the a an it its be been being present detected exists exist visible any some
    currently also still you we i at in on of to has have this that if when then car vehicle ego
    autonomous intersection approaching approach light lights signal state condition conditions
    true = == equals should must does do did having wants want""".split()
)
_OPS = {"and", "or", "not", "no", "neither", "nor", "(", ")"}


def _normalize_condition(text: str) -> str:
    s = text.lower()
    s = s.replace("∧", " and ").replace("∨", " or ").replace("¬", " not ")
    s = s.replace("&&", " and ").replace("||", " or ").replace("&", " and ").replace("|", " or ")
    s = re.sub(r"!(?=\s*[a-z(§])", " not ", s)
    s = re.sub(r"[\"'`*]", " ", s.replace("’", "'"))
    for pat, rep in _PHRASES_C:
        s = pat.sub(rep, s)
    # X = false / X is false  ->  not X ; X = true -> X
    s = re.sub(r"(§\w+§)\s*(?:==?|is)\s*(?:false|0)\b", r" not \1 ", s)
    s = re.sub(r"(§\w+§)\s*(?:==?|is)\s*(?:true|1)\b", r" \1 ", s)
    s = re.sub(r"\bwithout\b", " and no ", s)
    s = re.sub(r"\b(?:but|while|with|plus|as\s+well\s+as)\b", " and ", s)
    if "," in s:
        s = s.replace(",", " or " if (" or " in s and " and " not in s) else " and ")
    s = re.sub(r"[.;:]", " ", s)
    s = s.replace("(", " ( ").replace(")", " ) ")
    return re.sub(r"\s+", " ", s).strip()


class _CondParser:
    def __init__(self, tokens: List[str], unknown: Dict[str, str]):
        self.toks = tokens
        self.i = 0
        self.unknown = unknown  # phrase -> id (shared across a controller)

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self) -> Optional[Expr]:
        e = self.expr()
        return e

    def expr(self) -> Optional[Expr]:
        items = [self.term()]
        while self.peek() in ("or", "nor"):
            self.take()
            items.append(self.term())
        items = [x for x in items if x is not None]
        return mk_or(items) if items else None

    def term(self) -> Optional[Expr]:
        items = [self.factor()]
        while self.peek() is not None and self.peek() not in ("or", "nor", ")"):
            if self.peek() == "and":
                self.take()
            items.append(self.factor())  # explicit "and", or two adjacent factors
        items = [x for x in items if x is not None]
        return mk_and(items) if items else None

    def factor(self) -> Optional[Expr]:
        t = self.peek()
        if t is None:
            return None
        if t == "not":
            self.take()
            inner = self.factor()
            return mk_not(inner) if inner is not None else None
        if t in ("no", "neither"):
            # "no A or B" / "neither A nor B" -> not (A or B)
            self.take()
            items = [self.chunk()]
            while self.peek() in ("or", "nor") and self._next_is_chunk():
                self.take()
                items.append(self.chunk())
            items = [x for x in items if x is not None]
            return mk_not(mk_or(items)) if items else None
        if t == "(":
            self.take()
            e = self.expr()
            if self.peek() == ")":
                self.take()
            return e
        if t == ")":
            self.take()
            return None
        return self.chunk()

    def _next_is_chunk(self) -> bool:
        j = self.i + 1
        return j < len(self.toks) and self.toks[j] not in _OPS

    def chunk(self) -> Optional[Expr]:
        words = []
        while self.peek() is not None and self.peek() not in _OPS:
            words.append(self.take())
        if not words:
            return None
        text = " ".join(words)
        markers = re.findall(r"§(\w+)§", text)
        rest = re.sub(r"§\w+§", " ", text)
        negated = bool(_NEG_CUES.search(rest))
        if markers:
            parts = [MACROS[m] if m in MACROS else Var(m) for m in markers]
            e = mk_and(parts)
            return mk_not(e) if negated else e
        content = [w for w in re.findall(r"[a-z_']+", rest) if w not in _FILLER]
        content = [w for w in content if not _NEG_CUES.fullmatch(w)]
        if not content:
            return None
        key = " ".join(content)
        if key not in self.unknown:
            self.unknown[key] = f"u{len(self.unknown)}"
        e = Var(self.unknown[key])
        return mk_not(e) if negated else e


def parse_condition(text: str, unknown: Optional[Dict[str, str]] = None) -> Expr:
    """Parse a natural-language or formal condition into an expression."""
    unknown = {} if unknown is None else unknown
    tokens = _normalize_condition(text).split()
    e = _CondParser(tokens, unknown).parse()
    return TRUE if e is None else e


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_NON_ACTIONS = re.compile(
    r"\bstop\s+(?:sign|line|bar)s?\b|\b(?:prepare|be\s+prepared|be\s+ready|ready)\s+to\s+stop\b|\bslow(?:ing)?\s+down\b|"
    r"\bstopping\s+distance\b|\bnon-?stop\b",
    re.IGNORECASE,
)
_NEG_MOVE = re.compile(
    r"\b(?:do\s+not|don't|never|must\s+not|mustn't|should\s+not|shouldn't|cannot|can't|may\s+not|not)\s+"
    r"(?:\w+\s+){0,2}?(?:go|proceed|turn|enter|move|drive|cross|continue|accelerate)\w*",
    re.IGNORECASE,
)
_NEG_STOP = re.compile(r"\b(?:do\s+not|don't|never|no\s+need\s+to|without)\s+(?:\w+\s+)?stop\w*", re.IGNORECASE)

_ACTION_PATTERNS: List[Tuple[str, str]] = [
    ("turn_left", r"\bturn_left\b|\bturn(?:s|ing)?\s+(?:to\s+the\s+)?left\b|\bleft\s+turn\b|\bmake\s+(?:a|the)\s+left\b|\bgo\s+left\b"),
    ("turn_right", r"\bturn_right\b|\bturn(?:s|ing)?\s+(?:to\s+the\s+)?right\b|\bright\s+turn\b|\bmake\s+(?:a|the)\s+right\b|\bgo\s+right\b"),
    ("go_straight", r"\bgo_straight\b|\b(?:go|goes|going|proceed|proceeds|continue|continues|drive|drives|move|moves)\s+(?:\w+\s+){0,2}?straight\b|\bstraight\s+(?:ahead|through)\b"),
    ("stop", r"\bstop(?:s|ped|ping)?\b|\bwait(?:s|ing)?\b|\bhalt(?:s)?\b|\byield(?:s)?\b|\bbrake(?:s)?\b|\bremain\s+(?:stopped|stationary)\b|\bstay\s+(?:stopped|put)\b|\bcome\s+to\s+a\s+(?:complete\s+|full\s+)?stop\b"),
    ("go", r"\bgo(?:es)?\b|\bproceed(?:s)?\b|\bcontinue(?:s)?\b|\bdrive\s+(?:through|forward|on|ahead)\b|\bmove\s+(?:forward|ahead|through|on)\b|\baccelerate(?:s)?\b|\bcross\s+the\s+intersection\b|\benter\s+the\s+intersection\b"),
]
_ACTION_PATTERNS_C = [(a, re.compile(p, re.IGNORECASE)) for a, p in _ACTION_PATTERNS]


def _action_start(text: str) -> Optional[int]:
    """Character offset of the first action phrase in ``text`` (or None)."""
    s = _NON_ACTIONS.sub(lambda m: " " * len(m.group(0)), text)
    starts = [m.start() for m in _NEG_MOVE.finditer(s)]
    starts += [m.start() for _, pat in _ACTION_PATTERNS_C for m in [pat.search(s)] if m]
    return min(starts) if starts else None


def parse_action(text: str) -> Optional[str]:
    """Return the canonical action named in ``text`` (or None)."""
    s = _NON_ACTIONS.sub(" ", text)
    candidates: List[Tuple[int, str]] = []
    for m in _NEG_MOVE.finditer(s):
        candidates.append((m.start(), "stop"))
    s_masked = _NEG_MOVE.sub(lambda m: " " * len(m.group(0)), s)
    s_masked = _NEG_STOP.sub(lambda m: " " * len(m.group(0)), s_masked)
    for action, pat in _ACTION_PATTERNS_C:
        m = pat.search(s_masked)
        if m:
            candidates.append((m.start(), action))
    if not candidates:
        return None
    candidates.sort()
    first = candidates[0][1]
    if first == "go":
        specific = [a for _, a in candidates if a in ("turn_left", "turn_right", "go_straight")]
        return specific[0] if specific else "go_straight"
    return first


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

_LIST_MARKER = re.compile(r"^\s*(?:(?:rule|step)\s*\d+\s*[:.)\-]?|(?:rule|step)\s*[:.)\-]|\d+\s*[.):\-]|[-*•>]|\(\w\))\s*", re.IGNORECASE)
_COND_HEAD = r"(?:if|when|whenever|while|once|in\s+case|as\s+long\s+as|provided\s+(?:that)?|in\s+the\s+event\s+(?:that|of))"
_SPLITTERS = [r",?\s+then\s+", r"\s*(?:->|=>|→)\s*", r"\s*:\s+", r",\s*"]


_BLOCK_END = object()


def _lines(response: str):
    """Yield (indent, text) per non-empty line and _BLOCK_END for blank lines."""
    text = re.sub(r"\*\*|__|`", "", response.replace("\r", "\n")).replace("\t", "    ")
    for raw in text.split("\n"):
        if not raw.strip():
            yield _BLOCK_END
            continue
        indent = len(raw) - len(raw.lstrip())
        if re.match(r"^\s*[-*•>]\s", raw):
            indent += 1  # a bullet is a child of a header at the same column
        line = _LIST_MARKER.sub("", raw.strip(), count=1).strip()
        line = _LIST_MARKER.sub("", line, count=1).strip()
        if line:
            yield indent, line


def _sentences(line: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])|;\s*|\.\s+(?=otherwise|else)|,\s*(?=otherwise\b|else\b)", line)
    return [p.strip().rstrip(".!").strip() for p in parts if p.strip().rstrip(".!").strip()]


def _parse_conditional(clause: str) -> Optional[Tuple[str, str]]:
    """Split "If <cond>, <action>" at the best separator.

    Preferred split: the earliest one where the condition names no action and
    the remainder starts with an action (within its first eight words). This
    keeps comma-separated condition lists ("If a, b, c, then stop") and
    comma-separated action lists ("If a, stop and wait, then go") apart.
    """
    m = re.match(rf"^{_COND_HEAD}\s+(.+)$", clause, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    cands = []
    for sp in _SPLITTERS:
        for mm in re.finditer(sp, body):
            cond, act = body[: mm.start()], body[mm.end():]
            if cond.strip() and parse_action(act) is not None:
                cands.append((mm.start(), cond, act))
    if not cands:
        return None
    def starts_with_action(act: str) -> bool:
        pos = _action_start(act)
        if pos is None:
            return False
        prefix = act[:pos]
        return len(prefix.split()) <= 6 and "§" not in _normalize_condition(prefix)

    good = [c for c in cands if parse_action(c[1]) is None and starts_with_action(c[2])]
    pos, cond, act = min(good or cands, key=lambda c: c[0])
    return cond, act


def _header_condition(line: str) -> Optional[str]:
    """A condition line without an action, e.g. "If the light is green, then:"."""
    m = re.match(rf"^{_COND_HEAD}\s+(.+?)\s*(?:,?\s*then\s*:?|,|:)\s*$", line, re.IGNORECASE | re.DOTALL)
    if m and _parse_conditional(line) is None:
        return m.group(1)
    return None


def _parse_clause(clause: str, unknown: Dict[str, str]) -> Tuple[Optional[Expr], Optional[str]]:
    low = clause.lower()
    # 1. "If/When <cond>, <action>"
    ca = _parse_conditional(clause)
    if ca is not None:
        return parse_condition(ca[0], unknown), parse_action(ca[1])
    # 2. "Otherwise/Else <action>"
    if re.match(r"^(?:otherwise|else|in\s+all\s+other\s+cases|in\s+any\s+other\s+case|by\s+default|default)\b", low):
        return TRUE, parse_action(clause)
    # 3. "<cond> -> <action>" with formal identifiers
    m = re.match(r"^(.+?)\s*(?:->|=>|→|:)\s*(.+)$", clause)
    if m and re.search(r"§", _normalize_condition(m.group(1))) and parse_action(m.group(2)):
        return parse_condition(m.group(1), unknown), parse_action(m.group(2))
    # 4. "<action> if/when/unless/until <cond>"
    m = re.match(r"^(.+?)\s+(if|when|whenever|as\s+long\s+as|only\s+if|unless|until|while)\s+(.+)$", clause, re.IGNORECASE)
    if m and parse_action(m.group(1)) and len(m.group(1).split()) <= 10:
        cond = parse_condition(m.group(3), unknown)
        guard = mk_not(cond) if m.group(2).lower() in ("unless", "until") else cond
        return guard, parse_action(m.group(1))
    # 5. Unconditional imperative ("Stop.", "Turn right.")
    if re.match(r"^(?:turn_left|turn_right|go_straight|stop|wait|halt|yield|go|proceed|continue|turn|drive|remain|stay|come\s+to\s+a)\b", low):
        return TRUE, parse_action(clause)
    return None, None


def extract_rules(response: str) -> Controller:
    """
    Parse a model response into a :class:`Controller`.

    Nested conditions are supported: a line that is only a condition
    ("If A," / "When A:") opens a block, and rules inside the block (indented
    lines, a following "then ..." line, or lines up to the next blank line)
    get the conjunction of the enclosing conditions.
    """
    unknown: Dict[str, str] = {}
    rules: List[Rule] = []
    stack: List[Tuple[int, Expr]] = []
    prev_header_indent: Optional[int] = None
    for item in _lines(response):
        if item is _BLOCK_END:
            stack.clear()
            prev_header_indent = None
            continue
        indent, line = item
        then_m = re.match(r"^then\b[\s,:]*", line, re.IGNORECASE)
        if then_m and not line[then_m.end():].strip():
            continue  # bare "then:" keeps the open header for the next (indented) line
        header = None if then_m else _header_condition(line)
        chained = header is not None and prev_header_indent is not None and indent == prev_header_indent
        if not then_m and not chained:
            while stack and stack[-1][0] >= indent:
                stack.pop()
        if header is not None:
            stack.append((indent, parse_condition(header, unknown)))
            prev_header_indent = indent
            continue
        prev_header_indent = None
        context = [c for _, c in stack]
        body = line[then_m.end():] if then_m else line
        for clause in _sentences(body):
            guard, action = _parse_clause(clause, unknown)
            if then_m and action is None:
                action, guard = parse_action(clause), TRUE
            if action in ACTIONS and guard is not None:
                full = mk_and(context + [guard])
                src = clause if not context else f"[{' & '.join(c.to_text() for c in context)}] {clause}"
                rules.append(Rule(guard=full, action=action, text=src[:300]))
        if then_m and stack:
            stack.pop()
    unknown_atoms = {v: k for k, v in unknown.items() if any(v in r.guard.atoms() for r in rules)}
    return Controller(rules=rules, unknown_atoms=unknown_atoms)


def parse_response_to_fsa(llm_text_response: str, verbose: bool = False) -> nx.DiGraph:
    """Compatibility wrapper: returns the controller as a networkx FSA."""
    ctrl = extract_rules(llm_text_response)
    if verbose:
        print(ctrl.pretty())
    g = ctrl.to_networkx()
    g.graph["controller"] = ctrl
    g.graph["raw_response"] = llm_text_response
    return g
