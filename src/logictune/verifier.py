"""
Formal verification of language-model controllers.

A response is parsed into a :class:`~logictune.glm2fsa.Controller`, the
controller is composed with the environment model, and the composition is
model checked against the 15 LTL specifications in :mod:`logictune.specs`.

Two engines give the same verdicts:

* ``nuxmv``: writes an SMV model and runs the vendored nuXmv binary
  (``src/modelchecker/nuXmv``) with LTL model checking under the JUSTICE
  fairness constraint.
* ``explicit``: an exact explicit-state decision procedure. Because the
  controller is memoryless and the environment is unconstrained, every LTL
  formula in the suite reduces to a check over the set of single-step
  valuations (see ``_explicit_verdicts``). It is fast and needs no binary.

``engine="both"`` runs the two and raises if they disagree. The training
pipeline uses ``both`` so every score is backed by nuXmv.

The score of a response is the number of satisfied specifications (0..15).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .environment import ACTIONS, FAIRNESS_ASSIGNMENT, PROPOSITIONS, fairness_smv
from .glm2fsa import And, Const, Controller, Expr, Not, Or, Var, extract_rules
from .specs import NUM_SPECS, SPEC_IDS, SPECS

MAX_UNKNOWN_ATOMS_EXPLICIT = 10
_REPO_NUXMV = Path(__file__).resolve().parents[1] / "modelchecker" / "nuXmv"


def find_nuxmv() -> Optional[str]:
    """Locate nuXmv: $NUXMV, the vendored binary, or PATH."""
    for cand in (os.environ.get("NUXMV"), str(_REPO_NUXMV), shutil.which("nuXmv")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


@dataclass
class VerificationResult:
    score: int
    specs: Dict[str, bool]
    controller: Controller
    engine: str
    smv_hash: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.score / NUM_SPECS

    def to_json(self) -> dict:
        return {
            "score": self.score,
            "max_score": NUM_SPECS,
            "specs": dict(self.specs),
            "engine": self.engine,
            "smv_hash": self.smv_hash,
            "controller": self.controller.to_json(),
        }


# ---------------------------------------------------------------------------
# SMV generation
# ---------------------------------------------------------------------------


def controller_to_smv(ctrl: Controller) -> str:
    """Compose controller and environment into an SMV model with the 15 specs."""
    lines = ["MODULE main", "VAR"]
    for p in PROPOSITIONS:
        lines.append(f"  {p} : boolean;")
    for u in sorted(ctrl.unknown_atoms, key=lambda x: int(x[1:])):
        lines.append(f"  {u} : boolean; -- free atom: {ctrl.unknown_atoms[u][:60]}")
    lines.append(f"  nd_action : {{{', '.join(ACTIONS)}}};")
    lines.append("DEFINE")
    lines.append("  action :=")
    lines.append("    case")
    for r in ctrl.rules:
        lines.append(f"      {r.guard.to_smv()} : {r.action};")
    lines.append("      TRUE : nd_action;")
    lines.append("    esac;")
    defined = " | ".join(f"({r.guard.to_smv()})" for r in ctrl.rules) or "FALSE"
    lines.append(f"  action_defined := {defined};")
    lines.append(f"JUSTICE {fairness_smv()};")
    for s in SPECS:
        lines.append(f"LTLSPEC NAME {s.id} := {s.smv};")
    return "\n".join(lines) + "\n"


_PROP_LINE = re.compile(r"\[\s*LTL\s+(True|False)\s+\S+\s+(phi_\d+)\s*\]")


def run_nuxmv(smv: str, nuxmv: Optional[str] = None, timeout: int = 60) -> Dict[str, bool]:
    nuxmv = nuxmv or find_nuxmv()
    if nuxmv is None:
        raise FileNotFoundError("nuXmv binary not found (set $NUXMV)")
    with tempfile.TemporaryDirectory(prefix="logictune_") as td:
        model = Path(td) / "model.smv"
        cmds = Path(td) / "cmds.txt"
        model.write_text(smv)
        cmds.write_text("go\ncheck_ltlspec\nshow_property -l\nquit\n")
        out = subprocess.run(
            [nuxmv, "-dcx", "-source", str(cmds), str(model)],
            capture_output=True, text=True, timeout=timeout,
        )
    verdicts = {name: status == "True" for status, name in _PROP_LINE.findall(out.stdout)}
    if set(verdicts) != set(SPEC_IDS):
        raise RuntimeError(f"nuXmv returned {len(verdicts)} verdicts.\nstdout:\n{out.stdout[-2000:]}\nstderr:\n{out.stderr[-2000:]}")
    return verdicts


# ---------------------------------------------------------------------------
# Explicit-state checker
# ---------------------------------------------------------------------------


def _eval(e: Expr, env: Dict[str, np.ndarray], n: int) -> np.ndarray:
    if isinstance(e, Const):
        return np.full(n, e.value, dtype=bool)
    if isinstance(e, Var):
        return env[e.name]
    if isinstance(e, Not):
        return ~_eval(e.arg, env, n)
    if isinstance(e, And):
        out = np.ones(n, dtype=bool)
        for a in e.args:
            out &= _eval(a, env, n)
        return out
    if isinstance(e, Or):
        out = np.zeros(n, dtype=bool)
        for a in e.args:
            out |= _eval(a, env, n)
        return out
    raise TypeError(type(e))


def _explicit_verdicts(ctrl: Controller) -> Dict[str, bool]:
    """
    Exact LTL verdicts for a memoryless controller in the free environment.

    Let A(w) be the set of actions the controller can output in valuation w
    (one action if a rule fires, all four if none does). Because any
    valuation may follow any other, a path is an arbitrary sequence of
    valuations, and a path is fair iff it visits J (the JUSTICE set)
    infinitely often. Then:

    * G p (p over state and action) fails iff some w and a in A(w) violate p.
    * G(c -> F stop) fails iff some w |= c has a non-stop action and some
      j in J has a non-stop action (the suffix j^omega never stops).
    * G(c -> F !stop) fails iff some w |= c can stop and some j in J can stop.
    * F(green | green_left) -> F !stop fails iff some j in J can stop
      (every fair path visits green, so only an all-stop fair path fails).
    """
    atoms = list(PROPOSITIONS) + sorted(ctrl.unknown_atoms, key=lambda x: int(x[1:]))
    k = len(atoms)
    if k - len(PROPOSITIONS) > MAX_UNKNOWN_ATOMS_EXPLICIT:
        raise ValueError("too many free atoms for explicit checking")
    n = 1 << k
    idx = np.arange(n, dtype=np.int64)
    env = {name: ((idx >> i) & 1).astype(bool) for i, name in enumerate(atoms)}

    fired = np.zeros(n, dtype=bool)
    possible = {a: np.zeros(n, dtype=bool) for a in ACTIONS}
    for r in ctrl.rules:
        g = _eval(r.guard, env, n) & ~fired
        possible[r.action] |= g
        fired |= g
    undefined = ~fired
    for a in ACTIONS:
        possible[a] |= undefined

    E = env
    stop = possible["stop"]
    straight, left, right = possible["go_straight"], possible["turn_left"], possible["turn_right"]
    nonstop = straight | left | right
    J = np.ones(n, dtype=bool)
    for name, val in FAIRNESS_ASSIGNMENT.items():
        J &= E[name] if val else ~E[name]
    j_nonstop = bool(np.any(J & nonstop))
    j_stop = bool(np.any(J & stop))
    g, gl = E["green_traffic_light"], E["green_left_turn_light"]
    cl, cr, opp = E["car_from_left"], E["car_from_right"], E["opposite_car"]
    pf, pr, ss = E["pedestrian_in_front"], E["pedestrian_at_right"], E["stop_sign"]

    v = {
        "phi_1": not (np.any(pf & nonstop) and j_nonstop),
        "phi_2": not np.any(opp & ~gl & left),
        "phi_3": not np.any(~g & straight),
        "phi_4": not (np.any(ss & nonstop) and j_nonstop),
        "phi_5": not np.any((cl | pr) & right),
        "phi_6": not np.any(undefined),
        "phi_7": not j_stop,
        "phi_8": not (np.any(~g & nonstop) and j_nonstop),
        "phi_9": not np.any(cl & (left | right)),
        "phi_10": not (np.any(g & stop) and j_stop),
        "phi_11": not np.any(right & ~g & cl),
        "phi_12": not np.any(left & ~gl & (cr | cl | opp)),
        "phi_13": not (np.any(ss & ~cl & ~cr & stop) and j_stop),
        "phi_14": not np.any(straight & pf),
        "phi_15": not np.any(right & ss & cl),
    }
    return {k2: bool(val) for k2, val in v.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_cache: Dict[Tuple[str, str], Dict[str, bool]] = {}


def verify_controller(ctrl: Controller, engine: str = "explicit") -> VerificationResult:
    """Model check a parsed controller. ``engine``: explicit | nuxmv | both."""
    smv = controller_to_smv(ctrl)
    h = hashlib.sha1(smv.encode()).hexdigest()[:16]
    key = (h, engine)
    if key in _cache:
        verdicts = _cache[key]
    else:
        notes: List[str] = []
        if engine == "explicit":
            try:
                verdicts = _explicit_verdicts(ctrl)
            except ValueError:  # more than MAX_UNKNOWN_ATOMS_EXPLICIT free atoms
                verdicts = run_nuxmv(smv)
        elif engine == "nuxmv":
            verdicts = run_nuxmv(smv)
        elif engine == "both":
            nx_v = run_nuxmv(smv)
            try:
                ex_v = _explicit_verdicts(ctrl)
            except ValueError:
                ex_v = nx_v
            if ex_v != nx_v:
                diff = {k: (ex_v[k], nx_v[k]) for k in SPEC_IDS if ex_v[k] != nx_v[k]}
                raise AssertionError(f"explicit and nuXmv verdicts differ: {diff}\n{smv}")
            verdicts = nx_v
        else:
            raise ValueError(engine)
        _cache[key] = verdicts
    verdicts = {k: verdicts[k] for k in SPEC_IDS}
    return VerificationResult(
        score=sum(verdicts.values()), specs=verdicts, controller=ctrl, engine=engine, smv_hash=h
    )


def verify_response(response: str, engine: str = "explicit") -> VerificationResult:
    """Parse a model response and model check it."""
    ctrl = extract_rules(response)
    return verify_controller(ctrl, engine=engine)


def score_response(response: str, engine: str = "explicit", verbose: bool = False) -> Tuple[int, Dict[str, bool]]:
    """Return (number of satisfied specs, per-spec verdicts)."""
    res = verify_response(response, engine=engine)
    if verbose:
        print(format_result(res))
    return res.score, res.specs


def format_result(res: VerificationResult) -> str:
    out = ["Parsed controller:", res.controller.pretty(), "", f"Score: {res.score}/{NUM_SPECS} ({res.engine})"]
    for s in SPECS:
        mark = "PASS" if res.specs[s.id] else "FAIL"
        out.append(f"  [{mark}] {s.id:<6} {s.formula}")
    return "\n".join(out)
