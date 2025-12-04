"""
Formal Verification and Scoring

Implements automated feedback mechanism via formal verification.
Constructs product automaton M ⊗ C and checks safety specifications.

Reference: "Fine-Tuning Language Models Using Formal Methods Feedback"
"""

import os
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
from typing import Set, Dict, List, Tuple, Optional
from collections import deque
from .environment import TransitionSystem


class SafetySpec:
    """
    Represents a safety specification in LTL-style logic.
    """
    
    def __init__(self, spec_type: str, props: List[str], description: str):
        """
        Args:
            spec_type: Type of specification ('never_both', 'always_implies', 'eventually')
            props: List of propositions involved
            description: Human-readable description
        """
        self.spec_type = spec_type
        self.props = props
        self.description = description
    
    def check_state_violation(self, state_props: Set[str]) -> bool:
        """
        Check if a single state violates this safety spec.
        
        Returns:
            True if state violates the spec, False otherwise
        """
        if self.spec_type == 'never_both':
            return all(prop in state_props for prop in self.props)
        
        elif self.spec_type == 'always_implies':
            if self.props[0] in state_props:
                return self.props[1] not in state_props
            return False
        
        return False
    
    def __str__(self):
        return f"SafetySpec({self.spec_type}): {self.description}"


def get_traffic_safety_specs() -> List[SafetySpec]:
    """
    Define safety specifications for the traffic intersection scenario.
    Implements all 15 LTL specifications for comprehensive verification.
    
    Returns:
        List of SafetySpec objects covering all 15 LTL specifications
    """
    specs = [
        # Φ1 = □(pedestrian → (♢ stop))
        # Always: if pedestrian present, then eventually stop
        SafetySpec(
            spec_type='always_implies',
            props=['pedestrian', 'stop'],
            description="Φ1: Always if pedestrian present, then eventually stop"
        ),
        
        # Φ2 = □(opposite car ∧ ¬green left-turn light → ¬turn left)
        # Always: if opposite car and not green left-turn light, then don't turn left
        SafetySpec(
            spec_type='never_both',
            props=['opposite_car', 'turn_left'],
            description="Φ2: Never turn left when opposite car present and no green left-turn light"
        ),
        
        # Φ3 = □(¬green traffic light → ¬go straight)
        # Always: if not green traffic light, then don't go straight
        SafetySpec(
            spec_type='never_both',
            props=['red_light', 'go_straight'],
            description="Φ3: Never go straight on red light"
        ),
        SafetySpec(
            spec_type='never_both',
            props=['yellow_light', 'go_straight'],
            description="Φ3: Never go straight on yellow light (strict interpretation)"
        ),
        
        # Φ4 = □(stop sign → ♢ stop)
        # Always: if stop sign present, then eventually stop
        SafetySpec(
            spec_type='always_implies',
            props=['stop_sign', 'stop'],
            description="Φ4: Always if stop sign present, then eventually stop"
        ),
        
        # Φ5 = □(car from left ∨ pedestrian at right → ¬turn right)
        # Always: if car from left or pedestrian at right, then don't turn right
        SafetySpec(
            spec_type='never_both',
            props=['car_from_left', 'turn_right'],
            description="Φ5: Never turn right when car from left"
        ),
        SafetySpec(
            spec_type='never_both',
            props=['pedestrian_at_right', 'turn_right'],
            description="Φ5: Never turn right when pedestrian at right"
        ),
        
        # Φ6 = □(stop ∨ go straight ∨ turn left ∨ turn right)
        # Always: one of these actions must be available (covered by action space)
        # Note: This is a liveness property ensuring action availability, 
        # typically verified at the system level rather than as a safety violation
        
        # Φ7 = ♢(green traffic light ∨ green left-turn light) → ♢ ¬stop)
        # Eventually: if green light, then eventually not stop
        SafetySpec(
            spec_type='always_implies',
            props=['green_light', 'go_straight'],
            description="Φ7: Always if green traffic light, then eventually proceed (not stop)"
        ),
        SafetySpec(
            spec_type='always_implies',
            props=['green_left_turn_light', 'turn_left'],
            description="Φ7: Always if green left-turn light, then eventually proceed"
        ),
        
        # Φ8 = □(¬green traffic light → ♢ stop)
        # Always: if not green traffic light, then eventually stop
        SafetySpec(
            spec_type='always_implies',
            props=['red_light', 'stop'],
            description="Φ8: Always if red light, then eventually stop"
        ),
        SafetySpec(
            spec_type='always_implies',
            props=['yellow_light', 'stop'],
            description="Φ8: Always if yellow light, then eventually stop"
        ),
        
        # Φ9 = □(car from left → ¬(turn left ∨ turn right))
        # Always: if car from left, then don't turn left or right
        SafetySpec(
            spec_type='never_both',
            props=['car_from_left', 'turn_left'],
            description="Φ9: Never turn left when car from left"
        ),
        SafetySpec(
            spec_type='never_both',
            props=['car_from_left', 'turn_right'],
            description="Φ9: Never turn right when car from left"
        ),
        
        # Φ10 = □(green traffic light → ♢ ¬stop)
        # Always: if green traffic light, then eventually not stop
        SafetySpec(
            spec_type='always_implies',
            props=['green_light', 'go_straight'],
            description="Φ10: Always if green traffic light, then eventually proceed"
        ),
        
        # Φ11 = □((turn right ∧ ¬green traffic light) → ¬car from left)
        # Always: if turn right and not green traffic light, then no car from left
        # This is a precondition check - if turning right on non-green, ensure no car from left
        SafetySpec(
            spec_type='never_both',
            props=['turn_right', 'car_from_left'],
            description="Φ11: Never turn right on non-green light when car from left"
        ),
        
        # Φ12 = □((turn left ∧ ¬green left-turn light) → (¬car from right ∧ ¬car from left ∧ ¬opposite car))
        # Always: if turn left and not green left-turn light, then no cars from any direction
        SafetySpec(
            spec_type='never_both',
            props=['turn_left', 'car_from_right'],
            description="Φ12: Never turn left without green left-turn light when car from right"
        ),
        SafetySpec(
            spec_type='never_both',
            props=['turn_left', 'car_from_left'],
            description="Φ12: Never turn left without green left-turn light when car from left"
        ),
        SafetySpec(
            spec_type='never_both',
            props=['turn_left', 'opposite_car'],
            description="Φ12: Never turn left without green left-turn light when opposite car present"
        ),
        
        # Φ13 = □((stop sign ∧ ¬car from left ∧ ¬car from right) → (♢ ¬stop))
        # Always: if stop sign and no cars, then eventually not stop
        SafetySpec(
            spec_type='always_implies',
            props=['stop_sign', 'go_straight'],
            description="Φ13: Always if stop sign and no cars, then eventually proceed"
        ),
        
        # Φ14 = □((go straight → ¬pedestrian in front)
        # Always: if go straight, then no pedestrian in front
        SafetySpec(
            spec_type='never_both',
            props=['go_straight', 'pedestrian_in_front'],
            description="Φ14: Never go straight when pedestrian in front"
        ),
        
        # Φ15 = □((turn right ∧ stop sign) → ¬car from left)
        # Always: if turn right and stop sign, then no car from left
        SafetySpec(
            spec_type='never_both',
            props=['turn_right', 'car_from_left'],
            description="Φ15: Never turn right at stop sign when car from left"
        ),
    ]
    return specs


class ProductAutomaton:
    """
    Product Automaton M ⊗ C representing the composition of:
    - M: System transition model
    - C: Controller FSA
    
    States are pairs (s_m, s_c) where s_m ∈ M and s_c ∈ C
    """
    
    def __init__(self, system: TransitionSystem, controller: nx.DiGraph):
        """
        Args:
            system: The environment transition system (M)
            controller: The controller FSA (C)
        """
        self.system = system
        self.controller = controller
        self.graph = nx.DiGraph()
        self.state_labels = {}
        self.initial_state = None
        
    def build(self, controller_initial: int = 0) -> None:
        """
        Construct the product automaton M ⊗ C.
        
        Args:
            controller_initial: Initial state of the controller FSA
        """
        self.initial_state = (self.system.initial_state, controller_initial)
        
        queue = deque([self.initial_state])
        visited = {self.initial_state}
        
        while queue:
            s_m, s_c = queue.popleft()
            
            system_props = self.system.get_state_props(s_m)
            self.state_labels[(s_m, s_c)] = system_props
            self.graph.add_node((s_m, s_c))
            
            system_transitions = []
            for successor in self.system.graph.successors(s_m):
                edge_data = self.system.graph.get_edge_data(s_m, successor)
                action = edge_data.get('action', 'unknown')
                system_transitions.append((successor, action))
            
            controller_transitions = {}
            if s_c in self.controller.nodes():
                for successor in self.controller.successors(s_c):
                    edge_data = self.controller.get_edge_data(s_c, successor)
                    action = edge_data.get('action', 'unknown')
                    if action not in controller_transitions:
                        controller_transitions[action] = []
                    controller_transitions[action].append(successor)
            
            for s_m_next, action in system_transitions:
                if action in controller_transitions:
                    for s_c_next in controller_transitions[action]:
                        product_next = (s_m_next, s_c_next)
                        self.graph.add_edge((s_m, s_c), product_next, action=action)
                        
                        if product_next not in visited:
                            visited.add(product_next)
                            queue.append(product_next)
    
    def get_reachable_states(self) -> List[Tuple[int, int]]:
        """Get all reachable states in the product automaton from initial state."""
        if self.initial_state is None:
            return []
        
        reachable = set()
        queue = deque([self.initial_state])
        reachable.add(self.initial_state)
        
        while queue:
            state = queue.popleft()
            for successor in self.graph.successors(state):
                if successor not in reachable:
                    reachable.add(successor)
                    queue.append(successor)
        
        return list(reachable)
    
    def get_state_props(self, state: Tuple[int, int]) -> Set[str]:
        """Get atomic propositions for a product state."""
        return self.state_labels.get(state, set())


def build_product_automaton(system: TransitionSystem, 
                            controller: nx.DiGraph,
                            controller_initial: int = 0) -> ProductAutomaton:
    """
    Build the product automaton M ⊗ C.
    
    Args:
        system: The environment transition system
        controller: The controller FSA
        controller_initial: Initial state of controller
        
    Returns:
        ProductAutomaton object
    """
    product = ProductAutomaton(system, controller)
    product.build(controller_initial)
    return product


def verify_safety_specs(product: ProductAutomaton, 
                        specs: List[SafetySpec]) -> Tuple[int, Dict[str, bool]]:
    """
    Verify safety specifications on the product automaton.
    
    Args:
        product: The product automaton M ⊗ C
        specs: List of safety specifications to check
        
    Returns:
        Tuple of (score, detailed_results)
    """
    reachable_states = product.get_reachable_states()
    
    results = {}
    satisfied_count = 0
    
    for spec in specs:
        violated = False
        
        for state in reachable_states:
            state_props = product.get_state_props(state)
            
            if spec.spec_type == 'never_both':
                for successor in product.graph.successors(state):
                    edge_data = product.graph.get_edge_data(state, successor)
                    if edge_data:
                        action = edge_data.get('action')
                        if action:
                            state_props_with_action = state_props | {action}
                            if spec.check_state_violation(state_props_with_action):
                                violated = True
                                break
                                
            elif spec.spec_type == 'always_implies':
                if spec.props[0] in state_props:
                    has_required_action = False
                    for successor in product.graph.successors(state):
                        edge_data = product.graph.get_edge_data(state, successor)
                        if edge_data:
                            action = edge_data.get('action')
                            if action in [spec.props[1], 'wait']:
                                has_required_action = True
                                break
                    
                    if not has_required_action:
                        violated = True
                        break
            
            if violated:
                break
        
        satisfied = not violated
        results[spec.description] = satisfied
        if satisfied:
            satisfied_count += 1
    
    return satisfied_count, results


def score_response(system: TransitionSystem, 
                   controller_fsa: nx.DiGraph,
                   controller_initial: int = 0,
                   verbose: bool = False) -> Tuple[int, Dict[str, bool]]:
    """
    Score a controller response using formal verification on the
    composed product automaton M ⊗ C.

    Implements the automated feedback mechanism:
        1. Build product automaton M ⊗ C from the environment transition
           system (M) and the controller FSA (C) parsed from the LLM.
        2. Check all 15 safety specifications (Φ1–Φ15) over the product.
        3. Return a score equal to the number of satisfied specifications.
    
    Args:
        system: The environment transition system (M).
        controller_fsa: The controller FSA parsed from the LLM response.
        controller_initial: Initial state of the controller FSA.
        verbose: If True, print detailed results.
        
    Returns:
        Tuple of (score, detailed_results) where:
            - score is the number of satisfied specifications
            - detailed_results maps spec descriptions to booleans.
    """
    product = build_product_automaton(system, controller_fsa, controller_initial)
    specs = get_traffic_safety_specs()
    score, results = verify_safety_specs(product, specs)

    if verbose:
        print(f"\n{'='*60}")
        print("VERIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Score: {score}/{len(specs)} specifications satisfied\n")
        for spec_desc, satisfied in results.items():
            status = "✓ SATISFIED" if satisfied else "✗ VIOLATED"
            print(f"{status}: {spec_desc}")
        print(f"{'='*60}\n")

    return score, results


# ---------------------------------------------------------------------------
# nuXmv integration (sole verification backend)
# ---------------------------------------------------------------------------

NUXMV_PATH = Path(__file__).resolve().parents[1] / "modelchecker" / "nuXmv"
DEFAULT_SMV_MODEL = Path(__file__).resolve().parents[1] / "modelchecker" / "traffic_specs.smv"


def run_nuxmv_model(
    model_file: str,
    ltl_names: Optional[List[str]] = None,
    nuxmv_path: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, bool]:
    """
    Run nuXmv on a given SMV model file and check the requested LTL specs.

    This function assumes that the SMV model already contains the corresponding
    LTLSPEC declarations (e.g., the 15 Φ-specifications) with names matching
    the entries in ``ltl_names``.

    Args:
        model_file: Path to the .smv model file.
        ltl_names: Optional list of LTLSPEC names to check. If None, all
            LTLSPECs in the model are checked.
        nuxmv_path: Optional explicit path to the nuXmv binary. If None,
            uses the bundled binary at ``src/modelchecker/nuXmv``.
        timeout: Maximum time in seconds to allow nuXmv to run.

    Returns:
        Dict mapping each checked LTL spec name to True (satisfied) or
        False (violated).
    """
    model_file_abs = os.path.abspath(model_file)

    if nuxmv_path is None:
        nuxmv_exec = str(NUXMV_PATH)
    else:
        nuxmv_exec = nuxmv_path

    if not os.path.isfile(model_file_abs):
        raise FileNotFoundError(f"nuXmv model file not found: {model_file_abs}")

    if not os.path.isfile(nuxmv_exec):
        raise FileNotFoundError(f"nuXmv binary not found at: {nuxmv_exec}")

    # Build a nuXmv command script on the fly.
    commands: List[str] = [
        f"read_model -i {model_file_abs}",
        "go",
    ]

    if ltl_names:
        for name in ltl_names:
            # Use -n NAME to refer to the LTLSPEC NAME sample_phi_1 := ...
            commands.append(f"check_ltlspec -n {name}")
    else:
        # Fallback: check all LTL specifications defined in the model.
        commands.append("check_ltlspec")

    commands.append("quit")
    script_content = "\n".join(commands) + "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cmd", delete=False) as cmd_file:
        cmd_file.write(script_content)
        cmd_path = cmd_file.name

    try:
        completed = subprocess.run(
            [nuxmv_exec, "-source", cmd_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    finally:
        try:
            os.remove(cmd_path)
        except OSError:
            pass

    if completed.returncode != 0:
        raise RuntimeError(
            f"nuXmv exited with code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    # Parse nuXmv output lines like:
    #   -- specification sample_phi_1  is true
    #   -- specification sample_phi_2  is false
    results: Dict[str, bool] = {}
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-- specification"):
            continue

        parts = stripped.split()
        # Expected minimal structure:
        # ['--', 'specification', '<name>', 'is', 'true/false', ...]
        if len(parts) < 5:
            continue

        name = parts[2]
        is_true = "is true" in stripped
        is_false = "is false" in stripped

        if is_true:
            results[name] = True
        elif is_false:
            results[name] = False

    return results


def verify_with_nuxmv(
    model_file: str,
    ltl_names: Optional[List[str]] = None,
    nuxmv_path: Optional[str] = None,
    timeout: int = 60,
    verbose: bool = False,
) -> Dict[str, bool]:
    """
    Convenience wrapper to run nuXmv and (optionally) pretty-print results.

    Typical usage is to point ``model_file`` to an SMV model encoding the
    traffic intersection and controller (e.g., modules like
    ``turn_left_before_finetune``, ``turn_left_after_finetune``,
    ``turn_right_before_finetune``, ``turn_right_after_finetune``) together
    with the 15 LTL specifications (Φ1–Φ15) as named LTLSPECs.

    Example equivalent to the sample nuXmv script:
        read_model -i right_turn.smv
        go
        check_ltlspec -n phi_1
        check_ltlspec -n phi_2
        quit

    Args:
        model_file: Path to the .smv model file.
        ltl_names: Optional list of LTLSPEC names (e.g., ['phi_1', 'phi_2']).
        nuxmv_path: Optional path to nuXmv binary if not using the bundled one.
        timeout: Maximum time in seconds for nuXmv to run.
        verbose: If True, prints a short human-readable summary.

    Returns:
        Dict mapping LTL spec names to booleans (True = satisfied).
    """
    results = run_nuxmv_model(
        model_file=model_file,
        ltl_names=ltl_names,
        nuxmv_path=nuxmv_path,
        timeout=timeout,
    )

    if verbose:
        print(f"\n{'='*60}")
        print("nuXmv LTL VERIFICATION RESULTS")
        print(f"Model: {os.path.abspath(model_file)}")
        print(f"{'='*60}")
        for name, ok in sorted(results.items()):
            status = "✓ SATISFIED" if ok else "✗ VIOLATED"
            print(f"{status}: {name}")
        print(f"{'='*60}\n")

    return results

