"""
Formal Verification and Scoring

Implements automated feedback mechanism via formal verification.
Constructs product automaton M ⊗ C and checks safety specifications.

Reference: "Fine-Tuning Language Models Using Formal Methods Feedback"
"""

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
    
    Returns:
        List of SafetySpec objects
    """
    specs = [
        SafetySpec(
            spec_type='never_both',
            props=['red_light', 'go_straight'],
            description="Never go straight on red light"
        ),
        SafetySpec(
            spec_type='never_both',
            props=['red_light', 'turn_left'],
            description="Never turn left on red light"
        ),
        SafetySpec(
            spec_type='never_both',
            props=['yellow_light', 'go_straight'],
            description="Should not speed through yellow"
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
    Score a controller response using formal verification.
    
    Implements the automated feedback mechanism:
    1. Build product automaton M ⊗ C
    2. Check safety specifications
    3. Return score based on satisfied specs
    
    Args:
        system: The environment transition system
        controller_fsa: The controller FSA parsed from LLM response
        controller_initial: Initial state of controller
        verbose: Print detailed results
        
    Returns:
        Tuple of (score, detailed_results) where score is 0-3
    """
    product = build_product_automaton(system, controller_fsa, controller_initial)
    specs = get_traffic_safety_specs()
    score, results = verify_safety_specs(product, specs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VERIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Score: {score}/{len(specs)} specifications satisfied\n")
        for spec_desc, satisfied in results.items():
            status = "✓ SATISFIED" if satisfied else "✗ VIOLATED"
            print(f"{status}: {spec_desc}")
        print(f"{'='*60}\n")
    
    return score, results

