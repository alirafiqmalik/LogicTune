"""
Environment Transition System

Implements the transition system (M) for modeling the environment state space.
This module provides formal models for verification of control policies.

Reference: "Fine-Tuning Language Models Using Formal Methods Feedback"
"""

import networkx as nx
from typing import Set, Dict, List, Tuple, Optional


class TransitionSystem:
    """
    A Transition System M = (S, s0, AP, L, →) where:
    - S: Set of states
    - s0: Initial state
    - AP: Set of atomic propositions
    - L: Labeling function (S → 2^AP)
    - →: Transition relation
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.atomic_propositions = set()
        self.initial_state = None
        self.state_labels = {}
        self.state_counter = 0
        
    def add_state(self, propositions: Set[str], is_initial: bool = False) -> int:
        """
        Add a state with its atomic propositions.
        
        Args:
            propositions: Set of atomic propositions that are true in this state
            is_initial: Whether this is the initial state
            
        Returns:
            state_id: Integer identifier for the state
        """
        state_id = self.state_counter
        self.state_counter += 1
        
        self.graph.add_node(state_id)
        self.state_labels[state_id] = propositions
        self.atomic_propositions.update(propositions)
        
        if is_initial:
            self.initial_state = state_id
            
        return state_id
    
    def add_transition(self, from_state: int, to_state: int, action: str):
        """
        Add a transition from one state to another labeled with an action.
        
        Args:
            from_state: Source state ID
            to_state: Target state ID
            action: Action label
        """
        self.graph.add_edge(from_state, to_state, action=action)
    
    def get_state_props(self, state_id: int) -> Set[str]:
        """Get the atomic propositions true in a given state."""
        return self.state_labels.get(state_id, set())
    
    def get_valid_next_states(self, current_state: int, action: str) -> List[int]:
        """
        Get all valid next states from current state given an action.
        
        Args:
            current_state: Current state ID
            action: Action to take
            
        Returns:
            List of valid next state IDs
        """
        next_states = []
        for successor in self.graph.successors(current_state):
            edge_data = self.graph.get_edge_data(current_state, successor)
            if edge_data and edge_data.get('action') == action:
                next_states.append(successor)
        return next_states
    
    def get_all_states(self) -> List[int]:
        """Get all state IDs in the system."""
        return list(self.graph.nodes())
    
    def get_all_transitions(self) -> List[Tuple[int, int, str]]:
        """Get all transitions as (from_state, to_state, action) tuples."""
        transitions = []
        for u, v, data in self.graph.edges(data=True):
            transitions.append((u, v, data.get('action', 'unknown')))
        return transitions


def build_traffic_intersection_model() -> TransitionSystem:
    """
    Build a traffic intersection transition system model.
    
    Models a vehicle at an intersection with traffic lights capturing:
    - Light states: green, yellow, red
    - Car position: car_left (approaching), no_car_left (crossed/stopped)
    - Possible actions: go_straight, turn_right, turn_left, stop, wait
    
    Safety specifications to satisfy:
    1. Never go straight on red light
    2. Never turn left on red light
    3. Should not speed through yellow light
    
    Returns:
        TransitionSystem: The constructed traffic intersection model
    """
    ts = TransitionSystem()
    
    light_states = ['green_light', 'yellow_light', 'red_light']
    car_positions = ['car_left', 'no_car_left']
    
    state_map = {}
    
    for light in light_states:
        for car_pos in car_positions:
            props = {light, car_pos}
            is_init = (light == 'green_light' and car_pos == 'car_left')
            state_id = ts.add_state(props, is_initial=is_init)
            state_map[(light, car_pos)] = state_id
    
    green_car_left = state_map[('green_light', 'car_left')]
    green_no_car = state_map[('green_light', 'no_car_left')]
    yellow_car_left = state_map[('yellow_light', 'car_left')]
    yellow_no_car = state_map[('yellow_light', 'no_car_left')]
    red_car_left = state_map[('red_light', 'car_left')]
    red_no_car = state_map[('red_light', 'no_car_left')]
    
    # Green light transitions
    ts.add_transition(green_car_left, green_no_car, 'go_straight')
    ts.add_transition(green_car_left, green_no_car, 'turn_right')
    ts.add_transition(green_car_left, green_no_car, 'turn_left')
    ts.add_transition(green_car_left, green_car_left, 'stop')
    
    # Yellow light transitions
    ts.add_transition(yellow_car_left, yellow_no_car, 'go_straight')
    ts.add_transition(yellow_car_left, yellow_no_car, 'turn_right')
    ts.add_transition(yellow_car_left, yellow_no_car, 'turn_left')
    ts.add_transition(yellow_car_left, yellow_car_left, 'stop')
    
    # Red light transitions (including unsafe but physically possible actions)
    ts.add_transition(red_car_left, red_car_left, 'stop')
    ts.add_transition(red_car_left, red_no_car, 'turn_right')
    ts.add_transition(red_car_left, red_no_car, 'go_straight')
    ts.add_transition(red_car_left, red_no_car, 'turn_left')
    
    # Light changes
    ts.add_transition(green_car_left, yellow_car_left, 'wait')
    ts.add_transition(green_no_car, yellow_no_car, 'wait')
    ts.add_transition(yellow_car_left, red_car_left, 'wait')
    ts.add_transition(yellow_no_car, red_no_car, 'wait')
    ts.add_transition(red_car_left, green_car_left, 'wait')
    ts.add_transition(red_no_car, green_no_car, 'wait')
    
    # Self-loops
    ts.add_transition(green_no_car, green_no_car, 'wait')
    ts.add_transition(yellow_no_car, yellow_no_car, 'wait')
    ts.add_transition(red_no_car, red_no_car, 'wait')
    
    return ts

