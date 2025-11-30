"""
Controller Parser (GLM2FSA)

Parses natural language controller descriptions into executable FSAs.
Implements a simplified version of the GLM2FSA (Generative Language Model
to Finite State Automaton) logic.

Reference: "Fine-Tuning Language Models Using Formal Methods Feedback"
"""

import re
import networkx as nx
from typing import List, Tuple, Dict, Optional


class ControllerParser:
    """
    Parses natural language controller descriptions into FSA.
    
    Handles various input formats like numbered steps or structured descriptions.
    """
    
    def __init__(self):
        self.light_conditions = {
            'green': 'green_light',
            'yellow': 'yellow_light',
            'red': 'red_light'
        }
        
        self.action_keywords = {
            'go straight': 'go_straight',
            'straight': 'go_straight',
            'proceed': 'go_straight',
            'turn right': 'turn_right',
            'right': 'turn_right',
            'turn left': 'turn_left',
            'left': 'turn_left',
            'stop': 'stop',
            'wait': 'wait',
            'halt': 'stop'
        }
    
    def extract_steps(self, text: str) -> List[str]:
        """
        Extract numbered steps from the response.
        
        Args:
            text: LLM response text
            
        Returns:
            List of step strings
        """
        lines = text.strip().split('\n')
        
        steps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'^(?:\d+[\.\):]|\bstep\s+\d+[:\.]?)\s*(.+)$', 
                           line, re.IGNORECASE)
            if match:
                steps.append(match.group(1).strip())
            elif line and not steps:
                steps.append(line)
        
        return steps
    
    def parse_condition(self, text: str) -> Optional[str]:
        """
        Extract the condition (guard) from a step.
        
        Args:
            text: Step text
            
        Returns:
            Atomic proposition for the guard
        """
        text_lower = text.lower()
        
        for keyword, prop in self.light_conditions.items():
            patterns = [
                f'{keyword} light',
                f'light is {keyword}',
                f'light.*{keyword}',
                f'{keyword}'
            ]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return prop
        
        return 'any'
    
    def parse_action(self, text: str) -> Optional[str]:
        """
        Extract the action from a step.
        
        Args:
            text: Step text
            
        Returns:
            Action identifier
        """
        text_lower = text.lower()
        
        sorted_keywords = sorted(self.action_keywords.items(), 
                                key=lambda x: len(x[0]), 
                                reverse=True)
        
        for keyword, action in sorted_keywords:
            if keyword in text_lower:
                return action
        
        return 'wait'
    
    def parse_step(self, step_text: str, step_num: int) -> Tuple[str, str]:
        """
        Parse a single step into (guard, action) pair.
        
        Args:
            step_text: The text of the step
            step_num: Step number
            
        Returns:
            Tuple of (guard, action)
        """
        parts = re.split(r',|\bthen\b', step_text, maxsplit=1)
        
        if len(parts) == 2:
            condition_text, action_text = parts
            guard = self.parse_condition(condition_text)
            action = self.parse_action(action_text)
        else:
            guard = self.parse_condition(step_text)
            action = self.parse_action(step_text)
        
        return guard, action
    
    def build_fsa(self, steps: List[Tuple[str, str]]) -> nx.DiGraph:
        """
        Build FSA from parsed steps.
        
        Args:
            steps: List of (guard, action) tuples
            
        Returns:
            NetworkX DiGraph representing the controller FSA
        """
        fsa = nx.DiGraph()
        
        fsa.add_node(0, description="Initial - Check conditions")
        
        for i, (guard, action) in enumerate(steps):
            step_state = i + 1
            fsa.add_node(step_state, description=f"Step {i+1}: {action}")
            fsa.add_edge(0, step_state, guard=guard, action=action)
            fsa.add_edge(step_state, step_state, guard='any', action='wait')
        
        return fsa
    
    def simplify_fsa_for_verification(self, fsa: nx.DiGraph) -> nx.DiGraph:
        """
        Simplify FSA for verification by creating a reactive controller.
        
        Creates an FSA that can execute any of the actions mentioned in the
        controller description for verification purposes.
        
        Args:
            fsa: Original FSA
            
        Returns:
            Simplified FSA suitable for product automaton construction
        """
        simplified = nx.DiGraph()
        simplified.add_node(0, description="Initial")
        
        actions_seen = set()
        
        for u, v, data in fsa.edges(data=True):
            action = data.get('action')
            if action and action not in ['wait', 'any', None]:
                actions_seen.add(action)
        
        if not actions_seen:
            actions_seen = {'wait'}
        
        state_id = 1
        for action in sorted(actions_seen):
            action_state = state_id
            simplified.add_node(action_state, description=f"Action: {action}")
            simplified.add_edge(0, action_state, action=action)
            simplified.add_edge(action_state, action_state, action=action)
            simplified.add_edge(action_state, action_state, action='wait')
            state_id += 1
        
        simplified.add_edge(0, 0, action='wait')
        
        return simplified


def parse_response_to_fsa(llm_text_response: str, verbose: bool = False) -> nx.DiGraph:
    """
    Convert LLM text response to FSA.
    
    Implements the GLM2FSA pipeline:
    1. Extract structured steps from text
    2. Parse each step into (guard, action) pairs
    3. Construct FSA from parsed steps
    
    Args:
        llm_text_response: Text output from the language model
        verbose: Print parsing details
        
    Returns:
        NetworkX DiGraph representing the controller FSA
    """
    parser = ControllerParser()
    
    steps_text = parser.extract_steps(llm_text_response)
    
    if verbose:
        print(f"Extracted {len(steps_text)} steps:")
        for i, step in enumerate(steps_text, 1):
            print(f"  {i}. {step}")
    
    parsed_steps = []
    for i, step_text in enumerate(steps_text):
        guard, action = parser.parse_step(step_text, i)
        parsed_steps.append((guard, action))
        
        if verbose:
            print(f"  Step {i}: Guard={guard}, Action={action}")
    
    fsa = parser.build_fsa(parsed_steps)
    simplified_fsa = parser.simplify_fsa_for_verification(fsa)
    
    if verbose:
        print(f"\nFSA built with {len(simplified_fsa.nodes())} states")
        print(f"Transitions:")
        for u, v, data in simplified_fsa.edges(data=True):
            print(f"  State {u} --[{data.get('action')}]--> State {v}")
    
    return simplified_fsa

