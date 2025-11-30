#!/usr/bin/env python3
"""Debug script to understand simplification."""

import sys
sys.path.insert(0, '/home/alira/projects/Working/LogicTune/src')

from logictune.parser import ControllerParser

response = """
1. If the light is green, go straight.
2. If the light is yellow, stop.
3. If the light is red, stop.
"""

parser = ControllerParser()
steps_text = parser.extract_steps(response)
parsed_steps = [(parser.parse_step(st, i)) for i, st in enumerate(steps_text)]

print("Parsed steps:")
for guard, action in parsed_steps:
    print(f"  Guard={guard}, Action={action}")

fsa = parser.build_fsa(parsed_steps)

print("\nOriginal FSA edges:")
for u, v, data in fsa.edges(data=True):
    print(f"  {u} → {v}: {data}")

print("\n" + "="*60)
print("Manual simplification:")
actions_seen = set()

for u, v, data in fsa.edges(data=True):
    action = data.get('action')
    print(f"  Edge {u}→{v}: action='{action}'")
    if action and action not in ['wait', 'any', None]:
        print(f"    → Adding '{action}' to actions_seen")
        actions_seen.add(action)
    else:
        print(f"    → Skipping '{action}'")

print(f"\nFinal actions_seen: {actions_seen}")

