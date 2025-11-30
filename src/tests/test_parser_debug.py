#!/usr/bin/env python3
"""Debug script to understand parser behavior."""

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
print("Extracted steps:")
for i, step in enumerate(steps_text, 1):
    print(f"  {i}. {step}")

parsed_steps = []
for i, step_text in enumerate(steps_text):
    guard, action = parser.parse_step(step_text, i)
    parsed_steps.append((guard, action))
    print(f"\nStep {i}: '{step_text}'")
    print(f"  → Guard: {guard}")
    print(f"  → Action: {action}")

print("\n" + "="*60)
print("Building FSA...")
fsa = parser.build_fsa(parsed_steps)

print(f"FSA has {len(fsa.nodes())} nodes and {len(fsa.edges())} edges")
print("\nEdges in FSA:")
for u, v, data in fsa.edges(data=True):
    print(f"  {u} → {v}: {data}")

print("\n" + "="*60)
print("Simplifying FSA...")
simplified = parser.simplify_fsa_for_verification(fsa)

print(f"Simplified FSA has {len(simplified.nodes())} nodes and {len(simplified.edges())} edges")
print("\nEdges in simplified FSA:")
for u, v, data in simplified.edges(data=True):
    print(f"  {u} → {v}: action={data.get('action')}")

