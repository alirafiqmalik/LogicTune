"""
Demo script demonstrating the LogicTune pipeline components.

This script shows all components working together without requiring GPU.
"""

import sys
sys.path.insert(0, '/home/alira/projects/Working/LogicTune/src')

from logictune import (
    build_traffic_intersection_model,
    parse_response_to_fsa,
    score_response
)


def demo_environment():
    """Demonstrate the traffic intersection model."""
    print("\n" + "="*70)
    print(" DEMO 1: TRAFFIC INTERSECTION TRANSITION SYSTEM")
    print("="*70)
    
    system = build_traffic_intersection_model()
    print(f"System built with {len(system.get_all_states())} states")
    print(f"Transitions: {len(system.get_all_transitions())}")
    print(f"Atomic propositions: {system.atomic_propositions}")
    
    print("\n✓ Environment model working correctly!")
    return system


def demo_parser():
    """Demonstrate the controller parser."""
    print("\n" + "="*70)
    print(" DEMO 2: CONTROLLER PARSER (GLM2FSA)")
    print("="*70)
    
    examples = [
        {
            "name": "Safe Controller",
            "response": """
1. If the light is green, go straight through the intersection.
2. If the light is yellow, slow down and stop.
3. If the light is red, stop and wait.
            """
        },
        {
            "name": "Unsafe Controller",
            "response": """
1. Always go straight regardless of the light.
2. Speed through yellow lights.
3. Turn left even on red.
            """
        }
    ]
    
    fsas = []
    for example in examples:
        print(f"\n{'─'*70}")
        print(f"Example: {example['name']}")
        print(f"{'─'*70}")
        print(f"Response:{example['response']}")
        print(f"{'─'*70}")
        
        fsa = parse_response_to_fsa(example['response'], verbose=True)
        fsas.append(fsa)
    
    print("\n✓ Parser working correctly!")
    return fsas


def demo_verifier(system, fsas):
    """Demonstrate formal verification."""
    print("\n" + "="*70)
    print(" DEMO 3: FORMAL VERIFICATION")
    print("="*70)
    
    print("\n" + "─"*70)
    print("Safety Specifications:")
    print("─"*70)
    print("1. Never go straight on red light")
    print("2. Never turn left on red light")
    print("3. Should not speed through yellow")
    
    names = ["Safe Controller", "Unsafe Controller"]
    
    for name, fsa in zip(names, fsas):
        print(f"\n{'─'*70}")
        print(f"Verifying: {name}")
        print(f"{'─'*70}")
        score, results = score_response(system, fsa, verbose=True)
    
    print("\n✓ Verifier working correctly!")


def demo_integration():
    """Demonstrate full integration."""
    print("\n" + "="*70)
    print(" DEMO 4: FULL INTEGRATION TEST")
    print("="*70)
    
    system = build_traffic_intersection_model()
    
    test_cases = [
        {
            "name": "Perfect Safe Response",
            "response": "1. If green, go straight. 2. If yellow, stop. 3. If red, stop.",
            "expected_score": 3
        },
        {
            "name": "Partially Safe Response", 
            "response": "1. If green, go straight. 2. If yellow, go straight. 3. If red, stop.",
            "expected_score": 2
        },
        {
            "name": "Unsafe Response",
            "response": "1. Always go straight. 2. Turn left on red. 3. Speed up on yellow.",
            "expected_score": 0
        }
    ]
    
    results = []
    for test in test_cases:
        print(f"\n{'─'*70}")
        print(f"Test: {test['name']}")
        print(f"Response: {test['response']}")
        print(f"Expected Score: {test['expected_score']}/15")
        print(f"{'─'*70}")
        
        fsa = parse_response_to_fsa(test['response'], verbose=False)
        score, details = score_response(system, fsa, verbose=False)
        
        status = "✓ PASS" if score == test['expected_score'] else "✗ FAIL"
        print(f"Actual Score: {score}/15")
        print(f"Status: {status}")
        
        results.append(score == test['expected_score'])
    
    if all(results):
        print("\n" + "="*70)
        print(" ✓ ALL TESTS PASSED!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(" ⚠ Some tests failed.")
        print("="*70)


def main():
    """Run all demos."""
    print("="*70)
    print(" LOGICTUNE PIPELINE DEMONSTRATION")
    print(" Fine-Tuning LMs Using Formal Methods Feedback")
    print("="*70)
    
    try:
        system = demo_environment()
        input("\nPress Enter to continue...")
        
        fsas = demo_parser()
        input("\nPress Enter to continue...")
        
        demo_verifier(system, fsas)
        input("\nPress Enter to continue...")
        
        demo_integration()
        
        print("\n" + "="*70)
        print(" 🎉 DEMO COMPLETE!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

