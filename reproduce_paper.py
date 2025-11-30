"""
End-to-End Reproduction Script

This script reproduces the key results from the paper:
"Fine-Tuning Language Models Using Formal Methods Feedback"

Expected outcome: Improvement in specification satisfaction rate from ~60% to ~90%
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from logictune import (
    iterative_training,
    compare_models,
    evaluate_model_simple
)


def reproduce_paper_results(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quick_mode: bool = False
):
    """
    Reproduce the paper's main result: improvement in specification satisfaction.
    
    Args:
        base_model: Base model to use
        quick_mode: Use fewer samples for faster execution
    """
    print("="*70)
    print(" REPRODUCING PAPER RESULTS")
    print(" 'Fine-Tuning Language Models Using Formal Methods Feedback'")
    print("="*70)
    print("\nExpected outcome: Improvement from ~60% to ~90% satisfaction rate")
    print("\nThis will:")
    print("1. Evaluate base model performance")
    print("2. Generate training data using formal verification")
    print("3. Fine-tune model with DPO")
    print("4. Evaluate fine-tuned model")
    print("5. Report improvement")
    print("="*70)
    
    if quick_mode:
        print("\n⚠ QUICK MODE: Using reduced parameters for faster execution")
        print("   For full results, run without --quick flag")
        num_iterations = 1
        epochs_per_iter = 1
        n_responses = 2
        n_eval_samples = 10
        batch_size = 1
    else:
        num_iterations = 2
        epochs_per_iter = 3
        n_responses = 4
        n_eval_samples = 20
        batch_size = 2
    
    output_dir = "paper_reproduction"
    
    try:
        print("\n" + "="*70)
        print(" STARTING REPRODUCTION")
        print("="*70)
        
        results = iterative_training(
            base_model=base_model,
            initial_output_dir=output_dir,
            num_iterations=num_iterations,
            num_epochs_per_iter=epochs_per_iter,
            n_responses_per_prompt=n_responses,
            temperature=1.0,
            batch_size=batch_size,
            learning_rate=5e-5,
            beta=0.1,
            n_eval_samples=n_eval_samples,
            verbose=True
        )
        
        print("\n" + "="*70)
        print(" PAPER REPRODUCTION RESULTS")
        print("="*70)
        
        baseline_rate = results[0]['satisfaction_rate']
        final_rate = results[-1]['satisfaction_rate']
        improvement = final_rate - baseline_rate
        
        print(f"\nBase Model Satisfaction Rate: {baseline_rate:.1f}%")
        print(f"Fine-Tuned Model Satisfaction Rate: {final_rate:.1f}%")
        print(f"Improvement: {improvement:+.1f}%")
        
        print("\n" + "─"*70)
        print(" COMPARISON TO PAPER")
        print("─"*70)
        print(f"Paper reports: 60% → 90% (30% improvement)")
        print(f"This run: {baseline_rate:.1f}% → {final_rate:.1f}% ({improvement:+.1f}% improvement)")
        
        if improvement > 15:
            print("\n✓ Successfully reproduced significant improvement!")
            print("  The model learned to satisfy safety specifications better.")
        else:
            print("\n⚠ Improvement is modest. For better results:")
            print("  - Run without --quick flag")
            print("  - Increase number of iterations")
            print("  - Use more training data")
        
        print("\n" + "="*70)
        print(" REPRODUCTION COMPLETE")
        print("="*70)
        print(f"\nTrained model saved to: {output_dir}_iteration_{num_iterations}")
        print(f"Training history saved to: {output_dir}_history.json")
        
    except KeyboardInterrupt:
        print("\n\nReproduction interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error during reproduction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce results from 'Fine-Tuning Language Models Using Formal Methods Feedback'"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model to use (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced parameters for faster execution"
    )
    
    args = parser.parse_args()
    
    reproduce_paper_results(
        base_model=args.model,
        quick_mode=args.quick
    )


if __name__ == "__main__":
    main()

