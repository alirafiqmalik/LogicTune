"""
LogicTune: Fine-Tuning Language Models Using Formal Methods Feedback

Main script demonstrating the complete pipeline for training language models
with automated verification-based feedback using Direct Preference Optimization.

This reproduces results from the paper "Fine-Tuning Language Models Using 
Formal Methods Feedback" which combines techniques from:
- DPO (Direct Preference Optimization)
- Formal methods for automated feedback
- LoRA for efficient fine-tuning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from logictune import (
    build_traffic_intersection_model,
    parse_response_to_fsa,
    score_response,
    DPODatasetGenerator,
    train_dpo,
    test_trained_model,
    compare_models,
    evaluate_model_simple,
    iterative_training
)


def demo_verification_pipeline():
    """
    Demonstrate the formal verification pipeline on example responses.
    """
    print("="*70)
    print(" STEP 1: FORMAL VERIFICATION DEMO")
    print("="*70)
    
    # Build the environment model
    print("\nBuilding traffic intersection model...")
    system = build_traffic_intersection_model()
    print(f"✓ System created with {len(system.get_all_states())} states")
    
    # Test responses
    test_responses = {
        "Safe Controller": """
1. If the light is green, go straight through the intersection.
2. If the light is yellow, slow down and stop.
3. If the light is red, stop and wait.
        """,
        "Unsafe Controller": """
1. Always go straight regardless of the light color.
2. Speed through yellow lights.
3. Turn left on red lights.
        """
    }
    
    print("\nTesting controller responses:\n")
    for name, response in test_responses.items():
        print(f"{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        print(f"Response: {response.strip()}\n")
        
        # Parse to FSA
        controller_fsa = parse_response_to_fsa(response, verbose=False)
        
        # Score with formal verification
        score, results = score_response(system, controller_fsa, verbose=True)
        
        print(f"\nFinal Score: {score}/15\n")


def generate_training_dataset(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_path: str = "dpo_dataset.jsonl",
    n_responses: int = 4,
    temperature: float = 1.0
):
    """
    Generate DPO training dataset with formal verification feedback.
    
    Args:
        model_name: Base model to use for generation
        output_path: Where to save the dataset
        n_responses: Number of responses per prompt
        temperature: Sampling temperature for diversity
    """
    print("\n" + "="*70)
    print(" STEP 2: GENERATE DPO DATASET")
    print("="*70)
    
    generator = DPODatasetGenerator(model_name=model_name)
    
    generator.generate_dataset(
        output_path=output_path,
        n_responses_per_prompt=n_responses,
        temperature=temperature,
        max_pairs_per_prompt=3
    )
    
    print(f"\n✓ Dataset saved to: {output_path}")


def train_model(
    dataset_path: str = "dpo_dataset.jsonl",
    output_dir: str = "dpo_model",
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    beta: float = 0.1
):
    """
    Train model using DPO with the generated dataset.
    
    Args:
        dataset_path: Path to DPO dataset
        output_dir: Where to save trained model
        model_name: Base model to fine-tune
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        beta: DPO beta parameter
    """
    print("\n" + "="*70)
    print(" STEP 3: TRAIN WITH DPO")
    print("="*70)
    
    train_dpo(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        beta=beta,
        use_quantization=False
    )
    
    print(f"\n✓ Model saved to: {output_dir}")


def test_model(
    model_path: str = "dpo_model",
    test_prompt: str = "Generate a step-by-step controller for safely navigating a traffic intersection."
):
    """
    Test the trained model and verify its output.
    
    Args:
        model_path: Path to trained model
        test_prompt: Prompt to test with
    """
    print("\n" + "="*70)
    print(" STEP 4: TEST TRAINED MODEL")
    print("="*70)
    
    print(f"\nPrompt: {test_prompt}\n")
    
    response = test_trained_model(model_path, test_prompt)
    print(f"Generated Response:\n{response}\n")
    
    print("="*70)
    print("VERIFYING GENERATED RESPONSE")
    print("="*70)
    
    system = build_traffic_intersection_model()
    try:
        controller_fsa = parse_response_to_fsa(response, verbose=False)
        score, results = score_response(system, controller_fsa, verbose=True)
        print(f"\n✓ Verification Score: {score}/15")
    except Exception as e:
        print(f"✗ Error during verification: {e}")


def evaluate_models(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    fine_tuned_model: str = "dpo_model",
    n_samples: int = 20,
    output_file: str = "evaluation_results.json"
):
    """
    Evaluate and compare base vs fine-tuned models.
    
    Args:
        base_model: Base model name/path
        fine_tuned_model: Fine-tuned model path
        n_samples: Number of test samples
        output_file: Output file for results
    """
    print("\n" + "="*70)
    print(" STEP 5: EVALUATE MODEL IMPROVEMENT")
    print("="*70)
    
    try:
        comparison = compare_models(
            base_model=base_model,
            fine_tuned_model=fine_tuned_model,
            test_prompts=None,
            n_samples_per_prompt=max(1, n_samples // 10),
            output_file=output_file
        )
        
        print("\n✓ Evaluation complete!")
        print(f"   Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def run_iterative_training(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "dpo_model_iterative",
    num_iterations: int = 3,
    epochs_per_iter: int = 2,
    n_responses: int = 4,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    beta: float = 0.1
):
    """
    Run iterative fine-tuning over multiple iterations.
    
    Args:
        model_name: Base model to start from
        output_dir: Output directory
        num_iterations: Number of iterations
        epochs_per_iter: Epochs per iteration
        n_responses: Responses per prompt
        batch_size: Batch size
        learning_rate: Learning rate
        beta: DPO beta parameter
    """
    print("\n" + "="*70)
    print(" ITERATIVE TRAINING MODE")
    print("="*70)
    
    try:
        results = iterative_training(
            base_model=model_name,
            initial_output_dir=output_dir,
            num_iterations=num_iterations,
            num_epochs_per_iter=epochs_per_iter,
            n_responses_per_prompt=n_responses,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta=beta,
            verbose=True
        )
        
        print("\n✓ Iterative training complete!")
        
    except Exception as e:
        print(f"\n✗ Error during iterative training: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main entry point for the LogicTune pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LogicTune: Fine-Tuning Language Models Using Formal Methods Feedback"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "generate", "train", "test", "evaluate", "iterative", "full"],
        default="demo",
        help="Mode to run: demo (verification only), generate (dataset), train, test, evaluate (compare models), iterative (multi-iteration training), or full (all steps)"
    )
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Model name")
    parser.add_argument("--dataset", type=str, default="dpo_dataset.jsonl",
                       help="Dataset path")
    parser.add_argument("--output", type=str, default="dpo_model",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO beta parameter")
    parser.add_argument("--n-responses", type=int, default=4,
                       help="Number of responses per prompt for dataset generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature for dataset generation")
    parser.add_argument("--n-eval-samples", type=int, default=20,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval-output", type=str, default="evaluation_results.json",
                       help="Output file for evaluation results")
    parser.add_argument("--num-iterations", type=int, default=3,
                       help="Number of iterations for iterative training")
    
    args = parser.parse_args()
    
    print("="*70)
    print(" LOGICTUNE: FINE-TUNING LMS WITH FORMAL METHODS FEEDBACK")
    print(" Reproducing results from paper:")
    print(" 'Fine-Tuning Language Models Using Formal Methods Feedback'")
    print("="*70)
    
    try:
        if args.mode == "demo" or args.mode == "full":
            demo_verification_pipeline()
        
        if args.mode == "generate" or args.mode == "full":
            generate_training_dataset(
                model_name=args.model,
                output_path=args.dataset,
                n_responses=args.n_responses,
                temperature=args.temperature
            )
        
        if args.mode == "train" or args.mode == "full":
            train_model(
                dataset_path=args.dataset,
                output_dir=args.output,
                model_name=args.model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                beta=args.beta
            )
        
        if args.mode == "test" or args.mode == "full":
            test_model(model_path=args.output)
        
        if args.mode == "evaluate" or args.mode == "full":
            evaluate_models(
                base_model=args.model,
                fine_tuned_model=args.output,
                n_samples=args.n_eval_samples,
                output_file=args.eval_output
            )
        
        if args.mode == "iterative":
            run_iterative_training(
                model_name=args.model,
                output_dir=args.output,
                num_iterations=args.num_iterations,
                epochs_per_iter=args.epochs,
                n_responses=args.n_responses,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                beta=args.beta
            )
        
        print("\n" + "="*70)
        print(" ✓ COMPLETE!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

