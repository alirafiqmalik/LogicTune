"""
Iterative Fine-Tuning

Implements iterative refinement loop for continuous model improvement.
In each iteration, the model is evaluated, new preference pairs are generated,
and the model is further fine-tuned.

Reference: "Fine-Tuning Language Models Using Formal Methods Feedback"
"""

import os
import json
from typing import Optional, Dict, List
from .data_generator import DPODatasetGenerator
from .trainer import train_dpo
from .evaluator import evaluate_model_simple


def iterative_training(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    initial_output_dir: str = "dpo_model_iter",
    num_iterations: int = 3,
    num_epochs_per_iter: int = 2,
    n_responses_per_prompt: int = 4,
    temperature: float = 1.0,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    n_eval_samples: int = 20,
    verbose: bool = True
) -> List[Dict]:
    """
    Perform iterative fine-tuning with evaluation after each iteration.
    
    Args:
        base_model: Initial model to start from
        initial_output_dir: Base directory for outputs
        num_iterations: Number of training iterations
        num_epochs_per_iter: Training epochs per iteration
        n_responses_per_prompt: Responses per prompt for dataset generation
        temperature: Sampling temperature
        batch_size: Training batch size
        learning_rate: Learning rate
        beta: DPO beta parameter
        n_eval_samples: Number of samples for evaluation
        verbose: Print detailed progress
        
    Returns:
        List of evaluation results for each iteration
    """
    print("="*70)
    print(" ITERATIVE FINE-TUNING")
    print(" Training with formal methods feedback over multiple iterations")
    print("="*70)
    
    results_history = []
    current_model = base_model
    
    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f" ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")
        
        iteration_dir = f"{initial_output_dir}_iteration_{iteration + 1}"
        dataset_path = f"{iteration_dir}_dataset.jsonl"
        
        if iteration == 0:
            print(f"\nEvaluating base model: {base_model}")
            base_score = evaluate_model_simple(
                base_model,
                n_test_samples=n_eval_samples,
                verbose=verbose
            )
            
            results_history.append({
                'iteration': 0,
                'model': base_model,
                'satisfaction_rate': base_score,
                'is_baseline': True
            })
            
            print(f"Base model satisfaction rate: {base_score:.1f}%")
        
        print(f"\n{'─'*70}")
        print(f" Step 1: Generate Training Dataset")
        print(f"{'─'*70}")
        print(f"Generating from model: {current_model}")
        
        generator = DPODatasetGenerator(model_name=current_model)
        generator.generate_dataset(
            output_path=dataset_path,
            n_responses_per_prompt=n_responses_per_prompt,
            temperature=temperature,
            max_pairs_per_prompt=3
        )
        
        print(f"\n{'─'*70}")
        print(f" Step 2: Train Model")
        print(f"{'─'*70}")
        
        train_dpo(
            model_name=current_model,
            dataset_path=dataset_path,
            output_dir=iteration_dir,
            num_epochs=num_epochs_per_iter,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta=beta,
            use_quantization=False
        )
        
        print(f"\n{'─'*70}")
        print(f" Step 3: Evaluate Trained Model")
        print(f"{'─'*70}")
        
        trained_score = evaluate_model_simple(
            iteration_dir,
            n_test_samples=n_eval_samples,
            verbose=verbose
        )
        
        improvement = trained_score - results_history[-1]['satisfaction_rate']
        
        results_history.append({
            'iteration': iteration + 1,
            'model': iteration_dir,
            'satisfaction_rate': trained_score,
            'improvement_from_previous': improvement,
            'is_baseline': False
        })
        
        print(f"\n{'='*70}")
        print(f" ITERATION {iteration + 1} RESULTS")
        print(f"{'='*70}")
        print(f"Previous: {results_history[-2]['satisfaction_rate']:.1f}%")
        print(f"Current: {trained_score:.1f}%")
        print(f"Improvement: {improvement:+.1f}%")
        print(f"{'='*70}")
        
        current_model = iteration_dir
    
    print(f"\n{'='*70}")
    print(f" ITERATIVE TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nTraining History:")
    for result in results_history:
        iter_num = result['iteration']
        score = result['satisfaction_rate']
        if result['is_baseline']:
            print(f"  Iteration {iter_num} (Baseline): {score:.1f}%")
        else:
            improvement = result['improvement_from_previous']
            print(f"  Iteration {iter_num}: {score:.1f}% ({improvement:+.1f}%)")
    
    total_improvement = results_history[-1]['satisfaction_rate'] - results_history[0]['satisfaction_rate']
    print(f"\nTotal Improvement: {total_improvement:+.1f}%")
    print(f"{'='*70}\n")
    
    results_file = f"{initial_output_dir}_history.json"
    with open(results_file, 'w') as f:
        json.dump(results_history, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results_history


def iterative_training_with_curriculum(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "dpo_model_curriculum",
    num_iterations: int = 3,
    temperatures: Optional[List[float]] = None,
    num_epochs_per_iter: int = 2,
    **kwargs
) -> List[Dict]:
    """
    Iterative training with curriculum learning (decreasing temperature).
    
    Start with high temperature (diverse but possibly unsafe) and gradually
    decrease temperature (more focused on learned safe patterns).
    
    Args:
        base_model: Initial model
        output_dir: Output directory base
        num_iterations: Number of iterations
        temperatures: Temperature schedule (defaults to [1.2, 1.0, 0.8])
        num_epochs_per_iter: Epochs per iteration
        **kwargs: Additional arguments for iterative_training
        
    Returns:
        List of evaluation results
    """
    if temperatures is None:
        temperatures = [1.2, 1.0, 0.8]
    
    if len(temperatures) < num_iterations:
        last_temp = temperatures[-1]
        temperatures.extend([last_temp] * (num_iterations - len(temperatures)))
    
    print("="*70)
    print(" CURRICULUM-BASED ITERATIVE TRAINING")
    print(f" Temperature schedule: {temperatures[:num_iterations]}")
    print("="*70)
    
    results = []
    current_model = base_model
    
    for iteration in range(num_iterations):
        temp = temperatures[iteration]
        
        print(f"\n{'='*70}")
        print(f" ITERATION {iteration + 1}/{num_iterations} (Temperature: {temp})")
        print(f"{'='*70}")
        
        iter_dir = f"{output_dir}_iter_{iteration + 1}"
        
        result = iterative_training(
            base_model=current_model,
            initial_output_dir=iter_dir,
            num_iterations=1,
            num_epochs_per_iter=num_epochs_per_iter,
            temperature=temp,
            **kwargs
        )
        
        results.extend(result)
        current_model = iter_dir + "_iteration_1"
    
    return results

