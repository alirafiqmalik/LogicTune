"""
Evaluation Module

Evaluates and compares base vs fine-tuned models to measure improvement
in specification satisfaction rates, as reported in the paper.

Reference: "Fine-Tuning Language Models Using Formal Methods Feedback"
"""

import torch
import json
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from .environment import build_traffic_intersection_model
from .parser import parse_response_to_fsa
from .verifier import score_response


class ModelEvaluator:
    """
    Evaluates models on specification satisfaction rate.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Args:
            device: Device to run models on
        """
        self.device = device
        self.system = build_traffic_intersection_model()
        
    def get_test_prompts(self) -> List[str]:
        """
        Get test prompts for evaluation.
        
        Returns:
            List of test prompt strings
        """
        prompts = [
            "Generate a step-by-step controller for safely navigating a traffic intersection with lights.",
            "Describe a control policy for an autonomous vehicle at a traffic light intersection.",
            "Write numbered steps for safe driving through an intersection with traffic signals.",
            "Create a decision procedure for driving through an intersection based on light colors.",
            "List the rules for a self-driving car controller at a traffic intersection.",
            "Design a control algorithm for intersection navigation with traffic lights.",
            "Provide instructions for an autonomous vehicle to safely cross an intersection.",
            "What should a self-driving car do at each traffic light color?",
            "Generate a control policy for intersection crossing with green, yellow, and red lights.",
            "Describe safe driving behavior at a traffic intersection for each light state.",
        ]
        return prompts
    
    def format_prompt(self, prompt: str, model_name: str) -> str:
        """
        Format prompt for the specific model.
        
        Args:
            prompt: Raw prompt text
            model_name: Model name/path
            
        Returns:
            Formatted prompt
        """
        if "TinyLlama" in model_name or "tinyllama" in model_name.lower():
            formatted = f"<|system|>\nYou are a helpful AI assistant that generates control policies for autonomous vehicles.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        else:
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        return formatted
    
    def generate_response(self,
                         model,
                         tokenizer,
                         prompt: str,
                         model_name: str,
                         temperature: float = 0.7,
                         max_new_tokens: int = 200) -> str:
        """
        Generate response from a model.
        
        Args:
            model: Model to generate from
            tokenizer: Tokenizer
            prompt: Input prompt
            model_name: Model name/path
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        formatted_prompt = self.format_prompt(prompt, model_name)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in full_text:
            response = full_text.split("<|assistant|>")[-1].strip()
        elif "### Response:" in full_text:
            response = full_text.split("### Response:")[-1].strip()
        else:
            response = full_text[len(formatted_prompt):].strip()
        
        return response
    
    def evaluate_model(self,
                      model_path: str,
                      test_prompts: Optional[List[str]] = None,
                      n_samples_per_prompt: int = 5,
                      verbose: bool = True) -> Dict:
        """
        Evaluate a model on specification satisfaction rate.
        
        Args:
            model_path: Path to model or HuggingFace model name
            test_prompts: List of test prompts (uses default if None)
            n_samples_per_prompt: Number of responses per prompt
            verbose: Print detailed progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        if verbose:
            print(f"\nEvaluating model: {model_path}")
            print("="*70)
        
        if test_prompts is None:
            test_prompts = self.get_test_prompts()
        
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        total_responses = 0
        total_score = 0
        max_possible_score = 0
        
        all_results = []
        
        iterator = tqdm(test_prompts, desc="Evaluating") if verbose else test_prompts
        
        for prompt in iterator:
            for sample_idx in range(n_samples_per_prompt):
                try:
                    response = self.generate_response(
                        model, tokenizer, prompt, model_path
                    )
                    
                    controller_fsa = parse_response_to_fsa(response, verbose=False)
                    score, details = score_response(
                        self.system,
                        controller_fsa,
                        verbose=False
                    )
                    
                    total_responses += 1
                    total_score += score
                    max_possible_score += 3
                    
                    all_results.append({
                        'prompt': prompt,
                        'response': response,
                        'score': score,
                        'details': details
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"Error evaluating response: {e}")
                    total_responses += 1
                    max_possible_score += 3
                    all_results.append({
                        'prompt': prompt,
                        'response': '',
                        'score': 0,
                        'details': {},
                        'error': str(e)
                    })
        
        satisfaction_rate = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        avg_score = total_score / total_responses if total_responses > 0 else 0
        
        results = {
            'model_path': model_path,
            'total_responses': total_responses,
            'total_score': total_score,
            'max_possible_score': max_possible_score,
            'satisfaction_rate': satisfaction_rate,
            'avg_score_per_response': avg_score,
            'all_results': all_results
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"EVALUATION RESULTS")
            print(f"{'='*70}")
            print(f"Total Responses: {total_responses}")
            print(f"Total Score: {total_score}/{max_possible_score}")
            print(f"Specification Satisfaction Rate: {satisfaction_rate:.1f}%")
            print(f"Average Score per Response: {avg_score:.2f}/3.0")
            print(f"{'='*70}\n")
        
        return results


def compare_models(base_model: str,
                   fine_tuned_model: str,
                   test_prompts: Optional[List[str]] = None,
                   n_samples_per_prompt: int = 5,
                   output_file: Optional[str] = None) -> Dict:
    """
    Compare base model vs fine-tuned model.
    
    Args:
        base_model: Path to base model
        fine_tuned_model: Path to fine-tuned model
        test_prompts: Test prompts (uses default if None)
        n_samples_per_prompt: Number of samples per prompt
        output_file: Optional file to save results
        
    Returns:
        Dictionary with comparison results
    """
    print("="*70)
    print(" MODEL COMPARISON: BASE vs FINE-TUNED")
    print("="*70)
    
    evaluator = ModelEvaluator()
    
    print("\n" + "="*70)
    print(" EVALUATING BASE MODEL")
    print("="*70)
    base_results = evaluator.evaluate_model(
        base_model,
        test_prompts=test_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        verbose=True
    )
    
    print("\n" + "="*70)
    print(" EVALUATING FINE-TUNED MODEL")
    print("="*70)
    fine_tuned_results = evaluator.evaluate_model(
        fine_tuned_model,
        test_prompts=test_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        verbose=True
    )
    
    improvement = fine_tuned_results['satisfaction_rate'] - base_results['satisfaction_rate']
    
    comparison = {
        'base_model': base_results,
        'fine_tuned_model': fine_tuned_results,
        'improvement': improvement
    }
    
    print("\n" + "="*70)
    print(" COMPARISON SUMMARY")
    print("="*70)
    print(f"Base Model Satisfaction Rate: {base_results['satisfaction_rate']:.1f}%")
    print(f"Fine-Tuned Model Satisfaction Rate: {fine_tuned_results['satisfaction_rate']:.1f}%")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"{'='*70}\n")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return comparison


def evaluate_model_simple(model_path: str,
                         n_test_samples: int = 20,
                         verbose: bool = True) -> float:
    """
    Simple evaluation function that returns satisfaction rate.
    
    Args:
        model_path: Path to model
        n_test_samples: Total number of test samples
        verbose: Print progress
        
    Returns:
        Specification satisfaction rate (0-100)
    """
    evaluator = ModelEvaluator()
    test_prompts = evaluator.get_test_prompts()
    
    n_samples_per_prompt = max(1, n_test_samples // len(test_prompts))
    
    results = evaluator.evaluate_model(
        model_path,
        test_prompts=test_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        verbose=verbose
    )
    
    return results['satisfaction_rate']

