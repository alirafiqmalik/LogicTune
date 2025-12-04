"""
Training Tracker

Tracks detailed metrics during DPO training including loss, accuracy,
marginal preference, and specification satisfaction.

Integrates with TRL DPOTrainer via callbacks.
"""

import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl

from .environment import build_traffic_intersection_model
from .parser import parse_response_to_fsa
from .verifier import score_response, get_traffic_safety_specs


class DPOMetricsCallback(TrainerCallback):
    """
    Callback to track DPO-specific metrics during training.
    """
    
    def __init__(self, output_dir: str, eval_prompts: List[str], 
                 tokenizer, model_name: str):
        """
        Args:
            output_dir: Directory to save metrics
            eval_prompts: Prompts for periodic evaluation
            tokenizer: Tokenizer for generating responses
            model_name: Model name for prompt formatting
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.eval_prompts = eval_prompts
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        self.system = build_traffic_intersection_model()
        self.specs = get_traffic_safety_specs()
        
        # Metrics storage
        self.metrics = {
            'steps': [],
            'loss': [],
            'accuracy': [],
            'marginal_preference': [],
            'learning_rate': []
        }
        
        self.epoch_scores = {}
        self.per_spec_satisfaction = {}
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """
        Called when trainer logs metrics.
        """
        if logs is None:
            return
        
        # Track basic metrics
        if 'loss' in logs:
            self.metrics['steps'].append(state.global_step)
            self.metrics['loss'].append(logs['loss'])
            
            # DPO-specific metrics from TRL trainer
            # Look for the actual accuracy metric logged by DPOTrainer
            if 'rewards/accuracies' in logs:
                # TRL logs actual batch accuracy
                accuracy = logs['rewards/accuracies'] * 100.0
                self.metrics['accuracy'].append(accuracy)
            elif 'train/accuracy' in logs:
                self.metrics['accuracy'].append(logs['train/accuracy'] * 100.0)
            elif 'rewards/chosen' in logs and 'rewards/rejected' in logs:
                # Compute implicit accuracy from rewards
                chosen_reward = logs['rewards/chosen']
                rejected_reward = logs['rewards/rejected']
                
                # Use sigmoid to estimate accuracy from margin
                margin = chosen_reward - rejected_reward
                import math
                accuracy = 100.0 / (1.0 + math.exp(-margin)) if abs(margin) < 20 else (100.0 if margin > 0 else 0.0)
                self.metrics['accuracy'].append(accuracy)
            else:
                # Fallback - use loss-based estimate
                loss = logs['loss']
                # DPO loss of 0.69 ≈ 50% accuracy, 0 = 100%
                accuracy = max(0, min(100, 100 * (1 - loss / 0.7)))
                self.metrics['accuracy'].append(accuracy)
            
            # Marginal preference (reward margin)
            if 'rewards/margins' in logs:
                self.metrics['marginal_preference'].append(logs['rewards/margins'])
            elif 'rewards/chosen' in logs and 'rewards/rejected' in logs:
                margin = logs['rewards/chosen'] - logs['rewards/rejected']
                self.metrics['marginal_preference'].append(margin)
            else:
                # Estimate from loss - lower loss = higher margin
                loss = logs['loss']
                margin = max(0, 3.0 * (0.7 - loss) / 0.7) if loss < 0.7 else 0.0
                self.metrics['marginal_preference'].append(margin)
            
            if 'learning_rate' in logs:
                self.metrics['learning_rate'].append(logs['learning_rate'])
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each epoch - evaluate specification satisfaction.
        """
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        print(f"\n{'─'*70}")
        print(f" Evaluating Specification Satisfaction at Epoch {epoch}")
        print(f"{'─'*70}")
        
        model = kwargs.get('model')
        if model is None:
            return
        
        scores = []
        
        try:
            model.eval()
            
            # Evaluate on subset of prompts
            for prompt in self.eval_prompts[:5]:
                try:
                    # Format prompt
                    output_format= "Generate a response that aligns with the defined Boolean Propositions {    green_traffic_light,green_left_turn_light,opposite_car,car_from_left,car_from_right,pedestrian_at_left,pedestrian_at_right,pedestrian_in_front,side_car,stop_sign} and Actions {stop, go_straight, turn_left, turn_right}"
                    if "TinyLlama" in self.model_name:
                        formatted = f"<|system|>\nYou are a helpful assistant. Always answer as helpfully as possible, while being safe. Your answers should be detailed. </s>\n<|user|>\n{prompt}</s>\n<|output|>\n${output_format}</s>\n<|assistant|>\n"
                    else:
                        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
                    
                    # Generate response
                    inputs = self.tokenizer(formatted, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    import torch
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=150,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "<|assistant|>" in response:
                        response = response.split("<|assistant|>")[-1].strip()
                    elif "### Response:" in response:
                        response = response.split("### Response:")[-1].strip()
                    
                    # Verify
                    controller_fsa = parse_response_to_fsa(response, verbose=False)
                    score, _ = score_response(self.system, controller_fsa, verbose=False)
                    scores.append(score)
                    
                except Exception as e:
                    print(f"  Error evaluating prompt: {e}")
                    scores.append(0)
            
            model.train()
            
            self.epoch_scores[epoch] = scores
            avg_score = np.mean(scores) if scores else 0.0
            print(f"  Average Score: {avg_score:.2f}/15.0")
            print(f"  Scores: {scores}")
            
        except Exception as e:
            print(f"  Error during epoch evaluation: {e}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of training - save all metrics.
        """
        print(f"\n{'='*70}")
        print(" Saving Training Metrics")
        print(f"{'='*70}")
        
        # Convert steps to epochs for visualization
        # Group steps by epoch (assuming steps are logged per epoch)
        steps = self.metrics['steps']
        if steps:
            # Estimate epochs from steps (simplified: assume uniform distribution)
            # In practice, this should track actual epoch numbers
            num_epochs = len(set(steps)) if steps else max(steps) // 100 if steps else 200
            epochs = list(range(num_epochs))
        else:
            epochs = []
        
        # Save metrics
        metrics_file = self.output_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'training_metrics': {
                    'steps': self.metrics['steps'],
                    'epochs': epochs,  # Add epochs for visualization
                    'metrics': {
                        'loss': self.metrics['loss'],
                        'accuracy': self.metrics['accuracy'],
                        'marginal_preference': self.metrics['marginal_preference']
                    }
                },
                'epoch_scores': {str(k): v for k, v in self.epoch_scores.items()}
            }, f, indent=2)
        
        print(f"✓ Metrics saved to: {metrics_file}")


def create_visualization_callback(output_dir: str,
                                  tokenizer,
                                  model_name: str,
                                  eval_prompts: Optional[List[str]] = None) -> DPOMetricsCallback:
    """
    Create a callback for tracking and visualizing training metrics.
    
    Args:
        output_dir: Directory to save outputs
        tokenizer: Tokenizer
        model_name: Model name
        eval_prompts: Evaluation prompts (uses defaults if None)
        
    Returns:
        DPOMetricsCallback instance
    """
    if eval_prompts is None:
        eval_prompts = [
            "Generate a step-by-step controller for safely navigating a traffic intersection.",
            "Describe a control policy for an autonomous vehicle at a traffic light.",
            "Write steps for safe driving through an intersection with traffic signals."
        ]
    
    return DPOMetricsCallback(output_dir, eval_prompts, tokenizer, model_name)

