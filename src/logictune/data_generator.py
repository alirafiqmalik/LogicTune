"""
DPO Dataset Generation

Generates preference pairs for Direct Preference Optimization (DPO) using
automated formal verification feedback instead of human labels.

Reference: "Fine-Tuning Language Models Using Formal Methods Feedback"
DPO Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
"""

import json
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from .environment import build_traffic_intersection_model
from .parser import parse_response_to_fsa
from .verifier import score_response


class DPODatasetGenerator:
    """
    Generate DPO training pairs using formal verification as feedback.
    """
    
    def __init__(self, 
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 device: str = "auto"):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
        """
        print(f"Loading model: {model_name}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        print("Building formal verification system...")
        self.system = build_traffic_intersection_model()
        
        print("Setup complete!")
    
    def get_prompts(self) -> List[str]:
        """
        Define prompts for autonomous driving scenarios.
        
        Returns:
            List of prompt strings
        """
        # prompts = [
        #     "Generate a step-by-step controller for safely navigating a traffic intersection with lights. List the actions to take for each light color (green, yellow, red).",
        #     "Describe a control policy for an autonomous vehicle approaching a traffic light. What should the car do when the light is green, yellow, or red?",
        #     "Write numbered steps for a safe driving controller at an intersection. Consider green light, yellow light, and red light scenarios.",
        #     "Create a decision-making procedure for driving through an intersection. Specify actions for different traffic light colors.",
        #     "List the rules for an autonomous car controller at a traffic intersection. Include behavior for green, yellow, and red lights.",
        #     "Design a control algorithm for intersection navigation. Describe what actions to take based on traffic light state.",
        # ]
        # Load prompts from JSON file
        import json
        from pathlib import Path
        
        prompts_file = Path(__file__).parent / "prompts.json"
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            prompts = data['prompts']
        return prompts
    
    def format_prompt(self, prompt: str) -> str:
        """
        Format prompt for the specific model being used.
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            Formatted prompt for model
        """
        output_format= "Generate a response that aligns with the defined Boolean Propositions {    green_traffic_light,green_left_turn_light,opposite_car,car_from_left,car_from_right,pedestrian_at_left,pedestrian_at_right,pedestrian_in_front,side_car,stop_sign} and Actions {stop, go_straight, turn_left, turn_right}"
        if "TinyLlama" in self.model_name:
            formatted = f"<|system|>\nYou are a helpful assistant. Always answer as helpfully as possible, while being safe. Your answers should be detailed. </s>\n<|user|>\n{prompt}</s>\n${output_format}\n</s>\n<|assistant|>\n"
        else:
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        return formatted
    
    def generate_response(self, 
                         prompt: str, 
                         temperature: float = 1.0,
                         max_new_tokens: int = 200,
                         top_p: float = 0.9) -> str:
        """
        Generate a single response from the model.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text response
        """
        formatted_prompt = self.format_prompt(prompt)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in full_text:
            response = full_text.split("<|assistant|>")[-1].strip()
        elif "### Response:" in full_text:
            response = full_text.split("### Response:")[-1].strip()
        else:
            response = full_text[len(formatted_prompt):].strip()
        
        return response
    
    def generate_multiple_responses(self,
                                   prompt: str,
                                   n: int = 4,
                                   temperature: float = 1.0) -> List[str]:
        """
        Generate multiple diverse responses for a single prompt.
        
        Args:
            prompt: Input prompt
            n: Number of responses to generate
            temperature: Sampling temperature
            
        Returns:
            List of response strings
        """
        responses = []
        for i in range(n):
            response = self.generate_response(prompt, temperature=temperature)
            responses.append(response)
        return responses
    
    def score_responses(self, 
                       responses: List[str],
                       verbose: bool = False) -> List[Tuple[str, int, Dict]]:
        """
        Score all responses using formal verification.
        
        Args:
            responses: List of response texts
            verbose: Print scoring details
            
        Returns:
            List of (response, score, details) tuples
        """
        scored = []
        
        for i, response in enumerate(responses):
            try:
                controller_fsa = parse_response_to_fsa(response, verbose=False)
                score, details = score_response(
                    self.system, 
                    controller_fsa, 
                    verbose=verbose
                )
                scored.append((response, score, details))
                
            except Exception as e:
                if verbose:
                    print(f"Error scoring response {i}: {e}")
                scored.append((response, 0, {}))
        
        return scored
    
    def create_preference_pairs(self,
                               scored_responses: List[Tuple[str, int, Dict]]
                               ) -> List[Tuple[str, str, int, int]]:
        """
        Create preference pairs from scored responses.
        
        Args:
            scored_responses: List of (response, score, details) tuples
            
        Returns:
            List of (y_w, y_l, score_w, score_l) tuples
        """
        pairs = []
        
        sorted_responses = sorted(scored_responses, key=lambda x: x[1], reverse=True)
        
        for i in range(len(sorted_responses)):
            for j in range(i + 1, len(sorted_responses)):
                y_w, score_w, _ = sorted_responses[i]
                y_l, score_l, _ = sorted_responses[j]
                
                if score_w > score_l:
                    pairs.append((y_w, y_l, score_w, score_l))
        
        return pairs
    
    def generate_dataset(self,
                        output_path: str = "dpo_dataset.jsonl",
                        n_responses_per_prompt: int = 4,
                        temperature: float = 1.0,
                        max_pairs_per_prompt: int = 3) -> None:
        """
        Generate complete DPO dataset.
        
        Args:
            output_path: Path to save JSONL dataset
            n_responses_per_prompt: Number of responses to generate per prompt
            temperature: Sampling temperature
            max_pairs_per_prompt: Maximum preference pairs to save per prompt
        """
        prompts = self.get_prompts()
        
        all_pairs = []
        
        print(f"\nGenerating DPO dataset...")
        print(f"Prompts: {len(prompts)}")
        print(f"Responses per prompt: {n_responses_per_prompt}")
        print(f"Temperature: {temperature}")
        print(f"="*60)
        
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
            print(f"\n{'='*60}")
            print(f"Prompt {prompt_idx + 1}/{len(prompts)}")
            print(f"{'='*60}")
            print(f"{prompt[:100]}...")
            
            print(f"\nGenerating {n_responses_per_prompt} responses...")
            responses = self.generate_multiple_responses(
                prompt, 
                n=n_responses_per_prompt,
                temperature=temperature
            )
            
            print(f"Scoring responses with formal verification...")
            scored = self.score_responses(responses, verbose=False)
            
            print(f"\nScores:")
            for i, (resp, score, _) in enumerate(scored):
                print(f"  Response {i+1}: {score}/15 - {resp[:80]}...")
            
            pairs = self.create_preference_pairs(scored)
            
            if len(pairs) > max_pairs_per_prompt:
                pairs = sorted(pairs, key=lambda x: x[2] - x[3], reverse=True)
                pairs = pairs[:max_pairs_per_prompt]
            
            print(f"Created {len(pairs)} preference pairs")
            
            for y_w, y_l, score_w, score_l in pairs:
                all_pairs.append({
                    "prompt": prompt,
                    "chosen": y_w,
                    "rejected": y_l,
                    "score_chosen": score_w,
                    "score_rejected": score_l,
                    "score_diff": score_w - score_l
                })
        
        print(f"\n{'='*60}")
        print(f"Saving dataset to {output_path}")
        print(f"Total pairs: {len(all_pairs)}")
        
        with open(output_path, 'w') as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + '\n')
        
        print(f"\nDataset Statistics:")
        print(f"  Total pairs: {len(all_pairs)}")
        if all_pairs:
            avg_diff = sum(p['score_diff'] for p in all_pairs) / len(all_pairs)
            print(f"  Average score difference: {avg_diff:.2f}")
        
        print(f"\n{'='*60}")
        print(f"Dataset generation complete!")
        print(f"{'='*60}")

