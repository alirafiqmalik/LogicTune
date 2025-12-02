"""
DPO Training Module

Implements Direct Preference Optimization training using preference pairs
generated from formal verification feedback.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
Training techniques: LoRA (Low-Rank Adaptation), Quantization for efficiency
"""

import json
import torch
import os
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer

# Try to import DPOConfig for newer TRL versions
try:
    from trl import DPOConfig
    HAS_DPO_CONFIG = True
except ImportError:
    HAS_DPO_CONFIG = False


def load_dpo_dataset(file_path: str) -> Dataset:
    """
    Load DPO dataset from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        HuggingFace Dataset object
    """
    print(f"Loading dataset from {file_path}...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} preference pairs")
    
    dataset = Dataset.from_list(data)
    
    if len(data) > 0:
        avg_score_diff = sum(d['score_diff'] for d in data) / len(data)
        print(f"Average score difference: {avg_score_diff:.2f}")
    
    return dataset


def setup_model_and_tokenizer(
    model_name: str,
    use_quantization: bool = False
) -> Tuple:
    """
    Setup model and tokenizer for DPO training.
    
    Args:
        model_name: HuggingFace model identifier
        use_quantization: Whether to use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer, peft_config)
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    print("Model loaded and configured for LoRA training")
    
    return model, tokenizer, peft_config


def train_dpo(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset_path: str = "dpo_dataset.jsonl",
    output_dir: str = "dpo_model",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    use_quantization: bool = False,
    max_length: int = 512,
    max_prompt_length: int = 256,
) -> None:
    """
    Main DPO training function.
    
    Args:
        model_name: Base model to fine-tune
        dataset_path: Path to DPO dataset JSONL
        output_dir: Directory to save trained model
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        beta: DPO beta parameter
        use_quantization: Use 4-bit quantization
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
    """
    print("="*60)
    print("DPO TRAINING WITH FORMAL VERIFICATION FEEDBACK")
    print("="*60)
    
    dataset = load_dpo_dataset(dataset_path)
    
    if len(dataset) > 10:
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    else:
        train_dataset = dataset
        eval_dataset = None
        print("Dataset too small for split, using all data for training")
    
    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation samples: {len(eval_dataset)}")
    
    model, tokenizer, peft_config = setup_model_and_tokenizer(
        model_name,
        use_quantization=use_quantization
    )
    
    # Create callback for tracking training metrics and specification satisfaction
    from .training_tracker import create_visualization_callback
    
    eval_prompts = [
        "Generate a step-by-step controller for safely navigating a traffic intersection.",
        "Describe a control policy for an autonomous vehicle at a traffic light.",
        "Write steps for safe driving through an intersection with traffic signals."
    ]
    
    metrics_callback = create_visualization_callback(
        output_dir=output_dir,
        tokenizer=tokenizer,
        model_name=model_name,
        eval_prompts=eval_prompts
    )
    
    # Note: When using PEFT (LoRA), newer TRL versions automatically use the base model
    # as reference, so we don't need to load a separate ref_model.
    # We'll handle this in the DPOTrainer initialization.
    ref_model = None  # Will be set conditionally based on TRL version requirements
    
    # Configure training arguments - try DPOConfig first for DPO-specific params
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100 if eval_dataset else None,
        "eval_strategy": "steps" if eval_dataset else "no",
        "save_total_limit": 3,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "remove_unused_columns": False,
        "report_to": "none",
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
    }
    
    # Try to use DPOConfig if available (newer TRL versions)
    if HAS_DPO_CONFIG:
        try:
            training_args = DPOConfig(
                **training_args_dict,
                beta=beta,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
            )
            print(f"Using DPOConfig with beta={beta}")
        except TypeError:
            # DPOConfig doesn't support these params, fall back to TrainingArguments
            training_args = TrainingArguments(**training_args_dict)
    else:
        training_args = TrainingArguments(**training_args_dict)
    
    print("\nInitializing DPO Trainer...")
    # Initialize DPOTrainer with comprehensive version-agnostic approach
    # Different TRL versions have different parameter names and requirements
    
    # Check if beta is already in config (DPOConfig was used)
    beta_in_config = hasattr(training_args, 'beta')
    
    # Try different parameter combinations for maximum compatibility across TRL versions
    # 
    # Key version differences:
    # - Newer TRL (0.11+): processing_class instead of tokenizer, ref_model=None when using PEFT
    # - Mid TRL (0.8-0.10): tokenizer parameter, may or may not need ref_model with PEFT
    # - Older TRL (0.7.x): tokenizer, beta as params, needs ref_model even with PEFT
    #
    # PEFT + ref_model behavior:
    # - Newer versions: When using peft_config, ref_model should be None (base model used automatically)
    # - Older versions: ref_model required even with peft_config
    
    def try_init_trainer(**kwargs):
        """Try to initialize trainer with given kwargs, handling version differences"""
        try:
            return DPOTrainer(**kwargs), True
        except (TypeError, ValueError) as e:
            error_msg = str(e)
            
            # Handle PEFT + ref_model conflict (newer TRL)
            if "ref_model" in error_msg and "peft_config" in error_msg:
                if 'ref_model' in kwargs and kwargs['ref_model'] is not None:
                    kwargs_copy = kwargs.copy()
                    kwargs_copy['ref_model'] = None
                    try:
                        return DPOTrainer(**kwargs_copy), True
                    except (TypeError, ValueError):
                        pass
            
            # Handle specific parameter issues
            if "'tokenizer'" in error_msg:
                # Try with processing_class instead (newer TRL)
                if 'tokenizer' in kwargs:
                    kwargs_copy = kwargs.copy()
                    kwargs_copy['processing_class'] = kwargs_copy.pop('tokenizer')
                    try:
                        return DPOTrainer(**kwargs_copy), True
                    except (TypeError, ValueError):
                        pass
            elif "'processing_class'" in error_msg:
                # Try with tokenizer instead (older TRL)
                if 'processing_class' in kwargs:
                    kwargs_copy = kwargs.copy()
                    kwargs_copy['tokenizer'] = kwargs_copy.pop('processing_class')
                    try:
                        return DPOTrainer(**kwargs_copy), True
                    except (TypeError, ValueError):
                        pass
            
            # If it's about beta/max_length/max_prompt_length, try without them
            if any(param in error_msg for param in ['beta', 'max_length', 'max_prompt_length']):
                kwargs_copy = {k: v for k, v in kwargs.items() 
                              if k not in ['beta', 'max_length', 'max_prompt_length']}
                try:
                    return DPOTrainer(**kwargs_copy), False
                except (TypeError, ValueError):
                    pass
            
            return None, False
    
    # Base kwargs - start with ref_model=None for PEFT compatibility (newer TRL)
    base_kwargs = {
        "model": model,
        "ref_model": ref_model,  # None by default for PEFT
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
        "callbacks": [metrics_callback],  # Add metrics tracking callback
    }
    
    if eval_dataset is not None:
        base_kwargs["eval_dataset"] = eval_dataset
    
    # Try initialization with different parameter combinations
    trainer = None
    
    # Attempt 1: Try with processing_class and no ref_model (newest TRL with PEFT)
    if not beta_in_config:
        kwargs = {**base_kwargs, "processing_class": tokenizer,
                 "beta": beta, "max_length": max_length, "max_prompt_length": max_prompt_length}
        trainer, success = try_init_trainer(**kwargs)
        if trainer:
            print(f"Initialized DPOTrainer with beta={beta} (processing_class, PEFT mode)")
    
    # Attempt 2: Try with tokenizer and no ref_model (older TRL with PEFT)
    if trainer is None and not beta_in_config:
        kwargs = {**base_kwargs, "tokenizer": tokenizer, 
                 "beta": beta, "max_length": max_length, "max_prompt_length": max_prompt_length}
        trainer, success = try_init_trainer(**kwargs)
        if trainer:
            print(f"Initialized DPOTrainer with beta={beta} (tokenizer, PEFT mode)")
    
    # Attempt 3: Beta already in config, try with processing_class
    if trainer is None and beta_in_config:
        kwargs = {**base_kwargs, "processing_class": tokenizer}
        trainer, success = try_init_trainer(**kwargs)
        if trainer:
            print(f"Initialized DPOTrainer with DPOConfig (processing_class, PEFT mode)")
    
    # Attempt 4: Beta already in config, try with tokenizer
    if trainer is None and beta_in_config:
        kwargs = {**base_kwargs, "tokenizer": tokenizer}
        trainer, success = try_init_trainer(**kwargs)
        if trainer:
            print(f"Initialized DPOTrainer with DPOConfig (tokenizer, PEFT mode)")
    
    # Attempt 5: Try loading ref_model for older TRL that needs it even with PEFT
    if trainer is None and ref_model is None:
        print("Loading reference model (required by TRL version)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        ref_model.eval()
        base_kwargs["ref_model"] = ref_model
        
        # Try with ref_model + tokenizer
        if not beta_in_config:
            kwargs = {**base_kwargs, "tokenizer": tokenizer,
                     "beta": beta, "max_length": max_length, "max_prompt_length": max_prompt_length}
        else:
            kwargs = {**base_kwargs, "tokenizer": tokenizer}
        trainer, success = try_init_trainer(**kwargs)
        if trainer:
            print(f"Initialized DPOTrainer with ref_model (older TRL version)")
    
    # Attempt 6: No tokenizer parameter at all
    if trainer is None:
        if not beta_in_config:
            kwargs = {**base_kwargs, "beta": beta, "max_length": max_length, "max_prompt_length": max_prompt_length}
        else:
            kwargs = base_kwargs
        trainer, success = try_init_trainer(**kwargs)
        if trainer:
            print(f"Initialized DPOTrainer (minimal parameters)")
    
    if trainer is None:
        raise RuntimeError(
            "Failed to initialize DPOTrainer with any parameter combination. "
            "This may indicate an incompatible TRL version. "
            f"Please try: pip install 'trl>=0.7.4,<0.12.0'"
    )
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    trainer.train()
    
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel saved to: {output_dir}")
    print("Training complete!")
    
    info = {
        "base_model": model_name,
        "dataset": dataset_path,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "beta": beta,
        "training_samples": len(train_dataset),
    }
    
    with open(os.path.join(output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)


def test_trained_model(model_path: str, prompt: str) -> str:
    """
    Test the trained model on a prompt.
    
    Args:
        model_path: Path to trained model directory
        prompt: Test prompt
        
    Returns:
        Generated response
    """
    print(f"Loading trained model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    
    if "TinyLlama" in model_path:
        formatted = f"<|system|>\nYou are a helpful AI assistant that generates control policies for autonomous vehicles.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    else:
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    elif "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

