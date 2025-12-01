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
    
    print("Loading reference model for DPO...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    
    # Use TrainingArguments for compatibility across all TRL versions
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100 if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",  # Updated from evaluation_strategy (deprecated in transformers>=4.44)
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
    )
    
    print("\nInitializing DPO Trainer...")
    # DPO-specific parameters passed directly to trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
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

