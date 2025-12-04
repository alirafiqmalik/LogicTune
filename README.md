# LogicTune

Fine-tuning language models using formal methods feedback for safe autonomous control policy generation.


## Link to Results

https://colab.research.google.com/drive/1GglJdSaECa6XSV3iF1mS0UaVGiYiyZZA?usp=sharing


## Overview

This code reproduces results from the paper **"Fine-Tuning Language Models Using Formal Methods Feedback"**. It combines Direct Preference Optimization (DPO) with automated formal verification to train language models that generate safe control policies.

**Key Techniques:**
- **DPO (Direct Preference Optimization)**: Preference learning without reward models ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))
- **Formal Verification**: Automated feedback using transition systems and safety specifications
- **LoRA**: Parameter-efficient fine-tuning ([Hu et al., 2021](https://arxiv.org/abs/2106.09685))

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Reproduce Paper Results
```bash
python reproduce_paper.py
```
This runs the complete pipeline and shows improvement in specification satisfaction rate (60% → 90% as reported in paper).

### Generate Visualization (No Training Required)
```bash
python visualize_training.py --demo
```
Creates sample plots showing Loss, Accuracy, Marginal Preference, and specification satisfaction analysis.

### Run Demo (No GPU Required)
```bash
python src/tests/demo.py
```

### Full Pipeline

**1. Verification Demo**
```bash
python main.py --mode demo
```

**2. Generate Training Dataset** (requires GPU)
```bash
python main.py --mode generate --n-responses 4
```

**3. Train Model** (requires GPU)
```bash
python main.py --mode train --epochs 3
```

**4. Test Trained Model**
```bash
python main.py --mode test
```

**5. Evaluate and Compare Models**
```bash
python main.py --mode evaluate
```

**6. Iterative Training** (multiple refinement iterations)
```bash
python main.py --mode iterative --num-iterations 3
```

**7. Visualize Training Results**
```bash
# After training, generate visualization plots
python visualize_training.py --metrics dpo_model/training_metrics.json --eval evaluation_results.json
```


**Run All Steps**
```bash
python main.py --mode full
```

### Using Jupyter Notebook
```bash
jupyter notebook main.ipynb
```

## Library Usage

```python
from logictune import (
    build_traffic_intersection_model,
    parse_response_to_fsa,
    score_response,
    DPODatasetGenerator,
    train_dpo,
    compare_models,
    iterative_training
)

# Build environment model
system = build_traffic_intersection_model()

# Parse and verify a controller
response = "1. If green, go. 2. If yellow, stop. 3. If red, stop."
controller_fsa = parse_response_to_fsa(response)
score, results = score_response(system, controller_fsa, verbose=True)

# Generate training dataset
generator = DPODatasetGenerator()
generator.generate_dataset(output_path="dpo_dataset.jsonl")

# Train with DPO (single iteration)
train_dpo(
    dataset_path="dpo_dataset.jsonl",
    output_dir="trained_model"
)

# Evaluate improvement
comparison = compare_models(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    fine_tuned_model="trained_model"
)

# Or use iterative training (multiple refinement cycles)
results = iterative_training(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    initial_output_dir="dpo_model_iter",
    num_iterations=3
)
```

## Project Structure

```
LogicTune/
├── src/
│   ├── logictune/          # Core library
│   │   ├── __init__.py
│   │   ├── environment.py   # Transition system models
│   │   ├── parser.py        # LLM response → FSA (GLM2FSA)
│   │   ├── verifier.py      # Formal verification & scoring
│   │   ├── data_generator.py # DPO dataset generation
│   │   ├── trainer.py       # DPO training
│   │   ├── evaluator.py     # Model evaluation & comparison
│   │   └── iterative_trainer.py # Iterative refinement
│   └── tests/               # Test and demo scripts
├── main.py                  # Main execution script
├── main.ipynb               # Jupyter notebook
└── requirements.txt
```

## How It Works

1. **Environment Model**: Define transition system with safety specifications
2. **Response Generation**: LLM generates diverse control policies (high temperature)
3. **Formal Verification**: Each response scored against 15 LTL specifications (Φ1-Φ15)
4. **Preference Pairs**: Create (chosen, rejected) pairs based on scores
5. **DPO Training**: Fine-tune model to prefer high-scoring (safe) responses
6. **Metrics Tracking**: Automatic logging of loss, accuracy, marginal preference, and spec satisfaction
7. **Evaluation**: Compare base vs fine-tuned models on specification satisfaction rate
8. **Visualization**: Generate plots for training analysis and paper figures
9. **Iterative Refinement**: Optional multi-iteration training for continuous improvement

## LTL Specifications (Φ1-Φ15)

The verifier checks all 15 specifications from the paper:

| Spec | Description |
|------|-------------|
| Φ1 | If pedestrian present, eventually stop |
| Φ2 | No left turn when opposite car without green left-turn light |
| Φ3 | No going straight on non-green light |
| Φ4 | If stop sign, eventually stop |
| Φ5 | No right turn when car from left or pedestrian at right |
| Φ6 | Valid action always available |
| Φ7 | If green light, eventually proceed |
| Φ8 | If not green light, eventually stop |
| Φ9 | No turning when car from left |
| Φ10 | If green light, eventually proceed |
| Φ11 | Right turn on non-green only when no car from left |
| Φ12 | Left turn without green arrow only when clear |
| Φ13 | After stopping at sign with no cars, eventually proceed |
| Φ14 | No going straight with pedestrian in front |
| Φ15 | Right turn at stop sign only when no car from left |

## Visualization Features

LogicTune includes comprehensive visualization tools for analyzing training:

### Three Main Plot Types

1. **Training Metrics Line Charts** - Loss, Accuracy, Marginal Preference over 200 epochs with shaded error bands
2. **Box and Whisker Plot** - Specification satisfaction distribution vs training epoch
3. **Grouped Bar Chart** - Per-specification satisfaction rates (Base vs Fine-Tuned)

All metrics are automatically tracked during training.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Matplotlib 3.5+ and Seaborn 0.12+ (for visualization)
- GPU recommended for training (CPU works for demo/verification)

## Citation

This work implements techniques from:
- **Main Paper**: "Fine-Tuning Language Models Using Formal Methods Feedback"
- **DPO**: Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)
- **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)

