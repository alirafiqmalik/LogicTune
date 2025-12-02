"""
LogicTune: Fine-Tuning Language Models Using Formal Methods Feedback

This library implements techniques from the paper "Fine-Tuning Language Models 
Using Formal Methods Feedback" for training language models with automated 
verification-based feedback using Direct Preference Optimization (DPO).

Reference: "Fine-Tuning Language Models Using Formal Methods Feedback"
DPO Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
"""

from .environment import TransitionSystem, build_traffic_intersection_model
from .parser import ControllerParser, parse_response_to_fsa
from .verifier import (
    SafetySpec, 
    ProductAutomaton, 
    build_product_automaton,
    verify_safety_specs,
    score_response,
    get_traffic_safety_specs
)
from .data_generator import DPODatasetGenerator
from .trainer import (
    load_dpo_dataset,
    setup_model_and_tokenizer,
    train_dpo,
    test_trained_model
)
from .evaluator import (
    ModelEvaluator,
    compare_models,
    evaluate_model_simple
)
from .iterative_trainer import (
    iterative_training,
    iterative_training_with_curriculum
)
from .visualizer import (
    TrainingVisualizer,
    generate_sample_data_for_testing
)
from .training_tracker import (
    DPOMetricsCallback,
    create_visualization_callback
)

__version__ = "1.0.0"

__all__ = [
    "TransitionSystem",
    "build_traffic_intersection_model",
    "ControllerParser",
    "parse_response_to_fsa",
    "SafetySpec",
    "ProductAutomaton",
    "build_product_automaton",
    "verify_safety_specs",
    "score_response",
    "get_traffic_safety_specs",
    "DPODatasetGenerator",
    "load_dpo_dataset",
    "setup_model_and_tokenizer",
    "train_dpo",
    "test_trained_model",
    "ModelEvaluator",
    "compare_models",
    "evaluate_model_simple",
    "iterative_training",
    "iterative_training_with_curriculum",
    "TrainingVisualizer",
    "generate_sample_data_for_testing",
    "DPOMetricsCallback",
    "create_visualization_callback",
]

