"""
LogicTune: fine-tuning language models with formal-methods feedback.

Pipeline: an LLM writes a rule-based controller for a driving scenario, the
text is parsed into a finite-state controller (GLM2FSA-lite), the controller
is model checked against 15 LTL specifications (nuXmv, cross-checked by an
exact explicit-state checker), and verifier-ranked preference pairs fine-tune
the LLM with DPO.

Reference: Yang et al., "Fine-Tuning Language Models Using Formal Methods
Feedback: A Use Case in Autonomous Systems" (2023).

The light modules (parsing, specifications, verification, prompts) import
without PyTorch. Training and generation modules load on first use.
"""

from .environment import ACTIONS, PROPOSITIONS, describe_environment
from .glm2fsa import Controller, Rule, extract_rules, parse_condition, parse_action, parse_response_to_fsa
from .prompts import SCENARIOS, SYSTEM_PROMPT, eval_prompts, train_prompts
from .specs import NUM_SPECS, SPECS
from .verifier import (
    VerificationResult,
    controller_to_smv,
    find_nuxmv,
    format_result,
    score_response,
    verify_controller,
    verify_response,
)

__version__ = "2.0.0"

_LAZY = {
    "PipelineConfig": "config",
    "run_pipeline": "pipeline",
    "train_dpo": "trainer",
    "evaluate_model": "evaluator",
    "sample_responses": "generation",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(name)


__all__ = [
    "ACTIONS", "PROPOSITIONS", "describe_environment", "Controller", "Rule", "extract_rules",
    "parse_condition", "parse_action", "parse_response_to_fsa", "SCENARIOS", "SYSTEM_PROMPT",
    "eval_prompts", "train_prompts", "NUM_SPECS", "SPECS", "VerificationResult", "controller_to_smv",
    "find_nuxmv", "format_result", "score_response", "verify_controller", "verify_response",
    *_LAZY,
]
