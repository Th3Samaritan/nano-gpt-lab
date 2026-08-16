"""Evaluation package: unified loss, perplexity, generation.

The plan (section 27) demands ONE evaluation function used by every
experiment, with identical generation parameters, so that numbers reported
for A/B/C/D are comparable by construction.
"""
from src.evaluation.loss import estimate_loss
from src.evaluation.perplexity import perplexity_from_loss
from src.evaluation.generation import generate_text, load_tokenizer_for_run

__all__ = ["estimate_loss", "perplexity_from_loss", "generate_text",
           "load_tokenizer_for_run"]
