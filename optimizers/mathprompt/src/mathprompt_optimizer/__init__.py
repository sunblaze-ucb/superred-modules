"""MathPrompt (symbolic-math encoding) jailbreak optimizer for superred.

Ported from DeepTeam (deepteam/attacks/single_turn/math_problem).
"""

from mathprompt_optimizer.optimizer import DEFAULT_MAX_RETRIES, MathPromptOptimizer

__all__ = ["DEFAULT_MAX_RETRIES", "MathPromptOptimizer"]
