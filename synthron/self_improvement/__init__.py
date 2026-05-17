"""Synthron self-improvement package — the secret weapon."""

from synthron.self_improvement.failure_analyzer import FailureAnalyzer
from synthron.self_improvement.pattern_learner import PatternLearner
from synthron.self_improvement.performance_tracker import PerformanceTracker
from synthron.self_improvement.prompt_optimizer import PromptOptimizer
from synthron.self_improvement.retry_strategist import RetryStrategist, RetryStrategy

__all__ = [
    "FailureAnalyzer",
    "PatternLearner",
    "PerformanceTracker",
    "PromptOptimizer",
    "RetryStrategist",
    "RetryStrategy",
]
