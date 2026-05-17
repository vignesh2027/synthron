"""Synthron agents package."""

from synthron.agents.base_agent import (
    AgentResult,
    BaseAgent,
    CriticScore,
    FinalResult,
    SubTask,
    SubTaskResult,
    TaskPlan,
    TaskStatus,
    Thought,
)
from synthron.agents.coder_agent import CoderAgent
from synthron.agents.coordinator_agent import CoordinatorAgent
from synthron.agents.critic_agent import CriticAgent
from synthron.agents.executor_agent import ExecutorAgent
from synthron.agents.memory_agent import MemoryAgent
from synthron.agents.planner_agent import PlannerAgent
from synthron.agents.researcher_agent import ResearcherAgent

__all__ = [
    "AgentResult",
    "BaseAgent",
    "CriticScore",
    "FinalResult",
    "SubTask",
    "SubTaskResult",
    "TaskPlan",
    "TaskStatus",
    "Thought",
    "CoderAgent",
    "CoordinatorAgent",
    "CriticAgent",
    "ExecutorAgent",
    "MemoryAgent",
    "PlannerAgent",
    "ResearcherAgent",
]
