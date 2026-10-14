# -*- coding: utf-8 -*-
"""
Type definitions for ADA framework.

This module provides TypedDict and type aliases for better type safety
and code documentation.
"""

from typing import TypedDict, List, Dict, Any, Optional


class SolutionDict(TypedDict, total=False):
    """Solution dictionary structure returned by Solver."""
    decision_variables: Dict[str, Any]
    objective_value: float
    is_feasible: bool
    message: str
    gen_powers: List[float]
    topology: List[int]


class ProblemDict(TypedDict, total=False):
    """Problem dictionary structure."""
    objective: str
    constraints: List[str]
    variables: Dict[str, Any]


class EnvironmentStateDict(TypedDict, total=False):
    """Environment state dictionary structure."""
    initial_gen_powers: List[float]
    gen_powers: List[float]
    gen_costs: List[float]
    line_loadings: List[float]
    line_max_capacities: List[float]
    reference_topology: List[int]


class MCTSInfoDict(TypedDict, total=False):
    """MCTS search information dictionary."""
    root_visits: int
    best_path: List[str]
    tree_depth: int
    total_iterations: int


class EvaluationDict(TypedDict, total=False):
    """Evaluation result dictionary from Judger."""
    passed: bool
    reward: float
    physical_reward: float
    logical_reward: float
    physical_details: Dict[str, Any]
    logical_details: Dict[str, Any]


class ExperienceDict(TypedDict, total=False):
    """Experience dictionary for Summarizer updates."""
    observation: Any
    augmented_obs: Any
    mcts_info: MCTSInfoDict
    problem: ProblemDict
    solution: SolutionDict
    algorithm: str
    reward: float
    feedback: EvaluationDict

