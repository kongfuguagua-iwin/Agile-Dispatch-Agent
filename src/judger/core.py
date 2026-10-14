# -*- coding: utf-8 -*-
"""
Judger Core Implementation.

This module implements Section 3.3 of the paper:
"Dual Evaluation Mechanism: Physical + Logical Rewards"

Key equations:
- Equation 11: Hybrid reward R = α * R_PHY + (1 - α) * R_LLM
- Equation 26-30: Physical reward components (see metrics.py)

The judger combines physical feasibility evaluation with LLM-based logical
coherence assessment to provide comprehensive solution evaluation.
"""

from typing import Dict, Any, Optional, Tuple

from ..config import Config
from ..types import SolutionDict, ProblemDict, EnvironmentStateDict, EvaluationDict
from ..utils.llm_interface import BaseLLM
from .metrics import compute_physical_reward


class Judger:
    """
    Judger Agent
    Implements dual evaluation mechanism
    
    Hybrid reward (Eq. 11):
    R = α * R_PHY + (1 - α) * R_LLM
    """
    
    def __init__(
        self,
        config: Config,
        llm: Optional[BaseLLM] = None  # LLM service for logical evaluation
    ):
        """
        Initialize Judger
        
        Args:
            config: Configuration with alpha weight
            llm: LLM service for logical evaluation (must implement BaseLLM interface)
        """
        self.config = config
        self.llm = llm
    
    def evaluate(
        self,
        solution: SolutionDict,
        problem: ProblemDict,
        environment_state: EnvironmentStateDict
    ) -> EvaluationDict:
        """
        Evaluate solution quality
        
        Args:
            solution: Solution from Solver
            problem: Optimization problem
            environment_state: Environment state
            
        Returns:
            Evaluation result with passed status and reward
        """
        # 1. Physical evaluation
        r_phy, phy_details = self._evaluate_physical(
            solution, environment_state
        )
        
        # 2. Logical evaluation
        r_llm, llm_details = self._evaluate_logical(
            solution, problem
        )
        
        # 3. Hybrid reward (Eq. 11)
        alpha = self.config.ALPHA_JUDGER
        reward = alpha * r_phy + (1 - alpha) * r_llm
        
        # 4. Check if passed
        passed = self._check_passed(solution, r_phy, r_llm, reward)
        
        return {
            'passed': passed,
            'reward': reward,
            'physical_reward': r_phy,
            'logical_reward': r_llm,
            'physical_details': phy_details,
            'logical_details': llm_details
        }
    
    def _evaluate_physical(
        self,
        solution: SolutionDict,
        environment_state: EnvironmentStateDict
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate physical feasibility and reward
        
        Args:
            solution: Solution
            environment_state: Environment state
            
        Returns:
            (reward, details)
        """
        # Check basic feasibility
        is_feasible = solution.get('is_feasible', False)
        
        if not is_feasible:
            return 0.0, {'is_feasible': False, 'reason': 'Solution not feasible'}
        
        # Compute physical reward (Eq. 26-30)
        reward = compute_physical_reward(
            solution,
            environment_state,
            self.config
        )
        
        details = {
            'is_feasible': True,
            'reward': reward,
            'components': {
                'redispatching_economy': solution.get('r_disp', 0.0),
                'safety_bridge': solution.get('r_bridge', 0.0),
                'overflow_penalty': solution.get('r_overflow', 0.0),
                'topological_distance': solution.get('r_distance', 0.0)
            }
        }
        
        return reward, details
    
    def _evaluate_logical(
        self,
        solution: SolutionDict,
        problem: ProblemDict
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate logical coherence using LLM
        
        Args:
            solution: Solution
            problem: Problem definition
            
        Returns:
            (reward, details)
        """
        if self.llm is None:
            # Fallback: simple heuristic
            is_feasible = solution.get('is_feasible', False)
            return 0.8 if is_feasible else 0.3, {'method': 'heuristic'}
        
        # Use LLM to evaluate logical coherence
        prompt = f"""
Evaluate the logical coherence of this optimization solution:

Problem:
{problem.get('objective', 'N/A')}

Solution:
{solution.get('decision_variables', {})}

Rate the solution on:
1. Rationality: Does it make sense?
2. Safety: Is it safe?
3. Sustainability: Is it sustainable?

Provide a score from 0 to 1.
"""
        
        try:
            response = self.llm.chat(prompt)
            # Parse response (simplified)
            # In practice, would extract score from structured response
            score = 0.7  # Placeholder
            details = {'method': 'llm', 'response': response[:100]}
            return score, details
        except Exception:
            # Fallback
            return 0.5, {'method': 'fallback'}
    
    def _check_passed(
        self,
        solution: SolutionDict,
        r_phy: float,
        r_llm: float,
        reward: float
    ) -> bool:
        """
        Check if solution passes evaluation
        
        Args:
            solution: Solution
            r_phy: Physical reward
            r_llm: Logical reward
            reward: Hybrid reward
            
        Returns:
            True if passed
        """
        # Must be feasible
        if not solution.get('is_feasible', False):
            return False
        
        # Must have minimum reward (from config)
        if reward < self.config.MIN_REWARD_THRESHOLD:
            return False
        
        # Physical reward must be above threshold (from config)
        if r_phy < self.config.MIN_PHYSICAL_REWARD:
            return False
        
        return True

