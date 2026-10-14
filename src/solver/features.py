# -*- coding: utf-8 -*-
"""
Feature Extraction Module.

This module implements Section 3.2 of the paper:
"Problem Feature Extraction for Algorithm Selection"

Key equations:
- Equation 22: Local convexity feature f_cvx
- Equation 23: Constraint stiffness feature f_stiff
- Equation 24: Solving complexity feature f_var
- Equation 25: Objective computational complexity feature f_ops

These features form the problem feature vector φ used for algorithm matching.
"""

import numpy as np
import math
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from ..config import Config


@dataclass
class ProblemFeatures:
    """
    Problem feature vector φ
    All features normalized to [0, 1]
    """
    # Eq. 22: Local convexity
    f_cvx: float = 0.0
    
    # Eq. 23: Constraint stiffness
    f_stiff: float = 0.0
    
    # Eq. 24: Solving complexity
    f_var: float = 0.0
    
    # Eq. 25: Objective computational complexity
    f_ops: float = 0.0
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector"""
        return [self.f_cvx, self.f_stiff, self.f_var, self.f_ops]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'f_cvx': self.f_cvx,
            'f_stiff': self.f_stiff,
            'f_var': self.f_var,
            'f_ops': self.f_ops
        }


class FeatureExtractor:
    """
    Feature Extractor
    Implements feature extraction formulas from the paper
    """
    
    def __init__(self, config: Config):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration with sampling parameters
        """
        self.config = config
    
    def extract(
        self,
        objective: str,
        constraints: List[str],
        variables: Dict[str, Any]
    ) -> ProblemFeatures:
        """
        Extract problem features
        
        Args:
            objective: Objective function J(x)
            constraints: List of constraints Theta
            variables: Variable definitions with bounds
            
        Returns:
            ProblemFeatures object
        """
        features = ProblemFeatures()
        
        # Extract variable bounds
        lb = variables.get('lower_bounds', [])
        ub = variables.get('upper_bounds', [])
        var_types = variables.get('types', [])
        n_cont = variables.get('n_continuous', 0)
        n_int = variables.get('n_integer', 0)
        
        # Eq. 22: Local convexity f_cvx
        features.f_cvx = self._compute_convexity(
            objective, constraints, lb, ub
        )
        
        # Eq. 23: Constraint stiffness f_stiff
        features.f_stiff = self._compute_stiffness(
            constraints, lb, ub
        )
        
        # Eq. 24: Solving complexity f_var
        features.f_var = self._compute_complexity(n_cont, n_int)
        
        # Eq. 25: Objective computational complexity f_ops
        features.f_ops = self._compute_ops_complexity(objective)
        
        return features
    
    def _compute_convexity(
        self,
        objective: str,
        constraints: List[str],
        lb: List[float],
        ub: List[float]
    ) -> float:
        """
        Compute local convexity (Eq. 22)
        
        f_cvx = (1/N_s) * Σ I[J(λx1 + (1-λ)x2) > λJ(x1) + (1-λ)J(x2)]
        
        Returns:
            Convexity score [0, 1], closer to 0 = more convex
        """
        # Simplified implementation: sample points and check Jensen's inequality
        n_samples = self.config.N_SAMPLES_CONVEXITY
        
        # For simplicity, assume we can evaluate the objective
        # In practice, this would require a symbolic evaluator
        violations = 0
        
        # Sample pairs of points
        for _ in range(n_samples):
            # Sample two points within bounds
            x1 = self._sample_point(lb, ub)
            x2 = self._sample_point(lb, ub)
            lambda_val = np.random.uniform(0, 1)
            
            # Check Jensen's inequality
            # J(λx1 + (1-λ)x2) > λJ(x1) + (1-λ)J(x2) indicates non-convexity
            # Simplified: use heuristic based on objective structure
            if self._check_jensen_violation(objective, x1, x2, lambda_val):
                violations += 1
        
        # f_cvx is the fraction of violations
        f_cvx = violations / n_samples if n_samples > 0 else 0.0
        return min(1.0, f_cvx)
    
    def _compute_stiffness(
        self,
        constraints: List[str],
        lb: List[float],
        ub: List[float]
    ) -> float:
        """
        Compute constraint stiffness (Eq. 23)
        
        f_stiff = 1 - (Count(x_j ∈ Ω_Θ) / N_sample)
        
        Returns:
            Stiffness score [0, 1], higher = more constrained
        """
        n_samples = self.config.N_SAMPLES_STIFFNESS
        
        feasible_count = 0
        
        # Hit-and-Run sampling
        for _ in range(n_samples):
            # Sample random point
            x = self._sample_point(lb, ub)
            
            # Check if point satisfies constraints
            if self._check_feasibility(x, constraints):
                feasible_count += 1
        
        # f_stiff = 1 - feasibility_ratio
        feasibility_ratio = feasible_count / n_samples if n_samples > 0 else 0.0
        f_stiff = 1.0 - feasibility_ratio
        
        return min(1.0, max(0.0, f_stiff))
    
    def _compute_complexity(self, n_cont: int, n_int: int) -> float:
        """
        Compute solving complexity (Eq. 24)
        
        f_var = log(1 + N_cont + β * N_int * 2^N_int)
        
        Args:
            n_cont: Number of continuous variables
            n_int: Number of integer variables
            
        Returns:
            Normalized complexity score [0, 1]
        """
        beta = self.config.BETA_VAR
        
        # Compute raw complexity
        complexity = math.log(1 + n_cont + beta * n_int * (2 ** n_int))
        
        # Normalize (assume max complexity corresponds to ~100 vars)
        max_complexity = math.log(1 + 100 + beta * 50 * (2 ** 50))
        f_var = complexity / max_complexity if max_complexity > 0 else 0.0
        
        return min(1.0, f_var)
    
    def _compute_ops_complexity(self, objective: str) -> float:
        """
        Compute objective computational complexity (Eq. 25)
        
        f_ops = (Σ w_k * N_op,k) / T_ref
        
        Args:
            objective: Objective function string
            
        Returns:
            Normalized ops complexity [0, 1]
        """
        # Count operations in objective
        total_flops = 0.0
        
        # Count additions
        add_count = objective.count('+') + objective.count('-')
        total_flops += self.config.W_ADD * add_count
        
        # Count multiplications
        mult_count = objective.count('*')
        total_flops += self.config.W_ADD * mult_count  # Similar cost to add
        
        # Count exponentials
        exp_count = objective.count('**') + objective.count('exp') + objective.count('^')
        total_flops += self.config.W_EXP * exp_count
        
        # Count trigonometric functions
        trig_count = (
            objective.count('sin') + objective.count('cos') +
            objective.count('tan') + objective.count('log')
        )
        total_flops += self.config.W_EXP * 0.5 * trig_count  # Trig functions are expensive
        
        # Normalize by reference time
        f_ops = total_flops / self.config.T_REF
        
        # Normalize to [0, 1] (assume max ~1000 FLOPs)
        f_ops = min(1.0, f_ops / 1000.0)
        
        return f_ops
    
    def _sample_point(
        self,
        lb: List[float],
        ub: List[float]
    ) -> np.ndarray:
        """Sample a random point within bounds"""
        if not lb or not ub:
            return np.array([0.0])
        
        n = len(lb)
        point = np.zeros(n)
        
        for i in range(n):
            low = lb[i] if lb[i] != float('-inf') else -100.0
            high = ub[i] if ub[i] != float('inf') else 100.0
            point[i] = np.random.uniform(low, high)
        
        return point
    
    def _check_jensen_violation(
        self,
        objective: str,
        x1: np.ndarray,
        x2: np.ndarray,
        lambda_val: float
    ) -> bool:
        """
        Check if Jensen's inequality is violated (Eq. 22)
        
        MOCK IMPLEMENTATION FOR DEMO: This function uses heuristics and
        random sampling instead of actual function evaluation.
        
        For reproduction, replace with actual symbolic or numerical
        evaluation of the objective function J(x) to check:
        J(λx1 + (1-λ)x2) > λJ(x1) + (1-λ)J(x2)
        
        This requires either:
        1. A symbolic math evaluator (e.g., sympy)
        2. A numerical function evaluator with the actual problem formulation
        
        Args:
            objective: Objective function string
            x1: First sample point
            x2: Second sample point
            lambda_val: Convex combination parameter
            
        Returns:
            True if Jensen's inequality is violated (non-convex)
        """
        # MOCK: Heuristic-based detection
        # In production, this should evaluate the actual function
        non_convex_patterns = ['sin', 'cos', 'exp', '**', '^']
        has_non_convex = any(pattern in objective.lower() for pattern in non_convex_patterns)
        
        if has_non_convex:
            # MOCK: Randomly decide violation
            # Replace with actual function evaluation: 
            #   J_mid = J(λ*x1 + (1-λ)*x2)
            #   J_convex = λ*J(x1) + (1-λ)*J(x2)
            #   return J_mid > J_convex
            return np.random.random() > 0.5
        
        return False
    
    def _check_feasibility(
        self,
        x: np.ndarray,
        constraints: List[str]
    ) -> bool:
        """
        Check if point satisfies constraints (Eq. 23)
        
        MOCK IMPLEMENTATION FOR DEMO: This function uses random sampling
        instead of actual constraint evaluation.
        
        For reproduction, replace with actual constraint checking:
        1. Parse constraint strings into evaluable functions
        2. Evaluate each constraint: g_i(x) <= 0 or g_i(x) >= 0
        3. Return True if all constraints are satisfied
        
        This requires either:
        1. A symbolic constraint parser (e.g., using sympy)
        2. Integration with an actual optimization solver (e.g., Gurobi, CVXPY)
        
        Args:
            x: Point to check
            constraints: List of constraint strings
            
        Returns:
            True if point satisfies all constraints
        """
        # MOCK: Random feasibility check
        # Replace with: all(g_i(x) <= 0 for g_i in constraints)
        return np.random.random() > 0.3  # 70% feasibility rate (demo only)

