# -*- coding: utf-8 -*-
"""
Solver Core Implementation.

This module implements Section 3.2 of the paper:
"Adaptive Algorithm Selection via Problem-Algorithm Alignment"

Key equations:
- Equation 8: Alignment score G(A, φ) computation
- Equation 9: Feature weight computation (IDF-style)

The solver extracts problem features and matches them to algorithm capabilities
to select the most appropriate optimization algorithm.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ..config import Config
from .features import FeatureExtractor, ProblemFeatures


class Solver:
    """
    Solver Agent
    Implements adaptive algorithm selection via feature matching
    
    Algorithm:
    1. Extract problem features φ (Eq. 22-25)
    2. Compute alignment scores G(A, φ) for each algorithm (Eq. 8-9)
    3. Select best algorithm
    4. Execute and return solution
    """
    
    def __init__(
        self,
        config: Config,
        algorithms: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Solver
        
        Args:
            config: Configuration
            algorithms: Dictionary of available algorithms
        """
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.algorithms = algorithms or {}
    
    def solve(
        self,
        objective: str,
        constraints: List[str],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Solve optimization problem
        
        Args:
            objective: Objective function J(x)
            constraints: List of constraints Theta
            variables: Variable definitions
            
        Returns:
            Solution dictionary with decision variables and objective value
        """
        # 1. Extract features
        features = self.feature_extractor.extract(objective, constraints, variables)
        
        # 2. Compute alignment scores
        scores = self._compute_alignment_scores(features)
        
        # 3. Select best algorithm
        best_algorithm = self._select_algorithm(scores)
        
        # 4. Execute algorithm
        solution = self._execute_algorithm(
            best_algorithm,
            objective,
            constraints,
            variables
        )
        
        return {
            'algorithm': best_algorithm,
            'solution': solution,
            'features': features.to_dict(),
            'scores': scores
        }
    
    def _compute_alignment_scores(
        self,
        features: ProblemFeatures
    ) -> Dict[str, float]:
        """
        Compute alignment scores G(A, φ) for each algorithm (Eq. 8-9)
        
        G(A, φ) = Σ_i w(f_i) * R(ψ_j(A), φ_i)
        
        where w(f_i) is computed via IDF-style weighting (Eq. 9)
        
        Args:
            features: Problem features
            
        Returns:
            Dictionary mapping algorithm names to scores
        """
        scores = {}
        phi = features.to_vector()
        
        # Compute feature weights w(f_i) based on scarcity (Eq. 9)
        weights = self._compute_feature_weights(phi)
        
        for algo_name, algo in self.algorithms.items():
            # Get algorithm capability vector ψ(A)
            psi = self._get_algorithm_capabilities(algo)
            
            # Compute alignment score using paper formula
            score = self._alignment_score(phi, psi, weights)
            scores[algo_name] = score
        
        return scores
    
    def _compute_feature_weights(self, phi: List[float]) -> List[float]:
        """
        Compute feature weights w(f_i) (Eq. 9)
        
        IDF-style weighting based on feature scarcity:
        w(f_i) = ln(1 + N_total / N_i)
        
        where:
        - N_total: Total capability across all algorithms
        - N_i: Capability in dimension i across all algorithms
        
        Args:
            phi: Problem feature vector
            
        Returns:
            Feature weights vector
        """
        import math
        
        # Compute total capabilities across all algorithms
        total_capabilities = [0.0] * len(phi)
        total_sum = 0.0
        
        for algo in self.algorithms.values():
            psi = self._get_algorithm_capabilities(algo)
            for i in range(min(len(phi), len(psi))):
                total_capabilities[i] += psi[i]
                total_sum += psi[i]
        
        # Compute IDF-style weights
        weights = []
        epsilon = 1e-6
        
        for i in range(len(phi)):
            if total_capabilities[i] > epsilon:
                # IDF: w(f_i) = ln(1 + N_total / N_i)
                weight = math.log(1.0 + total_sum / (total_capabilities[i] + epsilon))
            else:
                weight = 1.0  # Default weight if no capability
            
            weights.append(weight)
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        
        return weights
    
    def _alignment_score(
        self,
        phi: List[float],
        psi: List[float],
        weights: List[float]
    ) -> float:
        """
        Compute alignment score G(A, φ) (Eq. 8-9)
        
        Implements the exact formula from the paper:
        G(A, φ) = Σ_i w(f_i) * R(ψ_j(A), φ_i)
        
        where:
        - w(f_i) is the feature weight based on scarcity (Eq. 9)
        - R(ψ_j(A), φ_i) is the regulation operator with saturation
        
        Args:
            phi: Problem feature vector φ
            psi: Algorithm capability vector ψ(A)
            weights: Feature weights w(f_i)
            
        Returns:
            Alignment score G(A, φ)
        """
        # Feature-to-capability mapping (from paper)
        # phi indices map to psi dimensions
        FEATURE_CAPABILITY_MAP = {
            0: 0,  # f_cvx (convexity) -> ψ_0 (convex handling)
            1: 1,  # f_stiff (stiffness) -> ψ_1 (constraint handling)
            2: 2,  # f_var (complexity) -> ψ_2 (speed)
            3: 3,  # f_ops (ops cost) -> ψ_3 (efficiency)
        }
        
        # BM25-style parameters for regulation operator
        K1 = 1.2  # Saturation parameter
        B = 0.75  # Normalization parameter
        
        score = 0.0
        
        for i, feature_value in enumerate(phi):
            # Map feature dimension to capability dimension
            j = FEATURE_CAPABILITY_MAP.get(i, i)
            if j >= len(psi):
                continue
            
            capability = psi[j]
            feature = feature_value
            
            # Regulation operator R(ψ_j(A), φ_i) with saturation (Eq. 8)
            # R = (ψ_j * (K1 + 1)) / (ψ_j + K1 * (1 - B + B * φ_i))
            numerator = capability * (K1 + 1.0)
            denominator = capability + K1 * (1.0 - B + B * feature)
            
            if denominator > 0:
                regulation = numerator / denominator
            else:
                regulation = 0.0
            
            # Weighted sum (Eq. 8)
            score += weights[i] * regulation
        
        return score
    
    def _get_algorithm_capabilities(self, algorithm: Any) -> List[float]:
        """
        Get algorithm capability vector ψ(A)
        
        Args:
            algorithm: Algorithm object
            
        Returns:
            Capability vector [4 dimensions matching features]
        """
        # Default capabilities if algorithm doesn't provide them
        if hasattr(algorithm, 'get_capability_vector'):
            return algorithm.get_capability_vector()
        
        # Default: assume algorithm has moderate capabilities
        return [0.5, 0.5, 0.5, 0.5]
    
    def _select_algorithm(self, scores: Dict[str, float]) -> str:
        """
        Select best algorithm based on scores
        
        Args:
            scores: Algorithm scores
            
        Returns:
            Best algorithm name
        """
        if not scores:
            raise ValueError("No algorithms available")
        
        return max(scores, key=scores.get)
    
    def _execute_algorithm(
        self,
        algorithm_name: str,
        objective: str,
        constraints: List[str],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute selected algorithm
        
        Args:
            algorithm_name: Name of algorithm to use
            objective: Objective function
            constraints: Constraints
            variables: Variable definitions
            
        Returns:
            Solution dictionary
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        algorithm = self.algorithms[algorithm_name]
        
        # Execute algorithm (simplified)
        if hasattr(algorithm, 'solve'):
            return algorithm.solve(objective, constraints, variables)
        
        # Fallback: return dummy solution
        return {
            'decision_variables': {},
            'objective_value': float('inf'),
            'is_feasible': False,
            'message': 'Algorithm execution not implemented'
        }
    
    def register_algorithm(self, name: str, algorithm: Any) -> None:
        """Register an algorithm"""
        self.algorithms[name] = algorithm

