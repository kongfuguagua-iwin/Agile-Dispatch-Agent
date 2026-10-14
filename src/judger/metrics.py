# -*- coding: utf-8 -*-
"""
Physical Metrics Calculation.

This module implements Section 3.3 of the paper:
"Hybrid Reward Evaluation"

Key equations:
- Equation 26: Hybrid physical reward (weighted sum)
- Equation 27: Redispatching economy component
- Equation 28: Safety bridge component
- Equation 29: Overflow penalty component
- Equation 30: Topological distance component

These components are combined with the LLM logical reward (Equation 11)
to form the final hybrid evaluation.
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional

from ..config import Config
from ..types import SolutionDict, EnvironmentStateDict


def compute_physical_reward(
    solution: SolutionDict,
    environment_state: EnvironmentStateDict,
    config: Config
) -> float:
    """
    Compute hybrid physical reward (Eq. 26)
    
    R = w1 * R_disp + w2 * R_bridge + w3 * R_overflow + w4 * R_distance
    
    Args:
        solution: Solution dictionary with decision variables
        environment_state: Environment state with grid information
        config: Configuration with weights
        
    Returns:
        Physical reward score [0, 1]
    """
    # Eq. 27: Redispatching economy
    r_disp = compute_redispatching_economy(solution, environment_state, config)
    
    # Eq. 28: Safety bridge
    r_bridge = compute_safety_bridge(environment_state, config)
    
    # Eq. 29: Overflow penalty
    r_overflow = compute_overflow_penalty(environment_state, config)
    
    # Eq. 30: Topological distance
    r_distance = compute_topological_distance(solution, environment_state, config)
    
    # Weighted sum (Eq. 26)
    reward = (
        config.W_DISP * r_disp +
        config.W_BRIDGE * r_bridge +
        config.W_OVERFLOW * r_overflow +
        config.W_DISTANCE * r_distance
    )
    
    return min(1.0, max(0.0, reward))


def compute_redispatching_economy(
    solution: SolutionDict,
    environment_state: EnvironmentStateDict,
    config: Config
) -> float:
    """
    Compute redispatching economy (Eq. 27)
    
    R_disp = 1 - (Σ C(P_g^t) - C(P_g^0)) / max(C_limit, ε)
    
    Args:
        solution: Solution with generator setpoints
        environment_state: Environment state
        config: Configuration
        
    Returns:
        Economy score [0, 1]
    """
    # Extract generator powers
    gen_powers = solution.get('gen_powers', [])
    initial_powers = environment_state.get('initial_gen_powers', [])
    gen_costs = environment_state.get('gen_costs', [1.0] * len(gen_powers))
    
    if not gen_powers or not initial_powers:
        return 0.5  # Default neutral score
    
    # Compute cost difference
    cost_after = sum(c * p for c, p in zip(gen_costs, gen_powers))
    cost_before = sum(c * p for c, p in zip(gen_costs, initial_powers))
    cost_diff = cost_after - cost_before
    
    # Normalize
    c_limit = max(abs(cost_diff), 1.0)  # Avoid division by zero
    r_disp = 1.0 - abs(cost_diff) / c_limit
    
    return min(1.0, max(0.0, r_disp))


def compute_safety_bridge(
    environment_state: EnvironmentStateDict,
    config: Config
) -> float:
    """
    Compute safety bridge (Eq. 28)
    
    R_bridge = min_{l in L} (1 - ρ_l)^p
    
    where ρ_l = I_l / I_max,l
    
    Args:
        environment_state: Environment state with line loadings
        config: Configuration with sensitivity factor p
        
    Returns:
        Safety bridge score [0, 1]
    """
    # Extract line loadings
    line_loadings = environment_state.get('line_loadings', [])
    line_max_capacities = environment_state.get('line_max_capacities', [])
    
    if not line_loadings or not line_max_capacities:
        return 0.5  # Default neutral score
    
    # Compute capacity usage ratios
    rho_values = []
    for loading, max_cap in zip(line_loadings, line_max_capacities):
        if max_cap > 0:
            rho = abs(loading) / max_cap
            rho_values.append(rho)
    
    if not rho_values:
        return 0.5
    
    # Find minimum safety margin
    p = config.P_BRIDGE
    safety_margins = [(1.0 - rho) ** p for rho in rho_values]
    r_bridge = min(safety_margins)
    
    return min(1.0, max(0.0, r_bridge))


def compute_overflow_penalty(
    environment_state: EnvironmentStateDict,
    config: Config
) -> float:
    """
    Compute overflow penalty (Eq. 29)
    
    R_overflow = (1/|L|) * Σ φ(ρ_l)
    
    where φ(ρ) = {
        1 if ρ < τ_warn
        (1-ρ)/(1-τ_warn) if τ_warn ≤ ρ ≤ 1
        -exp(ρ-1) if ρ > 1
    }
    
    Args:
        environment_state: Environment state
        config: Configuration with warning threshold
        
    Returns:
        Overflow penalty score [0, 1]
    """
    # Extract line loadings
    line_loadings = environment_state.get('line_loadings', [])
    line_max_capacities = environment_state.get('line_max_capacities', [])
    
    if not line_loadings or not line_max_capacities:
        return 0.5  # Default neutral score
    
    n_lines = len(line_loadings)
    if n_lines == 0:
        return 0.5
    
    tau_warn = config.TAU_WARN
    phi_values = []
    
    for loading, max_cap in zip(line_loadings, line_max_capacities):
        if max_cap > 0:
            rho = abs(loading) / max_cap
            
            # Compute φ(ρ)
            if rho < tau_warn:
                phi = 1.0
            elif rho <= 1.0:
                phi = (1.0 - rho) / (1.0 - tau_warn)
            else:
                phi = -math.exp(rho - 1.0)
            
            phi_values.append(phi)
    
    if not phi_values:
        return 0.5
    
    # Average penalty
    r_overflow = sum(phi_values) / len(phi_values)
    
    # Normalize to [0, 1]
    r_overflow = (r_overflow + 1.0) / 2.0
    
    return min(1.0, max(0.0, r_overflow))


def compute_topological_distance(
    solution: SolutionDict,
    environment_state: EnvironmentStateDict,
    config: Config
) -> float:
    """
    Compute topological distance penalty (Eq. 30)
    
    R_distance = exp(-λ * ||T_t - T_ref||_0)
    
    Args:
        solution: Solution with topology changes
        environment_state: Environment state with reference topology
        config: Configuration with lambda parameter
        
    Returns:
        Topological distance score [0, 1]
    """
    # Extract topologies
    current_topology = solution.get('topology', [])
    reference_topology = environment_state.get('reference_topology', [])
    
    if not current_topology or not reference_topology:
        return 0.5  # Default neutral score
    
    # Compute Hamming distance (L0 norm)
    if len(current_topology) != len(reference_topology):
        # Pad to same length
        max_len = max(len(current_topology), len(reference_topology))
        current_topology = list(current_topology) + [0] * (max_len - len(current_topology))
        reference_topology = list(reference_topology) + [0] * (max_len - len(reference_topology))
    
    hamming_distance = sum(
        1 for c, r in zip(current_topology, reference_topology) if c != r
    )
    
    # Compute reward (Eq. 30)
    lambda_val = config.LAMBDA_DISTANCE
    r_distance = math.exp(-lambda_val * hamming_distance)
    
    return min(1.0, max(0.0, r_distance))

