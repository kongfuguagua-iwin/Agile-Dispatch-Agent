# -*- coding: utf-8 -*-
"""
Configuration module for ADA framework.

This module stores all hyperparameters and constants referenced in the paper:
"ADA: Resolving Attribution Ambiguity in End-to-End Power System Dispatch 
via Two-Time-Scale Stochastic Approximation"

All parameters are organized by component and reference the corresponding
equations or sections from the paper.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """
    Configuration class for ADA framework
    
    All hyperparameters are based on the paper specifications
    """
    
    # ============= Judger Configuration =============
    # Eq. 11: Hybrid reward weight
    # R = alpha * R_PHY + (1 - alpha) * R_LLM
    ALPHA_JUDGER: float = 0.7  # Physical reward weight (Table 3 suggests [0.5, 0.8])
    
    # Reward component weights (Eq. 26)
    W_DISP: float = 0.25      # Redispatching economy weight
    W_BRIDGE: float = 0.30   # Safety bridge weight
    W_OVERFLOW: float = 0.25 # Overflow penalty weight
    W_DISTANCE: float = 0.20 # Topological distance weight
    
    # Safety parameters
    TAU_WARN: float = 0.9    # Warning threshold for line loading
    P_BRIDGE: float = 2.0    # Safety bridge sensitivity factor (Eq. 28)
    LAMBDA_DISTANCE: float = 1.0  # Topological distance penalty factor (Eq. 30)
    
    # ============= MCTS Configuration =============
    # UCB exploration constant (Eq. 2)
    MCTS_EXPLORATION_C: float = 1.414  # sqrt(2) for balanced exploration/exploitation
    
    # MCTS search parameters
    MCTS_MAX_ITERATIONS: int = 100
    MCTS_MAX_DEPTH: int = 10
    MCTS_SIMULATION_ROLLOUTS: int = 10
    
    # ============= Update Frequencies (TTSA) =============
    # Fast timescale: Action Knowledge (AK) update frequency
    UPDATE_FREQ_AK: int = 1  # Update every step
    
    # Slow timescale: Task Knowledge (TK) update frequency
    UPDATE_FREQ_TK: int = 10  # Update every N steps
    
    # ============= Retry Configuration =============
    MAX_RETRIES: int = 3  # Maximum retries per step (Section 4.1)
    
    # ============= Feature Extraction Parameters =============
    # Eq. 22: Local convexity sampling
    N_SAMPLES_CONVEXITY: int = 100  # Number of point pairs for Jensen's inequality test
    
    # Eq. 23: Constraint stiffness sampling
    N_SAMPLES_STIFFNESS: int = 1000  # Hit-and-Run sampling count
    
    # Eq. 24: Complexity quantification
    BETA_VAR: float = 1.0  # Weight factor for integer programming complexity
    
    # Eq. 25: Objective computational complexity
    T_REF: float = 1.0  # Reference time for FLOPs normalization
    W_ADD: float = 1.0   # Weight for addition operation
    W_EXP: float = 20.0  # Weight for exponential operation
    
    # ============= Solver Configuration =============
    SOLVER_TIMEOUT: float = 60.0  # Timeout in seconds
    
    # ============= Knowledge Base Configuration =============
    KB_TOP_K: int = 3  # Number of retrieved knowledge items
    
    # ============= Planner Configuration =============
    PLANNER_MAX_AUGMENTATION_STEPS: int = 5  # Maximum state augmentation steps
    
    # ============= Summarizer Configuration (TTSA) =============
    # Fast timescale learning rate (Eq. 12)
    ALPHA_AK: float = 0.1  # Action Knowledge update learning rate
    
    # Slow timescale learning rate (Eq. 13)
    ALPHA_TK: float = 0.05  # Task Knowledge update learning rate
    
    # Success threshold for AK updates
    SUCCESS_THRESHOLD: float = 0.6  # Reward threshold for considering solution successful
    
    # Transition probability update learning rate
    ALPHA_TRANSITION: float = 0.1  # Transition probability update rate
    
    # ============= Judger Configuration =============
    # Evaluation thresholds
    MIN_REWARD_THRESHOLD: float = 0.6  # Minimum hybrid reward to pass
    MIN_PHYSICAL_REWARD: float = 0.5  # Minimum physical reward to pass
    
    # High reward threshold for TK updates
    HIGH_REWARD_THRESHOLD: float = 0.7  # Threshold for high-reward trajectories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'ALPHA_JUDGER': self.ALPHA_JUDGER,
            'MCTS_EXPLORATION_C': self.MCTS_EXPLORATION_C,
            'UPDATE_FREQ_AK': self.UPDATE_FREQ_AK,
            'UPDATE_FREQ_TK': self.UPDATE_FREQ_TK,
            'MAX_RETRIES': self.MAX_RETRIES,
        }

