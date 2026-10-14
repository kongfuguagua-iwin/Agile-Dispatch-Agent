# -*- coding: utf-8 -*-
"""
Experience Replay Buffer
Stores execution traces for TTSA updates
"""

from typing import List, Optional, Dict, Any
from collections import deque
import random


class Experience:
    """
    Single experience tuple
    """
    def __init__(
        self,
        observation: Any,
        augmented_obs: Any,
        mcts_info: Dict[str, Any],
        problem: Any,
        solution: Any,
        algorithm: str,
        reward: float,
        feedback: Any
    ):
        self.observation = observation
        self.augmented_obs = augmented_obs
        self.mcts_info = mcts_info
        self.problem = problem
        self.solution = solution
        self.algorithm = algorithm
        self.reward = reward
        self.feedback = feedback


class ReplayBuffer:
    """
    Experience replay buffer for storing execution traces
    
    Used by Summarizer for TTSA updates (Eq. 12-13)
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize replay buffer
        
        Args:
            max_size: Maximum number of experiences to store
        """
        self.buffer: deque = deque(maxlen=max_size)
        self.max_size = max_size
    
    def push(
        self,
        observation: Any,
        augmented_obs: Any,
        mcts_info: Dict[str, Any],
        problem: Any,
        solution: Any,
        algorithm: str,
        reward: float,
        feedback: Any
    ) -> None:
        """
        Add experience to buffer
        
        Args:
            observation: Raw observation
            augmented_obs: Augmented observation from Planner
            mcts_info: MCTS search information
            problem: Optimization problem
            solution: Solution from Solver
            algorithm: Algorithm used
            reward: Reward from Judger
            feedback: Feedback from Judger
        """
        experience = Experience(
            observation=observation,
            augmented_obs=augmented_obs,
            mcts_info=mcts_info,
            problem=problem,
            solution=solution,
            algorithm=algorithm,
            reward=reward,
            feedback=feedback
        )
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def get_high_reward_traces(self, threshold: float = 0.7) -> List[Experience]:
        """
        Get experiences with reward above threshold
        
        Used for slow timescale TK updates
        
        Args:
            threshold: Minimum reward threshold
            
        Returns:
            List of high-reward experiences
        """
        return [exp for exp in self.buffer if exp.reward >= threshold]
    
    def get_recent(self, n: int) -> List[Experience]:
        """
        Get most recent n experiences
        
        Args:
            n: Number of recent experiences
            
        Returns:
            List of recent experiences
        """
        return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) >= self.max_size

