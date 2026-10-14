# -*- coding: utf-8 -*-
"""
Mock L2RPN Environment
Simulates a simplified power grid for testing
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class GridState:
    """Power grid state"""
    # Generator states
    gen_powers: List[float] = field(default_factory=lambda: [50.0] * 10)
    gen_costs: List[float] = field(default_factory=lambda: [1.0 + i * 0.1 for i in range(10)])
    gen_pmin: List[float] = field(default_factory=lambda: [0.0] * 10)
    gen_pmax: List[float] = field(default_factory=lambda: [100.0] * 10)
    
    # Load states
    load_powers: List[float] = field(default_factory=lambda: [40.0 + i * 2.0 for i in range(10)])
    
    # Line states
    line_loadings: List[float] = field(default_factory=lambda: [0.7] * 20)
    line_max_capacities: List[float] = field(default_factory=lambda: [100.0] * 20)
    line_status: List[int] = field(default_factory=lambda: [1] * 20)
    
    # Topology
    topology: List[int] = field(default_factory=lambda: [1] * 20)
    reference_topology: List[int] = field(default_factory=lambda: [1] * 20)
    
    # Time
    timestep: int = 0


class MockL2RPN:
    """
    Mock L2RPN Environment
    
    Simulates a simplified power grid that changes dynamically
    when actions are applied.
    """
    
    def __init__(
        self,
        n_generators: int = 10,
        n_loads: int = 10,
        n_lines: int = 20,
        seed: int = 42
    ):
        """
        Initialize mock environment
        
        Args:
            n_generators: Number of generators
            n_loads: Number of loads
            n_lines: Number of transmission lines
            seed: Random seed
        """
        self.n_generators = n_generators
        self.n_loads = n_loads
        self.n_lines = n_lines
        self.rng = np.random.RandomState(seed)
        
        # Initialize state
        self.state = GridState(
            gen_powers=[50.0] * n_generators,
            gen_costs=[1.0 + i * 0.1 for i in range(n_generators)],
            gen_pmin=[0.0] * n_generators,
            gen_pmax=[100.0] * n_generators,
            load_powers=[40.0 + i * 2.0 for i in range(n_loads)],
            line_loadings=[0.7] * n_lines,
            line_max_capacities=[100.0] * n_lines,
            line_status=[1] * n_lines,
            topology=[1] * n_lines,
            reference_topology=[1] * n_lines
        )
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action and return new state
        
        Args:
            action: Action dictionary with generator powers, etc.
            
        Returns:
            New state dictionary
        """
        # Update generator powers if provided
        if 'gen_powers' in action:
            gen_powers = action['gen_powers']
            for i, power in enumerate(gen_powers):
                if i < len(self.state.gen_powers):
                    # Clamp to bounds
                    self.state.gen_powers[i] = np.clip(
                        power,
                        self.state.gen_pmin[i],
                        self.state.gen_pmax[i]
                    )
        
        # Update topology if provided
        if 'topology' in action:
            self.state.topology = action['topology']
        
        # Simulate load changes (random walk)
        for i in range(self.n_loads):
            change = self.rng.normal(0, 1.0)
            self.state.load_powers[i] = max(0, self.state.load_powers[i] + change)
        
        # Update line loadings based on power flow (simplified)
        total_gen = sum(self.state.gen_powers)
        total_load = sum(self.state.load_powers)
        imbalance = total_gen - total_load
        
        # Distribute imbalance across lines (simplified model)
        for i in range(self.n_lines):
            if self.state.line_status[i] == 1:
                # Line loading increases with imbalance
                base_loading = 0.7
                loading_change = imbalance / (self.n_lines * 10.0)
                self.state.line_loadings[i] = np.clip(
                    base_loading + loading_change,
                    0.0,
                    1.5  # Can exceed 1.0 (overload)
                )
        
        # Advance timestep
        self.state.timestep += 1
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state
        
        Returns:
            State dictionary compatible with agent interface
        """
        return {
            'initial_gen_powers': self.state.gen_powers.copy(),
            'gen_powers': self.state.gen_powers.copy(),
            'gen_costs': self.state.gen_costs.copy(),
            'load_powers': self.state.load_powers.copy(),
            'line_loadings': self.state.line_loadings.copy(),
            'line_max_capacities': self.state.line_max_capacities.copy(),
            'line_status': self.state.line_status.copy(),
            'topology': self.state.topology.copy(),
            'reference_topology': self.state.reference_topology.copy(),
            'timestep': self.state.timestep
        }
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state
        
        Returns:
            Initial state
        """
        self.state = GridState(
            gen_powers=[50.0] * self.n_generators,
            gen_costs=[1.0 + i * 0.1 for i in range(self.n_generators)],
            gen_pmin=[0.0] * self.n_generators,
            gen_pmax=[100.0] * self.n_generators,
            load_powers=[40.0 + i * 2.0 for i in range(self.n_loads)],
            line_loadings=[0.7] * self.n_lines,
            line_max_capacities=[100.0] * self.n_lines,
            line_status=[1] * self.n_lines,
            topology=[1] * self.n_lines,
            reference_topology=[1] * self.n_lines,
            timestep=0
        )
        return self.get_state()

