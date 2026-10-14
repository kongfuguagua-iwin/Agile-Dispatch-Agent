# -*- coding: utf-8 -*-
"""
Planner Core Implementation.

This module implements Section 3.1 of the paper:
"Active State Augmentation via MCTS Belief Tree Search"

Key equations:
- Equation 2: UCB exploration formula for tool selection
- Equation 6: Value backpropagation in MCTS tree
- Equation 3-5: State-to-problem mapping (formalization)

The planner uses MCTS to explore the state space and resolve attribution
ambiguity by actively augmenting observations with relevant information.
"""

import math
import random
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from ..config import Config
from ..knowledge import ActionKnowledge
from ..utils.llm_interface import BaseLLM


@dataclass
class MCTSNode:
    """
    MCTS search tree node
    """
    state: str  # State description
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action: str = ""  # Tool action that led to this state
    
    def ucb_score(self, exploration_constant: float) -> float:
        """
        Calculate UCB score (Eq. 2)
        
        UCB(s, a) = Q(s, a) + C * sqrt(ln(N(s)) / N(s, a))
        
        Args:
            exploration_constant: C parameter
            
        Returns:
            UCB score
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits  # Q(s, a)
        
        if self.parent is None:
            exploration = 0.0
        else:
            # Exploration term: C * sqrt(ln(N(s)) / N(s, a))
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
        
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float) -> 'MCTSNode':
        """Select best child based on UCB"""
        if not self.children:
            return self
        return max(self.children, key=lambda c: c.ucb_score(exploration_constant))
    
    def is_leaf(self) -> bool:
        """Check if node is leaf"""
        return len(self.children) == 0


class Planner:
    """
    Planner Agent
    Implements Active State Augmentation via MCTS
    
    Core algorithm:
    1. MCTS search to explore state space
    2. Tool selection based on UCB and AK priors
    3. State augmentation until sufficient information
    4. Problem formulation using TK templates
    """
    
    def __init__(
        self,
        config: Config,
        tools: Any = None,  # Tool registry
        llm: Optional[BaseLLM] = None,  # LLM service (must implement BaseLLM interface)
        ak: Optional[ActionKnowledge] = None
    ):
        """
        Initialize Planner
        
        Args:
            config: Configuration
            tools: Tool registry for state augmentation
            llm: LLM service for problem formulation (must implement BaseLLM interface)
            ak: Action Knowledge for tool priors
        """
        self.config = config
        self.tools = tools
        self.llm = llm
        self.ak = ak
        
        # MCTS state
        self._mcts_root: Optional[MCTSNode] = None
    
    def plan(
        self,
        observation: Any,
        ak_prior: Optional[ActionKnowledge] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Main planning method
        
        Args:
            observation: Raw environment observation
            ak_prior: Action Knowledge for tool selection priors
            
        Returns:
            (problem_tuple, mcts_info)
            problem_tuple: (objective, constraints) as (J, Theta)
            mcts_info: MCTS search information for Summarizer
        """
        # Use provided AK or instance AK
        ak = ak_prior or self.ak
        
        # Initialize root node
        root_state = self._observation_to_state(observation)
        self._mcts_root = MCTSNode(state=root_state)
        
        # MCTS search loop
        for iteration in range(self.config.MCTS_MAX_ITERATIONS):
            # Selection: Traverse to leaf using UCB
            leaf_node = self._select(self._mcts_root, ak)
            
            # Expansion: Add children if not terminal
            if leaf_node.visits > 0 and not self._is_terminal(leaf_node.state):
                self._expand(leaf_node, ak)
            
            # Simulation: Rollout to estimate value
            value = self._simulate(leaf_node, ak)
            
            # Backpropagation: Update values (Eq. 6)
            self._backpropagate(leaf_node, value)
        
        # Generate problem formulation from best path
        augmented_state = self._get_best_path_state()
        problem_tuple = self._map_to_formal(augmented_state)
        
        # Collect MCTS info for Summarizer
        mcts_info = {
            'root_visits': self._mcts_root.visits,
            'best_path': self._get_best_path(),
            'tree_depth': self._get_tree_depth(),
            'total_iterations': self.config.MCTS_MAX_ITERATIONS
        }
        
        return problem_tuple, mcts_info
    
    def _select(self, root: MCTSNode, ak: Optional[ActionKnowledge]) -> MCTSNode:
        """
        Selection phase: Traverse to leaf using UCB (Eq. 2)
        
        Args:
            root: Root node
            ak: Action Knowledge for priors
            
        Returns:
            Selected leaf node
        """
        node = root
        
        while not node.is_leaf():
            # Select child with highest UCB score
            # Incorporate AK priors if available
            best_child = None
            best_score = float('-inf')
            
            for child in node.children:
                ucb = child.ucb_score(self.config.MCTS_EXPLORATION_C)
                
                # Incorporate AK prior if available
                if ak is not None:
                    prior = ak.get_tool_prior(child.action)
                    ucb = 0.7 * ucb + 0.3 * prior
                
                if ucb > best_score:
                    best_score = ucb
                    best_child = child
            
            if best_child is None:
                break
            node = best_child
        
        return node
    
    def _expand(self, node: MCTSNode, ak: Optional[ActionKnowledge]) -> None:
        """
        Expansion phase: Add child nodes for available actions
        
        Args:
            node: Node to expand
            ak: Action Knowledge for tool priors
        """
        if self.tools is None:
            return
        
        # Get available tools/actions
        available_tools = self._get_available_tools(node.state)
        
        for tool_name in available_tools:
            # Create child node
            child_state = f"{node.state} -> {tool_name}"
            child = MCTSNode(
                state=child_state,
                parent=node,
                action=tool_name
            )
            node.children.append(child)
    
    def _simulate(self, node: MCTSNode, ak: Optional[ActionKnowledge]) -> float:
        """
        Simulation phase: Rollout to estimate value (Eq. 6)
        
        Implements Belief Tree Search: evaluates state clarity to resolve
        attribution ambiguity
        
        NOTE: This is a MOCK IMPLEMENTATION for demo purposes.
        In production, this should use a trained value network or actual
        problem evaluation to estimate state value. The current implementation
        uses heuristics and random sampling.
        
        Args:
            node: Node to simulate from
            ak: Action Knowledge
            
        Returns:
            Estimated value [0, 1]
        """
        state = node.state
        
        # Evaluate state clarity (information completeness)
        clarity_score = self._evaluate_state_clarity(state)
        
        # Count number of tool calls
        tool_count = state.count('->')
        
        # Check if state has sufficient information
        has_sufficient_info = self._has_sufficient_info(state)
        
        # Value based on information completeness and clarity
        # MOCK: Uses random component for demonstration
        # In production, replace with actual value network evaluation
        if has_sufficient_info:
            # High value if sufficient info and high clarity
            # Random component simulates uncertainty in value estimation
            value = 0.7 + 0.2 * clarity_score + 0.1 * random.random()
        else:
            # Progressive value based on clarity and tool usage
            progress = tool_count / self.config.PLANNER_MAX_AUGMENTATION_STEPS
            value = 0.2 + 0.3 * clarity_score + 0.3 * progress
        
        return min(1.0, max(0.0, value))
    
    def _evaluate_state_clarity(self, state: str) -> float:
        """
        Evaluate state clarity based on key information presence
        
        Implements belief evaluation: higher clarity = less attribution ambiguity
        
        Args:
            state: State string
            
        Returns:
            Clarity score [0, 1]
        """
        state_lower = state.lower()
        clarity = 0.0
        
        # Key information indicators
        key_indicators = {
            'load': 0.2,
            'generation': 0.2,
            'constraint': 0.15,
            'objective': 0.15,
            'variable': 0.1,
            'bound': 0.1,
            'cost': 0.1
        }
        
        # Check for key indicators
        for indicator, weight in key_indicators.items():
            if indicator in state_lower:
                clarity += weight
        
        # Bonus for having multiple indicators
        indicator_count = sum(1 for ind in key_indicators if ind in state_lower)
        if indicator_count >= 3:
            clarity += 0.1
        
        return min(1.0, clarity)
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagation phase: Update node values (Eq. 6)
        
        Q(s, a) = (1/N) * Σ r_i
        
        Args:
            node: Node to update
            value: Simulated value
        """
        current = node
        while current is not None:
            current.visits += 1
            # Update value: running average
            current.value = current.value + (value - current.value) / current.visits
            current = current.parent
    
    def _map_to_formal(
        self,
        augmented_state: str
    ) -> Tuple[str, List[str]]:
        """
        Map augmented state to formal optimization problem (Eq. 3-5)
        
        Uses LLM to generate mathematical formulation
        
        Args:
            augmented_state: Augmented state string
            
        Returns:
            (objective_function, constraints_list)
        """
        if self.llm is None:
            # Fallback: simple template
            return ("min x", ["x >= 0", "x <= 100"])
        
        # Build prompt for LLM
        prompt = f"""
Convert the following augmented state into a formal optimization problem.

State:
{augmented_state}

Please provide:
1. Objective function (minimize or maximize)
2. List of constraints

Format as JSON:
{{
    "objective": "min f(x)",
    "constraints": ["g1(x) <= 0", "g2(x) >= 0"]
}}
"""
        
        try:
            response = self.llm.chat(prompt)
            # Parse response (simplified)
            # In practice, use proper JSON parsing
            objective = "min x"  # Placeholder
            constraints = ["x >= 0"]  # Placeholder
            
            return (objective, constraints)
        except Exception:
            # Fallback
            return ("min x", ["x >= 0"])
    
    def _observation_to_state(self, observation: Any) -> str:
        """Convert observation to state string"""
        if hasattr(observation, 'to_prompt_string'):
            return observation.to_prompt_string()
        return str(observation)
    
    def _is_terminal(self, state: str) -> bool:
        """Check if state is terminal"""
        # Terminal if we have sufficient information or max steps reached
        tool_count = state.count('->')
        return (
            tool_count >= self.config.PLANNER_MAX_AUGMENTATION_STEPS or
            self._has_sufficient_info(state)
        )
    
    def _has_sufficient_info(self, state: str) -> bool:
        """Check if state has sufficient information for formulation"""
        # Simple heuristic: check for key information
        keywords = ['load', 'generation', 'constraint', 'objective']
        return any(keyword in state.lower() for keyword in keywords)
    
    def _get_available_tools(self, state: str) -> List[str]:
        """Get available tools for current state"""
        if self.tools is None:
            return []
        
        # Return list of tool names
        if hasattr(self.tools, 'list_tools'):
            return self.tools.list_tools()
        return []
    
    def _get_best_path_state(self) -> str:
        """Get augmented state from best MCTS path"""
        if self._mcts_root is None:
            return ""
        
        path = self._get_best_path()
        return " -> ".join(path) if path else ""
    
    def _get_best_path(self) -> List[str]:
        """Get best path from MCTS tree"""
        if self._mcts_root is None or not self._mcts_root.children:
            return []
        
        path = []
        node = self._mcts_root
        
        while node.children:
            # Select child with highest value
            best_child = max(node.children, key=lambda c: c.value / max(c.visits, 1))
            path.append(best_child.action)
            node = best_child
        
        return path
    
    def _get_tree_depth(self) -> int:
        """Get maximum depth of MCTS tree"""
        if self._mcts_root is None:
            return 0
        
        def depth(node: MCTSNode) -> int:
            if node.is_leaf():
                return 1
            if not node.children:
                return 1
            return 1 + max(depth(child) for child in node.children)
        
        return depth(self._mcts_root)

