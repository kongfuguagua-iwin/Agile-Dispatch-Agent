# -*- coding: utf-8 -*-
"""
Knowledge Base Management
Manages Task Knowledge (TK) and Action Knowledge (AK) as per the paper
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TaskKnowledge:
    """
    Task Knowledge (TK) - Slow timescale knowledge
    Stores templates of feasible regions and objective functions
    """
    # Constraint templates (feasible region boundaries)
    tk_constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Objective function templates
    tk_objectives: List[Dict[str, Any]] = field(default_factory=list)
    
    # Manifold boundaries (for dynamic matching)
    manifold_boundaries: Dict[str, Any] = field(default_factory=dict)
    
    def add_constraint_template(self, template: Dict[str, Any]) -> None:
        """Add a constraint template"""
        self.tk_constraints.append(template)
    
    def add_objective_template(self, template: Dict[str, Any]) -> None:
        """Add an objective template"""
        self.tk_objectives.append(template)
    
    def update_manifold(self, boundaries: Dict[str, Any]) -> None:
        """Update manifold boundaries (slow timescale)"""
        self.manifold_boundaries.update(boundaries)
    
    def get_relevant_templates(self, context: str) -> List[Dict[str, Any]]:
        """Retrieve relevant templates based on context"""
        # Simple keyword matching (can be enhanced with embeddings)
        relevant = []
        context_lower = context.lower()
        
        for template in self.tk_constraints + self.tk_objectives:
            if any(keyword in context_lower for keyword in template.get('keywords', [])):
                relevant.append(template)
        
        return relevant


@dataclass
class ActionKnowledge:
    """
    Action Knowledge (AK) - Fast timescale knowledge
    Stores transition probabilities and tool selection priors
    """
    # Transition probability matrix P(s'|s, a)
    # Format: {(state, action): {next_state: probability}}
    ak_matrix: Dict[tuple, Dict[str, float]] = field(default_factory=dict)
    
    # Tool selection priors (for MCTS)
    tool_priors: Dict[str, float] = field(default_factory=dict)
    
    # Success statistics
    tool_success_count: Dict[str, int] = field(default_factory=dict)
    tool_total_count: Dict[str, int] = field(default_factory=dict)
    
    def update_transition(
        self, 
        state: str, 
        action: str, 
        next_state: str, 
        probability: float
    ) -> None:
        """Update transition probability (fast timescale)"""
        key = (state, action)
        if key not in self.ak_matrix:
            self.ak_matrix[key] = {}
        self.ak_matrix[key][next_state] = probability
    
    def get_transition_prob(
        self, 
        state: str, 
        action: str, 
        next_state: str
    ) -> float:
        """Get transition probability"""
        key = (state, action)
        if key not in self.ak_matrix:
            return 0.0
        return self.ak_matrix[key].get(next_state, 0.0)
    
    def update_tool_prior(self, tool_name: str, success: bool) -> None:
        """Update tool selection prior based on success"""
        if tool_name not in self.tool_total_count:
            self.tool_total_count[tool_name] = 0
            self.tool_success_count[tool_name] = 0
        
        self.tool_total_count[tool_name] += 1
        if success:
            self.tool_success_count[tool_name] += 1
        
        # Update prior as success rate
        total = self.tool_total_count[tool_name]
        success_count = self.tool_success_count[tool_name]
        self.tool_priors[tool_name] = success_count / total if total > 0 else 0.5
    
    def get_tool_prior(self, tool_name: str) -> float:
        """Get tool selection prior"""
        return self.tool_priors.get(tool_name, 0.5)  # Default uniform prior


class KnowledgeBase:
    """
    Unified knowledge base managing both TK and AK
    """
    
    def __init__(self):
        self.tk = TaskKnowledge()
        self.ak = ActionKnowledge()
    
    def get_tk(self) -> TaskKnowledge:
        """Get Task Knowledge"""
        return self.tk
    
    def get_ak(self) -> ActionKnowledge:
        """Get Action Knowledge"""
        return self.ak
    
    def update_tk(self, update_data: Dict[str, Any]) -> None:
        """Update Task Knowledge (slow timescale)"""
        if 'constraints' in update_data:
            for constraint in update_data['constraints']:
                self.tk.add_constraint_template(constraint)
        
        if 'objectives' in update_data:
            for objective in update_data['objectives']:
                self.tk.add_objective_template(objective)
        
        if 'manifold' in update_data:
            self.tk.update_manifold(update_data['manifold'])
    
    def update_ak(self, update_data: Dict[str, Any]) -> None:
        """Update Action Knowledge (fast timescale)"""
        if 'transitions' in update_data:
            for trans in update_data['transitions']:
                self.ak.update_transition(
                    trans['state'],
                    trans['action'],
                    trans['next_state'],
                    trans['probability']
                )
        
        if 'tool_feedback' in update_data:
            for feedback in update_data['tool_feedback']:
                self.ak.update_tool_prior(
                    feedback['tool_name'],
                    feedback['success']
                )

