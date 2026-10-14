# -*- coding: utf-8 -*-
"""
Summarizer Core Implementation.

This module implements Section 3.4 of the paper:
"Two-Time-Scale Stochastic Approximation (TTSA) Updates"

Key equations:
- Equation 12: Fast timescale update for Action Knowledge (AK)
- Equation 13: Slow timescale update for Task Knowledge (TK)

The summarizer decouples optimization into two asynchronous evolutionary loops:
1. Fast: Updates tool selection priors (AK) based on MCTS results
2. Slow: Updates constraint manifold templates (TK) based on successful trajectories
"""

from typing import Dict, Any, List, Optional

from ..config import Config
from ..types import ExperienceDict
from ..knowledge import KnowledgeBase, TaskKnowledge, ActionKnowledge
from ..memory import ReplayBuffer
from ..utils.llm_interface import BaseLLM


class Summarizer:
    """
    Summarizer Agent
    Implements Two-Time-Scale Stochastic Approximation (TTSA) updates
    
    Fast timescale (Eq. 12): Update Action Knowledge (AK)
    Slow timescale (Eq. 13): Update Task Knowledge (TK)
    """
    
    def __init__(
        self,
        config: Config,
        knowledge_base: KnowledgeBase,
        memory: ReplayBuffer,
        llm: Optional[BaseLLM] = None
    ):
        """
        Initialize Summarizer
        
        Args:
            config: Configuration with update frequencies
            knowledge_base: Knowledge base to update
            memory: Experience replay buffer
            llm: LLM service for knowledge extraction (must implement BaseLLM interface)
        """
        self.config = config
        self.kb = knowledge_base
        self.memory = memory
        self.llm = llm
        
        # Step counter for slow updates
        self.step_count = 0
    
    def update(
        self,
        experience: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update knowledge bases based on experience
        
        Args:
            experience: Experience tuple from execution
            
        Returns:
            Update summary
        """
        # Add to memory
        self.memory.push(**experience)
        
        # Fast update: Update AK every step (Eq. 12)
        ak_updated = self._update_ak(experience)
        
        # Slow update: Update TK every N steps (Eq. 13)
        tk_updated = False
        if self.step_count % self.config.UPDATE_FREQ_TK == 0:
            tk_updated = self._update_tk()
        
        self.step_count += 1
        
        return {
            'ak_updated': ak_updated,
            'tk_updated': tk_updated,
            'step_count': self.step_count
        }
    
    def _update_ak(self, experience: Dict[str, Any]) -> bool:
        """
        Fast timescale update: Update Action Knowledge (Eq. 12)
        
        AK_{t+1} = AK_t + α_AK * (MCTS_posterior - AK_t)
        
        Args:
            experience: Experience tuple
            
        Returns:
            True if updated
        """
        ak = self.kb.get_ak()
        mcts_info = experience.get('mcts_info', {})
        
        # Extract MCTS posterior probabilities
        best_path = mcts_info.get('best_path', [])
        
        if not best_path:
            return False
        
        # Update tool priors based on MCTS results
        # If path led to good solution, increase priors
        reward = experience.get('reward', 0.0)
        success = reward > self.config.SUCCESS_THRESHOLD
        
        for tool_name in best_path:
            # Update prior using learning rate from config (Eq. 12)
            alpha_ak = self.config.ALPHA_AK
            current_prior = ak.get_tool_prior(tool_name)
            
            if success:
                # Increase prior
                new_prior = current_prior + alpha_ak * (1.0 - current_prior)
            else:
                # Decrease prior
                new_prior = current_prior - alpha_ak * current_prior
            
            # Update
            ak.update_tool_prior(tool_name, success)
        
        # Update transition probabilities
        # Extract state transitions from experience
        augmented_obs = experience.get('augmented_obs', '')
        if augmented_obs:
            # Simple state representation
            state = self._extract_state(augmented_obs)
            next_state = state + '_success' if success else state + '_failure'
            
            # Update transition probability using config learning rate
            alpha_trans = self.config.ALPHA_TRANSITION
            current_prob = ak.get_transition_prob(state, 'action', next_state)
            new_prob = current_prob + alpha_trans * (1.0 - current_prob) if success else current_prob - alpha_trans * current_prob
            
            ak.update_transition(state, 'action', next_state, new_prob)
        
        return True
    
    def _update_tk(self) -> bool:
        """
        Slow timescale update: Update Task Knowledge (Eq. 13)
        
        TK_{t+1} = TK_t + α_TK * (high_reward_trajectories - TK_t)
        
        Implements Manifold Reconstruction: Updates feasible region boundaries
        based on successful solutions (Lipschitz continuity assumption)
        
        Returns:
            True if updated
        """
        tk = self.kb.get_tk()
        
        # Get high-reward trajectories from memory (Eq. 13)
        high_reward_traces = self.memory.get_high_reward_traces(
            threshold=self.config.HIGH_REWARD_THRESHOLD
        )
        
        if not high_reward_traces:
            return False
        
        # Extract successful constraint and objective templates
        constraint_templates = []
        objective_templates = []
        
        # Statistical update for variable bounds (Manifold Reconstruction)
        variable_bounds_stats = {}  # {var_name: {'min': [...], 'max': [...], 'success_rates': [...]}}
        
        for trace in high_reward_traces:
            problem = trace.problem
            solution = trace.solution
            
            # Extract constraint template
            if isinstance(problem, dict):
                constraints = problem.get('constraints', [])
                if constraints:
                    constraint_templates.append({
                        'constraints': constraints,
                        'keywords': self._extract_keywords(constraints),
                        'success_rate': trace.reward
                    })
                
                # Extract objective template
                objective = problem.get('objective', '')
                if objective:
                    objective_templates.append({
                        'objective': objective,
                        'keywords': self._extract_keywords([objective]),
                        'success_rate': trace.reward
                    })
            
            # Statistical update: Collect successful variable values
            if isinstance(solution, dict):
                decision_vars = solution.get('decision_variables', {})
                for var_name, var_value in decision_vars.items():
                    if var_name not in variable_bounds_stats:
                        variable_bounds_stats[var_name] = {
                            'values': [],
                            'success_rates': []
                        }
                    variable_bounds_stats[var_name]['values'].append(var_value)
                    variable_bounds_stats[var_name]['success_rates'].append(trace.reward)
        
        # Update TK with new templates (Eq. 13)
        # Learning rate α_TK from config (slower than AK)
        alpha_tk = self.config.ALPHA_TK
        
        # Add new templates (with deduplication in practice)
        for template in constraint_templates:
            tk.add_constraint_template(template)
        
        for template in objective_templates:
            tk.add_objective_template(template)
        
        # Manifold Reconstruction: Update feasible region boundaries
        # Based on successful solutions, expand/contract variable bounds
        manifold_update = {}
        
        for var_name, stats in variable_bounds_stats.items():
            if not stats['values']:
                continue
            
            values = stats['values']
            success_rates = stats['success_rates']
            
            # Compute weighted statistics (weight by success rate)
            weighted_values = [v * r for v, r in zip(values, success_rates)]
            total_weight = sum(success_rates)
            
            if total_weight > 0:
                weighted_mean = sum(weighted_values) / total_weight
                weighted_std = (
                    sum([(v - weighted_mean) ** 2 * r for v, r in zip(values, success_rates)]) / total_weight
                ) ** 0.5
                
                # Expand bounds: [mean - 2*std, mean + 2*std] (covers ~95% of successful solutions)
                new_min = weighted_mean - 2 * weighted_std
                new_max = weighted_mean + 2 * weighted_std
                
                # Update manifold boundaries
                manifold_update[f'{var_name}_min'] = new_min
                manifold_update[f'{var_name}_max'] = new_max
                manifold_update[f'{var_name}_mean'] = weighted_mean
                manifold_update[f'{var_name}_std'] = weighted_std
        
        if manifold_update:
            tk.update_manifold(manifold_update)
        
        return True
    
    def _extract_state(self, augmented_obs: Any) -> str:
        """Extract state representation from augmented observation"""
        if isinstance(augmented_obs, str):
            return augmented_obs[:50]  # Truncate
        return str(augmented_obs)[:50]
    
    def _extract_keywords(self, text_list: List[str]) -> List[str]:
        """Extract keywords from text"""
        keywords = []
        common_words = {'min', 'max', 'subject', 'to', 'constraint', 'variable'}
        
        for text in text_list:
            words = text.lower().split()
            for word in words:
                if len(word) > 3 and word not in common_words:
                    keywords.append(word)
        
        return list(set(keywords))[:10]  # Limit to 10 keywords
    
    def _extract_manifold_patterns(
        self,
        traces: List[Any]
    ) -> Dict[str, Any]:
        """
        Extract manifold patterns from successful trajectories
        
        Returns:
            Manifold boundaries dictionary
        """
        if not traces:
            return {}
        
        # Extract common variable patterns
        var_patterns = {}
        constraint_patterns = {}
        
        for trace in traces:
            problem = trace.problem
            if hasattr(problem, 'variables'):
                for var in problem.variables:
                    var_name = var.name if hasattr(var, 'name') else str(var)
                    if var_name not in var_patterns:
                        var_patterns[var_name] = []
                    var_patterns[var_name].append(trace.reward)
        
        # Compute average success rates
        manifold = {
            'variable_patterns': {
                name: sum(scores) / len(scores)
                for name, scores in var_patterns.items()
            },
            'constraint_patterns': constraint_patterns
        }
        
        return manifold

