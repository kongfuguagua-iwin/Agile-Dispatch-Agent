# -*- coding: utf-8 -*-
"""
ADA Agent - Main Implementation.

This module implements Algorithm 1 from the paper:
"ADA: Resolving Attribution Ambiguity in End-to-End Power System Dispatch 
via Two-Time-Scale Stochastic Approximation"

The agent orchestrates the four main components:
- Planner (Section 3.1): Active State Augmentation via MCTS
- Solver (Section 3.2): Adaptive Algorithm Selection
- Judger (Section 3.3): Hybrid Reward Evaluation
- Summarizer (Section 3.4): Two-Time-Scale Stochastic Approximation Updates
"""

from typing import Dict, Any, Optional, Tuple, List
import logging

from .config import Config
from .types import SolutionDict, ProblemDict, EnvironmentStateDict, MCTSInfoDict, EvaluationDict
from .knowledge import KnowledgeBase
from .memory import ReplayBuffer
from .planner import Planner
from .solver import Solver
from .judger import Judger
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


class ADAgent:
    """
    Agile Dispatch Agent
    Main agent class implementing Algorithm 1
    
    Core workflow:
    1. Planner: Active State Augmentation (MCTS)
    2. Solver: Adaptive Algorithm Selection
    3. Judger: Dual Evaluation
    4. Summarizer: TTSA Updates
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        planner: Optional[Planner] = None,
        solver: Optional[Solver] = None,
        judger: Optional[Judger] = None,
        summarizer: Optional[Summarizer] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        memory: Optional[ReplayBuffer] = None
    ):
        """
        Initialize ADAgent
        
        Args:
            config: Configuration
            planner: Planner instance
            solver: Solver instance
            judger: Judger instance
            summarizer: Summarizer instance
            knowledge_base: Knowledge base
            memory: Experience replay buffer
        """
        self.config = config or Config()
        
        # Initialize knowledge base and memory
        self.kb = knowledge_base or KnowledgeBase()
        self.memory = memory or ReplayBuffer()
        
        # Initialize components
        self.planner = planner or Planner(
            config=self.config,
            ak=self.kb.get_ak()
        )
        
        # Initialize solver with default algorithms if not provided
        if solver is None:
            from .solver.algorithms import create_default_algorithms
            algorithms = create_default_algorithms()
            self.solver = Solver(config=self.config, algorithms=algorithms)
        else:
            self.solver = solver
        
        self.judger = judger or Judger(config=self.config)
        self.summarizer = summarizer or Summarizer(
            config=self.config,
            knowledge_base=self.kb,
            memory=self.memory
        )
        
        # Step counter
        self.step_count = 0
    
    def run_step(
        self,
        observation: Any,
        environment: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute one step of Algorithm 1
        
        Args:
            observation: Environment observation
            environment: Environment instance (optional)
            
        Returns:
            Execution result with solution and metadata
        """
        self.step_count += 1
        logger.info(f"Step {self.step_count}: Starting ADA execution")
        
        # 1. Planner: Active State Augmentation
        # Uses AK for tool selection priors
        augmented_obs, mcts_info = self.planner.plan(
            observation,
            ak_prior=self.kb.get_ak()
        )
        
        # Formulate mathematical problem (J, Theta) using TK
        problem_tuple = self._formulate_problem(augmented_obs)
        objective, constraints = problem_tuple
        
        # Extract variables from observation/environment
        variables = self._extract_variables(observation, environment)
        
        # Retry Loop (Max Retries K)
        solution = None
        feedback = None
        
        for retry in range(self.config.MAX_RETRIES):
            logger.debug(f"Attempt {retry + 1}/{self.config.MAX_RETRIES}")
            
            # 2. Solver: Adaptive Selection
            try:
                solution_result = self.solver.solve(
                    objective=objective,
                    constraints=constraints,
                    variables=variables
                )
                
                solution = solution_result.get('solution', {})
                algorithm_used = solution_result.get('algorithm', 'unknown')
                
            except (ValueError, KeyError) as e:
                # Specific errors: missing algorithm, invalid input
                logger.error(f"Solver configuration error: {e}")
                solution = {
                    'is_feasible': False,
                    'decision_variables': {},
                    'objective_value': float('inf'),
                    'message': f"Configuration error: {str(e)}"
                }
                algorithm_used = 'failed'
            except RuntimeError as e:
                # Runtime errors: solver timeout, numerical issues
                logger.error(f"Solver runtime error: {e}")
                solution = {
                    'is_feasible': False,
                    'decision_variables': {},
                    'objective_value': float('inf'),
                    'message': f"Runtime error: {str(e)}"
                }
                algorithm_used = 'failed'
            except Exception as e:
                # Catch-all for unexpected errors (should be rare)
                logger.exception(f"Unexpected solver error: {e}")
                solution = {
                    'is_feasible': False,
                    'decision_variables': {},
                    'objective_value': float('inf'),
                    'message': f"Unexpected error: {str(e)}"
                }
                algorithm_used = 'failed'
            
            # 3. Judger: Evaluation
            # Check physical feasibility
            is_feasible = solution.get('is_feasible', False)
            
            if is_feasible:
                # Prepare environment state for reward calculation
                env_state = self._prepare_environment_state(
                    observation, environment, solution
                )
                
                # Compute hybrid reward
                evaluation = self.judger.evaluate(
                    solution=solution,
                    problem={'objective': objective, 'constraints': constraints},
                    environment_state=env_state
                )
                
                feedback = evaluation
                reward = evaluation.get('reward', 0.0)
                passed = evaluation.get('passed', False)
                
                # Store experience
                experience = {
                    'observation': observation,
                    'augmented_obs': augmented_obs,
                    'mcts_info': mcts_info,
                    'problem': {'objective': objective, 'constraints': constraints},
                    'solution': solution,
                    'algorithm': algorithm_used,
                    'reward': reward,
                    'feedback': feedback
                }
                
                self.memory.push(**experience)
                
                # Break if good enough (using config threshold)
                if passed or reward > self.config.HIGH_REWARD_THRESHOLD:
                    logger.info(f"Solution accepted: reward={reward:.4f}")
                    break
                else:
                    logger.warning(f"Solution rejected: reward={reward:.4f}, retrying...")
            else:
                # Solution not feasible, refine constraints
                logger.warning("Solution not feasible, refining constraints...")
                problem_tuple = self._refine_constraints(
                    problem_tuple, solution.get('message', '')
                )
                objective, constraints = problem_tuple
        
        # 4. Summarizer: Asynchronous Update
        # Fast Update (Every step)
        if solution is not None and feedback is not None:
            update_result = self.summarizer.update({
                'observation': observation,
                'augmented_obs': augmented_obs,
                'mcts_info': mcts_info,
                'problem': {'objective': objective, 'constraints': constraints},
                'solution': solution,
                'algorithm': algorithm_used,
                'reward': feedback.get('reward', 0.0),
                'feedback': feedback
            })
            
            logger.debug(f"Knowledge update: AK={update_result['ak_updated']}, "
                        f"TK={update_result['tk_updated']}")
        
        # Return result
        return {
            'step': self.step_count,
            'solution': solution,
            'algorithm': algorithm_used,
            'reward': feedback.get('reward', 0.0) if feedback else 0.0,
            'passed': feedback.get('passed', False) if feedback else False,
            'mcts_info': mcts_info,
            'problem': {'objective': objective, 'constraints': constraints}
        }
    
    def _formulate_problem(
        self,
        augmented_obs: Any
    ) -> Tuple[str, List[str]]:
        """
        Formulate optimization problem from augmented observation
        
        Uses TK templates to guide formulation
        
        Args:
            augmented_obs: Augmented observation from Planner
            
        Returns:
            (objective, constraints_list)
        """
        # Retrieve relevant TK templates
        tk = self.kb.get_tk()
        context = str(augmented_obs)
        templates = tk.get_relevant_templates(context)
        
        # Use templates to guide formulation
        # In practice, this would use LLM with TK context
        if templates:
            # Use template structure
            template = templates[0]
            objective = template.get('objective', 'min x')
            constraints = template.get('constraints', ['x >= 0'])
        else:
            # Default formulation
            objective = 'min x'
            constraints = ['x >= 0']
        
        return objective, constraints
    
    def _extract_variables(
        self,
        observation: Any,
        environment: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Extract variable definitions from observation
        
        Args:
            observation: Environment observation
            environment: Environment instance
            
        Returns:
            Variable definitions dictionary
        """
        # Extract variable information
        # In practice, this would parse observation structure
        variables = {
            'n_continuous': 10,
            'n_integer': 0,
            'lower_bounds': [0.0] * 10,
            'upper_bounds': [100.0] * 10,
            'types': ['continuous'] * 10
        }
        
        return variables
    
    def _prepare_environment_state(
        self,
        observation: Any,
        environment: Optional[Any],
        solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare environment state for reward calculation
        
        Args:
            observation: Observation
            environment: Environment (MockL2RPN or real)
            solution: Solution
            
        Returns:
            Environment state dictionary
        """
        # Use environment if available
        if environment is not None:
            if hasattr(environment, 'get_state'):
                env_state = environment.get_state()
            elif hasattr(environment, 'state'):
                env_state = environment.state
            else:
                env_state = {}
        else:
            # Fallback: create minimal state
            env_state = {}
        
        # Ensure required fields exist
        if 'initial_gen_powers' not in env_state:
            env_state['initial_gen_powers'] = env_state.get('gen_powers', [50.0] * 10)
        
        if 'gen_costs' not in env_state:
            env_state['gen_costs'] = [1.0 + i * 0.1 for i in range(10)]
        
        if 'line_loadings' not in env_state:
            env_state['line_loadings'] = [0.7] * 20
        
        if 'line_max_capacities' not in env_state:
            env_state['line_max_capacities'] = [100.0] * 20
        
        if 'reference_topology' not in env_state:
            env_state['reference_topology'] = [1] * 20
        
        # Update with solution
        decision_vars = solution.get('decision_variables', {})
        if 'gen_powers' in decision_vars:
            env_state['gen_powers'] = decision_vars['gen_powers']
        elif 'gen_powers' not in env_state:
            env_state['gen_powers'] = env_state.get('initial_gen_powers', [50.0] * 10)
        
        return env_state
    
    def _refine_constraints(
        self,
        problem_tuple: Tuple[str, List[str]],
        error_message: str
    ) -> Tuple[str, List[str]]:
        """
        Refine constraints based on failure feedback
        
        Args:
            problem_tuple: Current problem (objective, constraints)
            error_message: Error message from solver
            
        Returns:
            Refined problem tuple
        """
        objective, constraints = problem_tuple
        
        # Simple refinement: add safety constraints
        # In practice, would use LLM to analyze error and refine
        if 'bound' in error_message.lower() or 'limit' in error_message.lower():
            # Add bound constraints
            constraints.append('x >= 0')
            constraints.append('x <= 100')
        
        return objective, constraints
    
    def reset(self) -> None:
        """Reset agent state"""
        self.step_count = 0
        self.memory.clear()
        logger.info("Agent reset")

