# -*- coding: utf-8 -*-
"""
Planner Prompts
LLM prompts for problem formulation
"""

SYSTEM_PROMPT = """
You are a planning agent that converts unstructured requirements into formal optimization problems.
Your task is to analyze the augmented state and generate a mathematical optimization formulation.
"""


def build_formulation_prompt(
    augmented_state: str,
    task_knowledge: str = "",
    feedback: str = ""
) -> str:
    """
    Build prompt for problem formulation
    
    Args:
        augmented_state: Augmented state from MCTS
        task_knowledge: Task knowledge context
        feedback: Previous feedback (if retrying)
        
    Returns:
        Formatted prompt
    """
    prompt = f"""
{SYSTEM_PROMPT}

Augmented State:
{augmented_state}

Task Knowledge:
{task_knowledge}

{f'Previous Feedback: {feedback}' if feedback else ''}

Please generate a formal optimization problem in JSON format:
{{
    "objective": "min f(x) or max f(x)",
    "constraints": ["g1(x) <= 0", "g2(x) >= 0", ...],
    "variables": [
        {{"name": "x1", "type": "continuous", "lower_bound": 0, "upper_bound": 100}}
    ]
}}
"""
    return prompt

