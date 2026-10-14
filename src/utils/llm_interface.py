# -*- coding: utf-8 -*-
"""
LLM Interface Abstraction.

This module defines the abstract interface for LLM services used throughout
the ADA framework. Users should implement this interface to integrate their
own LLM provider (e.g., OpenAI, DeepSeek, local models).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLM(ABC):
    """
    Abstract base class for LLM services.
    
    This interface defines the methods required for LLM integration in:
    - Planner: Problem formulation from augmented states
    - Judger: Logical coherence evaluation
    - Summarizer: Knowledge extraction from experiences
    
    To integrate your own LLM:
    1. Inherit from BaseLLM
    2. Implement the `chat` method
    3. Pass your LLM instance to the respective components
    """
    
    @abstractmethod
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Send a chat message to the LLM and get a response.
        
        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt for context
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLM response as a string
            
        Raises:
            RuntimeError: If LLM call fails (API error, timeout, etc.)
        """
        pass
    
    def format_response(self, response: str) -> Dict[str, Any]:
        """
        Optional: Format LLM response into structured data.
        
        This is a helper method that can be overridden to parse
        structured responses (e.g., JSON) from the LLM.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Formatted response dictionary
        """
        return {"raw_response": response}

