# -*- coding: utf-8 -*-
"""
Planner Tools
Tool definitions for state augmentation
"""

from typing import Dict, Any, List


class ToolRegistry:
    """
    Tool registry for Planner
    Manages available tools for state augmentation
    """
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
    
    def register(self, name: str, tool: Any) -> None:
        """Register a tool"""
        self._tools[name] = tool
    
    def list_tools(self) -> List[str]:
        """List available tool names"""
        return list(self._tools.keys())
    
    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool"""
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found")
        return self._tools[name].execute(**kwargs)


def create_default_tools() -> ToolRegistry:
    """Create default tool registry"""
    registry = ToolRegistry()
    
    # Add default tools (placeholders)
    # In practice, these would be actual tool implementations
    class SummarizationTool:
        def execute(self, **kwargs):
            return {"summary": "State summary"}
    
    class FaultClassificationTool:
        def execute(self, **kwargs):
            return {"fault_type": "normal"}
    
    registry.register("summarize", SummarizationTool())
    registry.register("classify_fault", FaultClassificationTool())
    
    return registry

