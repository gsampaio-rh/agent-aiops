"""
Agent Factory

Factory for creating different types of agents based on configuration.
Supports both ReactAgent and LangGraphAgent implementations.
"""

from typing import Optional, Dict, Any
from core.interfaces.agent_service import AgentServiceInterface
from core.interfaces.llm_service import LLMServiceInterface
from config.constants import AGENT_CONFIG
from utils.logger import get_logger


class AgentFactory:
    """Factory for creating agent instances."""
    
    @staticmethod
    def create_agent(
        agent_type: Optional[str] = None,
        model: str = "llama3.2:3b",
        llm_service: Optional[LLMServiceInterface] = None,
        **kwargs
    ) -> AgentServiceInterface:
        """
        Create an agent instance based on configuration.
        
        Args:
            agent_type: Type of agent to create ("react" or "langgraph")
            model: Model name to use
            llm_service: Optional LLM service instance
            **kwargs: Additional arguments passed to agent constructor
            
        Returns:
            AgentServiceInterface: Configured agent instance
            
        Raises:
            ValueError: If agent_type is not supported
            ImportError: If required dependencies are missing
        """
        logger = get_logger(__name__)
        
        # Determine agent type from config if not specified
        if agent_type is None:
            # Default to LangGraph if available, fallback to react
            if AgentFactory.is_langgraph_available():
                agent_type = AGENT_CONFIG.get("agent_type", "langgraph")
            else:
                agent_type = "react"
        
        logger.info("Creating agent", agent_type=agent_type, model=model)
        
        if agent_type == "react":
            return AgentFactory._create_react_agent(model, llm_service, **kwargs)
        elif agent_type == "langgraph":
            return AgentFactory._create_langgraph_agent(model, llm_service, **kwargs)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}. Use 'react' or 'langgraph'")
    
    @staticmethod
    def _create_react_agent(
        model: str,
        llm_service: Optional[LLMServiceInterface] = None,
        **kwargs
    ) -> AgentServiceInterface:
        """Create a ReactAgent instance."""
        from services.agent_service import ReactAgent
        from services.search_service import WebSearchService
        
        search_service = kwargs.get("search_service") or WebSearchService()
        
        agent = ReactAgent(
            model=model,
            llm_service=llm_service,
            search_service=search_service
        )
        
        # Register tools based on configuration
        AgentFactory._register_default_tools(agent)
        
        return agent
    
    @staticmethod
    def _create_langgraph_agent(
        model: str,
        llm_service: Optional[LLMServiceInterface] = None,
        **kwargs
    ) -> AgentServiceInterface:
        """Create a LangGraphAgent instance."""
        try:
            from services.langgraph_agent_service import LangGraphAgent
        except ImportError as e:
            raise ImportError(
                f"LangGraph dependencies not available: {e}. "
                "Install with: pip install langgraph langchain langchain-core langchain-community"
            )
        
        agent = LangGraphAgent(
            model=model,
            llm_service=llm_service
        )
        
        # Register tools based on configuration
        AgentFactory._register_default_tools(agent)
        
        return agent
    
    @staticmethod
    def _register_default_tools(agent: AgentServiceInterface) -> None:
        """Register default tools based on configuration."""
        logger = get_logger(__name__)
        
        # Register web search tool if enabled
        if AGENT_CONFIG.get("enable_web_search", True):
            try:
                # Import here to avoid circular imports
                from services.agent_service import WebSearchTool
                web_search_tool = WebSearchTool()
                agent.register_tool(web_search_tool)
                logger.info("Registered web search tool")
            except ImportError as e:
                logger.error("Could not import WebSearchTool", error=str(e))
            except Exception as e:
                logger.error("Failed to register web search tool", error=str(e))
        
        # Register terminal tool if enabled
        if AGENT_CONFIG.get("enable_terminal", False):
            try:
                # Import here to avoid circular imports
                from services.terminal_tool import TerminalTool
                terminal_tool = TerminalTool(require_confirmation=False)
                agent.register_tool(terminal_tool)
                logger.info("Registered terminal tool")
            except ImportError as e:
                logger.error("Could not import TerminalTool", error=str(e))
            except Exception as e:
                logger.error("Failed to register terminal tool", error=str(e))
    
    @staticmethod
    def get_available_agent_types() -> Dict[str, str]:
        """Get available agent types and their descriptions."""
        types = {
            "react": "Original ReAct-pattern agent with step-by-step reasoning",
        }
        
        # Check if LangGraph is available
        try:
            import langgraph
            types["langgraph"] = "LangGraph-based agent with workflow management"
        except ImportError:
            pass
        
        return types
    
    @staticmethod
    def is_langgraph_available() -> bool:
        """Check if LangGraph dependencies are available."""
        try:
            import langgraph
            import langchain
            import langchain_core
            return True
        except ImportError:
            return False


# Convenience function for creating agents
def create_agent(
    agent_type: Optional[str] = None,
    model: str = "llama3.2:3b",
    **kwargs
) -> AgentServiceInterface:
    """
    Convenience function to create an agent.
    
    Args:
        agent_type: Type of agent ("react" or "langgraph")
        model: Model name
        **kwargs: Additional arguments
        
    Returns:
        AgentServiceInterface: Configured agent instance
    """
    return AgentFactory.create_agent(agent_type=agent_type, model=model, **kwargs)
