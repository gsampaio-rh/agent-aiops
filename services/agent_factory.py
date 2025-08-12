"""
Agent Factory

Factory for creating LangGraphAgent instances with proper tool registration.
Simplified after ReactAgent deprecation.
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
        model: str = "llama3.2:3b",
        llm_service: Optional[LLMServiceInterface] = None,
        **kwargs
    ) -> AgentServiceInterface:
        """
        Create a LangGraphAgent instance.
        
        Args:
            model: Model name to use
            llm_service: Optional LLM service instance
            **kwargs: Additional arguments passed to agent constructor
            
        Returns:
            AgentServiceInterface: Configured LangGraphAgent instance
            
        Raises:
            ImportError: If LangGraph dependencies are missing
        """
        logger = get_logger(__name__)
        
        logger.info("Creating LangGraphAgent", model=model)
        
        return AgentFactory._create_langgraph_agent(model, llm_service, **kwargs)
    

    
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
                from services.langgraph_agent_service import WebSearchTool
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
        
        # Register RAG tool if enabled
        from config.constants import RAG_CONFIG
        if RAG_CONFIG.get("enabled", True):
            try:
                # Import here to avoid circular imports
                from services.rag_tool import RAGTool
                rag_tool = RAGTool()
                agent.register_tool(rag_tool)
                logger.info("Registered RAG tool")
            except ImportError as e:
                logger.error("Could not import RAGTool", error=str(e))
            except Exception as e:
                logger.error("Failed to register RAG tool", error=str(e))
    
    @staticmethod
    def get_available_agent_types() -> Dict[str, str]:
        """Get available agent types and their descriptions."""
        return {
            "langgraph": "LangGraph-based agent with workflow management"
        }
    
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
    model: str = "llama3.2:3b",
    **kwargs
) -> AgentServiceInterface:
    """
    Convenience function to create a LangGraphAgent.
    
    Args:
        model: Model name
        **kwargs: Additional arguments
        
    Returns:
        AgentServiceInterface: Configured LangGraphAgent instance
    """
    return AgentFactory.create_agent(model=model, **kwargs)
