#!/usr/bin/env python3
"""
Test script for LangGraph integration.

This script demonstrates the LangGraph agent working alongside the original ReAct agent.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.agent_factory import create_agent, AgentFactory
from core.models.agent import StepType


def test_agent_comparison():
    """Compare ReactAgent and LangGraphAgent implementations."""
    print("🔬 Testing Agent Implementations")
    print("=" * 50)
    
    # Test query
    test_query = "Hello! Can you tell me about yourself?"
    
    print(f"📝 Test Query: {test_query}")
    print()
    
    # Test ReactAgent
    print("🤖 Testing ReactAgent:")
    print("-" * 25)
    try:
        react_agent = create_agent(agent_type="react", model="llama3.2:3b")
        print(f"✅ Created: {type(react_agent).__name__}")
        print(f"📊 Available tools: {[tool.name for tool in react_agent.get_available_tools()]}")
        
        # Test a simple query (non-streaming for simplicity)
        try:
            response = react_agent.process_query(test_query)
            print(f"📈 Steps generated: {len(response.steps)}")
            print(f"⏱️  Processing time: {response.total_time_ms}ms")
            if response.get_final_answer():
                print(f"💬 Final answer: {response.get_final_answer()[:100]}...")
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
        
    except Exception as e:
        print(f"❌ ReactAgent creation failed: {e}")
    
    print()
    
    # Test LangGraphAgent
    print("🔗 Testing LangGraphAgent:")
    print("-" * 25)
    try:
        langgraph_agent = create_agent(agent_type="langgraph", model="llama3.2:3b")
        print(f"✅ Created: {type(langgraph_agent).__name__}")
        print(f"📊 Available tools: {[tool.name for tool in langgraph_agent.get_available_tools()]}")
        
        # Test a simple query (non-streaming for simplicity)
        try:
            response = langgraph_agent.process_query(test_query)
            print(f"📈 Steps generated: {len(response.steps)}")
            print(f"⏱️  Processing time: {response.total_time_ms}ms")
            if response.get_final_answer():
                print(f"💬 Final answer: {response.get_final_answer()[:100]}...")
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
        
    except Exception as e:
        print(f"❌ LangGraphAgent creation failed: {e}")
    
    print()
    print("✨ Test completed!")


def test_agent_factory():
    """Test the agent factory functionality."""
    print("🏭 Testing Agent Factory")
    print("=" * 30)
    
    # Test availability checks
    print(f"🔍 LangGraph available: {AgentFactory.is_langgraph_available()}")
    
    # Test available types
    types = AgentFactory.get_available_agent_types()
    print(f"📋 Available agent types:")
    for agent_type, description in types.items():
        print(f"  • {agent_type}: {description}")
    
    print()


def main():
    """Main test function."""
    print("🚀 LangGraph Integration Test")
    print("=" * 40)
    print()
    
    test_agent_factory()
    test_agent_comparison()


if __name__ == "__main__":
    main()
