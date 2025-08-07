#!/usr/bin/env python3
"""
Development server runner for the Ollama Chatbot.
"""

import subprocess
import sys
import os


def check_ollama():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Run the Streamlit application."""
    print("🤖 Starting Ollama Chatbot...")
    
    # Check if Ollama is running
    if not check_ollama():
        print("⚠️  Warning: Ollama service doesn't seem to be running.")
        print("   Please start it with: ollama serve")
        print("   Then install a model: ollama pull llama3.2:3b")
        print()
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
