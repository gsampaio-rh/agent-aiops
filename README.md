# 🤖 Agent-AIOps

A sophisticated AI agent framework featuring LangGraph-based workflow management with a premium Streamlit interface. Built for enterprise-grade AI operations with local Ollama models.

## ✨ Key Features

### 🧠 **Advanced Agent Architecture**
- **LangGraphAgent**: Workflow management with state machines and checkpointing
- **Tool Integration**: Web search, terminal commands with safety validation
- **Real-time Reasoning**: Step-by-step thought process visualization

### 🔧 **Enterprise Features**
- **Performance Monitoring**: Comprehensive metrics and logging
- **Model Flexibility**: Easy switching between Ollama models
- **Configuration Management**: Environment-based settings with validation

## 🚀 Quick Start

### Prerequisites
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull a model**: `ollama pull llama3.2:3b`
3. **Start Ollama**: `ollama serve`

### Installation
```bash
git clone <repository-url>
cd agent-aiops
pip install -r requirements.txt
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 🏗️ Architecture

### Clean, Modular Design
```
agent-aiops/
├── app.py                          # Main application (68 lines)
├── config/                         # Configuration & constants
│   ├── settings.py                # Pydantic settings
│   └── constants.py               # Agent & tool configuration
├── core/                          # Core interfaces & models
│   ├── interfaces/               # Service contracts
│   │   ├── agent_service.py      # Agent interface
│   │   ├── llm_service.py        # LLM interface
│   │   └── search_service.py     # Search interface
│   ├── models/                   # Data models
│   │   ├── agent.py             # Agent response models
│   │   ├── chat.py              # Chat models
│   │   └── search.py            # Search models
│   ├── exceptions/               # Error handling
│   └── validators/               # Input validation
├── services/                      # Business logic
│   ├── agent_factory.py         # Agent creation & management
│   ├── langgraph_agent_service.py # LangGraphAgent implementation
│   ├── ollama_service.py        # Ollama LLM integration
│   ├── search_service.py        # Multi-provider web search
│   ├── mcp_service.py           # Model Context Protocol
│   └── terminal_tool.py         # Safe terminal execution
├── ui/                           # Modular UI components
│   ├── components.py            # Reusable UI components
│   ├── styles.py               # Custom CSS styling
│   ├── streamlit_utils.py      # Session & utilities
│   ├── chat_display.py         # Chat interface
│   ├── sidebar.py              # Configuration sidebar
│   └── tool_execution.py       # Tool execution UI
└── utils/                        # Utilities
    ├── logger.py               # Enterprise logging
    ├── chat_utils.py           # Chat utilities
    └── log_analyzer.py         # Log analysis tools
```

## 🤖 Agent Implementation

### LangGraphAgent
- **Workflow Management**: Graph-based state machines with conditional routing
- **Checkpointing**: Memory persistence across conversations
- **Conditional Logic**: Smart routing between reasoning and tool usage
- **Performance**: ~1.6s average response time
- **Tool Integration**: Seamless web search and terminal execution

## ⚙️ Configuration

### Environment Variables
```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.2:3b

# Model Parameters
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1000
DEFAULT_TOP_P=0.9

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
ENABLE_FILE_LOGGING=true
ENABLE_JSON_LOGGING=true
```

### Agent Configuration (`config/constants.py`)
```python
AGENT_CONFIG = {
    "max_iterations": 10,
    "enable_web_search": True,
    "enable_terminal": True,
    "show_thinking_process": True,
}
```

## 🔍 Tools & Capabilities

### Built-in Tools
- **Web Search**: DuckDuckGo and SearX providers with fallback
- **Terminal Commands**: Safe execution with security validation
- **Model Context Protocol**: Desktop commander integration

### Tool Safety
- Command validation with allow/deny patterns
- User confirmation for potentially dangerous operations
- Execution timeouts and output size limits
- Comprehensive logging of all tool usage

## 📊 Monitoring & Logging

### Enterprise-Grade Logging
- **Structured JSON logs** for analysis
- **Correlation IDs** for request tracking
- **Performance metrics** with detailed timing
- **Real-time log viewing** with filtering

### Log Analysis
```bash
# Real-time log viewing
python -m utils.log_analyzer --follow

# Performance analysis
python -m utils.log_analyzer --format json

# Filter by level or module
python -m utils.log_analyzer --level ERROR --module search
```

## 🛠️ Development

### Running in Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start with auto-reload
streamlit run app.py --server.runOnSave true

# Run tests
python test_agent.py
```

### Adding New Features
1. **New Tools**: Implement `ToolInterface` in `services/`
2. **UI Components**: Add to `ui/components.py`
3. **Agent Features**: Extend LangGraphAgent workflows
4. **Styling**: Modify `ui/styles.py`

## 🐛 Troubleshooting

### Common Issues
1. **"Ollama not running"**: Start with `ollama serve`
2. **"No models available"**: Install with `ollama pull llama3.2:3b`
3. **Slow responses**: Try smaller models or reduce max_tokens
4. **LangGraph errors**: Ensure all dependencies are installed

### Performance Optimization
- Use smaller models for faster responses
- Adjust temperature and token limits
- Disable agent mode for simple queries
- Monitor logs for bottlenecks

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions welcome! Please:
- Follow the modular architecture patterns
- Maintain interface contracts
- Add appropriate tests and documentation
- Test the LangGraph agent implementation

---

**Built with ❤️ using enterprise-grade architecture and modern AI patterns.**