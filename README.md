# 🤖 Streamlit Ollama AI Agent Chatbot

A minimalist, high-end AI agent interface built on Streamlit and Ollama, inspired by Dieter Rams' design principles and Apple's user experience philosophy. Features intelligent agent capabilities with web search and real-time reasoning visualization.

## ✨ Features

### 🎨 **Premium Interface**
- **Jobs/Ive Aesthetic**: Ultra-minimalist design with Apple-inspired typography and animations
- **Real-time Streaming**: Live response streaming with elegant visual feedback
- **Responsive Design**: Seamless experience across desktop and mobile devices

### 🧠 **Intelligent Agent Mode**
- **React Agent Pattern**: Step-by-step reasoning with tool selection and execution
- **Web Search Integration**: Real-time web search with DuckDuckGo and fallback providers
- **Thinking Visualization**: Watch the AI reason through problems in real-time
- **Tool Orchestration**: Intelligent tool selection and result synthesis

### 📊 **Advanced Analytics**
- **Comprehensive Metrics**: Technical performance data (latency, tokens, throughput)
- **Agent Step Tracking**: Detailed reasoning process with execution times
- **Session Statistics**: Message counts, response times, and usage patterns

### ⚙️ **Powerful Configuration**
- **Model Flexibility**: Easy switching between Ollama models with live agent updates
- **Parameter Tuning**: Real-time adjustment of temperature, tokens, and sampling
- **Agent Controls**: Toggle reasoning visibility and tool availability

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull a model**: 
   ```bash
   ollama pull llama3.2:3b
   ```
3. **Start Ollama service**:
   ```bash
   ollama serve
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agent-aiops
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 🏗️ Project Structure

**Clean, modular architecture following enterprise best practices:**

```text
agent-aiops/
├── app.py                      # 🏠 Main application orchestration (68 lines)
├── requirements.txt            # 📦 Python dependencies
├── config/
│   ├── __init__.py
│   └── settings.py            # ⚙️ Configuration settings
├── services/
│   ├── __init__.py
│   ├── ollama_service.py      # 🤖 Ollama API integration
│   ├── agent_service.py       # 🧠 React Agent implementation
│   └── search_service.py      # 🔍 Web search providers
├── utils/
│   ├── __init__.py
│   └── chat_utils.py          # 💬 Chat utility functions
└── ui/                         # 🎨 Modular UI components
    ├── __init__.py
    ├── styles.py              # 🎨 CSS styling (322 lines)
    ├── components.py          # 🧩 UI components (332 lines)
    └── streamlit_utils.py     # ⚙️ Session & logic (278 lines)
```

### 📈 **Architecture Benefits**
- **92% reduction** in main app file size (817 → 68 lines)
- **Single responsibility** principle applied throughout
- **Reusable components** for future features
- **Easy testing** and debugging
- **Clean separation** of concerns

## ⚙️ Configuration

### Default Settings

- **Model**: `llama3.2:3b`
- **Temperature**: `0.7`
- **Max Tokens**: `1000`
- **Top-p**: `0.9`

### Environment Variables

- `OLLAMA_BASE_URL`: Ollama API URL (default: `http://localhost:11434`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `LOG_DIR`: Log directory (default: `logs`)
- `ENABLE_FILE_LOGGING`: Enable file logging (default: `true`)
- `ENABLE_JSON_LOGGING`: Enable structured JSON logging (default: `true`)
- `LOG_MAX_FILE_SIZE`: Maximum log file size in bytes (default: 10MB)
- `LOG_BACKUP_COUNT`: Number of backup log files to keep (default: 5)

## 🎨 Design Philosophy

This application follows Dieter Rams' ten principles of good design:

1. **Innovative**: Leveraging modern LLM technology
2. **Useful**: Practical chatbot with real metrics
3. **Aesthetic**: Clean, minimalist interface
4. **Understandable**: Intuitive user experience
5. **Unobtrusive**: Focused on functionality
6. **Honest**: Transparent performance metrics
7. **Long-lasting**: Timeless design principles
8. **Thorough**: Every detail considered
9. **Environmentally friendly**: Efficient resource usage
10. **As little design as possible**: Minimal yet complete

## 🔧 Advanced Usage

### 🤖 **Agent Mode**

Switch to **Agent Mode** in the sidebar to enable intelligent reasoning:

1. **Ask complex questions** that require research or multi-step thinking
2. **Watch the AI reason** through problems step-by-step
3. **See tool selection** and web search in real-time
4. **Toggle thinking visibility** to focus on results when needed

**Example Agent Queries:**
- "What's the latest news about AI developments?"
- "Compare the pros and cons of different programming languages"
- "Research and summarize recent scientific breakthroughs"

### 🔍 **Web Search Integration**

The agent automatically searches the web when needed:
- **Primary**: DuckDuckGo HTML scraping for comprehensive results
- **Fallback**: Instant Answer API for quick facts
- **Backup**: SearX provider for reliability
- **Manual**: Direct search links as last resort

### 🎛️ **Model Management**

1. **Add new models**:
   ```bash
   ollama pull <model-name>
   ```

2. **Live agent updates**: Agent automatically adapts to model changes
3. **Parameter tuning**: Real-time adjustment affects both chat and agent modes

### 📊 **Metrics & Analytics**

- **Technical Metrics**: Toggle detailed performance data
- **Agent Analytics**: Step execution times and tool usage
- **Session Export**: Download complete chat history with metadata

### 🎨 **UI Customization**

The modular architecture makes customization easy:
- **Styles**: Modify `ui/styles.py` for design changes
- **Components**: Extend `ui/components.py` for new features
- **Logic**: Add functionality in `ui/streamlit_utils.py`

## 📊 Logging & Monitoring

### Enterprise-Grade Logging System

The application includes a comprehensive logging system with:

#### **Features**
- **Structured JSON Logging**: Machine-readable logs for analysis
- **Beautiful Terminal Output**: Colored, formatted logs for development
- **Request Correlation**: Track requests across components with correlation IDs
- **Performance Monitoring**: Automatic timing and metrics collection
- **File Rotation**: Automatic log file rotation and retention
- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

#### **Log Files**
- `logs/agent-aiops.log`: Human-readable log file
- `logs/agent-aiops.json`: Structured JSON logs for analysis

#### **Real-Time Log Viewing**
```bash
# Follow logs in real-time with beautiful formatting
python scripts/view_logs.py --follow

# Filter logs by level
python scripts/view_logs.py --level ERROR --follow

# Filter by module
python scripts/view_logs.py --module ollama

# Search log messages
python scripts/view_logs.py --grep "search"

# Follow with correlation ID
python scripts/view_logs.py --correlation abc123 --follow
```

#### **Log Analysis**
```bash
# Generate comprehensive analysis report
python -m utils.log_analyzer

# Analyze last 6 hours
python -m utils.log_analyzer --hours 6

# Export performance data to CSV
python -m utils.log_analyzer --export-csv performance.csv

# JSON output for programmatic use
python -m utils.log_analyzer --format json
```

#### **What Gets Logged**
- **User Interactions**: Chat inputs, mode switches, parameter changes
- **Ollama API Calls**: Request/response times, token counts, model performance
- **Agent Reasoning**: Step-by-step thinking process and tool usage
- **Search Queries**: Provider performance, success rates, query patterns
- **Performance Metrics**: Response times, throughput, resource usage
- **Error Tracking**: Detailed error context and stack traces

#### **Sample Log Analysis Report**
```
============================================================
AGENT-AIOPS LOG ANALYSIS REPORT
Time Period: Last 24 hours
Generated: 2024-01-15 14:30:25
============================================================

📊 PERFORMANCE SUMMARY
------------------------------
Ollama API:
  • Total Requests: 47
  • Avg Response Time: 1,245.3ms
  • Min/Max Response: 456.2ms / 3,891.7ms

Search Performance:
  • Total Queries: 12
  • Avg Response Time: 847.1ms
  • Min/Max Response: 234.5ms / 2,156.3ms

👥 USER ACTIVITY
------------------------------
Sessions: 8
Mode Usage: Normal (15) | Agent (32)
Avg Session Length: 3.2 interactions

🔍 SEARCH ANALYTICS
------------------------------
Total Queries: 12
  • duckduckgo: 9 queries (88.9% success)
  • searx: 3 queries (100.0% success)

🚨 ERROR SUMMARY
------------------------------
Total Errors: 2
Error Rate: 0.12%
Errors by Module:
  • search_service: 2
```

## 🐛 Troubleshooting

### Common Issues

1. **"Ollama service is not running"**
   - Start Ollama: `ollama serve`
   - Check if port 11434 is available
   - Verify service health in sidebar

2. **"No models available"**
   - Install a model: `ollama pull llama3.2:3b`
   - Verify installation: `ollama list`
   - Restart the application

3. **"Agent not properly initialized"**
   - Ensure Ollama is running and model is selected
   - Check browser console for errors
   - Try switching models in sidebar

4. **Web search not working**
   - Check internet connection
   - DuckDuckGo may be rate-limiting
   - Agent will automatically try fallback providers

5. **Slow responses**
   - Check hardware resources (CPU/RAM)
   - Try a smaller model (e.g., `llama3.2:1b`)
   - Reduce max_tokens parameter
   - Disable agent mode for faster responses

6. **UI components not loading**
   - Clear browser cache
   - Check browser console for JavaScript errors
   - Ensure all UI modules are properly imported

7. **Logging issues**
   - Check write permissions for the `logs/` directory
   - Verify log directory exists: `mkdir -p logs`
   - Check disk space: `df -h`
   - View recent errors: `python scripts/view_logs.py --level ERROR`
   - Monitor in real-time: `python scripts/view_logs.py --follow`

## 🛠️ Development

### **Running in Development Mode**

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd agent-aiops
   pip install -r requirements.txt
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ollama pull llama3.2:3b
   ```

3. **Run with hot reload**:
   ```bash
   streamlit run app.py --server.runOnSave true
   ```

### **Code Structure**

The modular architecture makes development easier:
- **`app.py`**: Main orchestration (keep minimal)
- **`ui/`**: All UI components and styling
- **`services/`**: Business logic and API integrations  
- **`utils/`**: Shared utility functions
- **`config/`**: Application configuration

### **Adding New Features**

1. **New UI components**: Add to `ui/components.py`
2. **New agent tools**: Extend `services/agent_service.py`
3. **New styling**: Modify `ui/styles.py`
4. **New utilities**: Add to appropriate `utils/` module

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### **Contribution Guidelines**
- Follow the modular architecture patterns
- Maintain Jobs/Ive design principles
- Add appropriate documentation
- Test both normal and agent modes

## 📞 Support

For support and questions, please open an issue in the repository.

---

**Built with ❤️ using enterprise-grade architecture and Apple-inspired design principles.**
