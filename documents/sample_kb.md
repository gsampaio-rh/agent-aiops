# Agent-AIOps Knowledge Base

## Getting Started

### Installation Steps

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull a model**: `ollama pull llama3.2:3b`
3. **Start Ollama**: `ollama serve`
4. **Install dependencies**: `pip install -r requirements.txt`
5. **Run the application**: `streamlit run app.py`

### System Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended for larger models)
- 10GB free disk space for models
- Internet connection for initial model download

## Architecture Overview

### Core Components

The Agent-AIOps system consists of several key components:

- **LangGraph Agent**: Manages workflow and tool orchestration
- **Streamlit UI**: Provides the interactive web interface
- **Ollama Service**: Handles local LLM inference
- **Tool System**: Extensible framework for adding capabilities

### Agent Workflow

1. **Query Processing**: User input is parsed and validated
2. **Reasoning**: Agent determines if tools are needed
3. **Tool Selection**: Appropriate tools are chosen
4. **Execution**: Tools are executed with user confirmation
5. **Synthesis**: Results are combined into final response

## Troubleshooting

### Common Issues

#### Ollama Not Running
**Problem**: "Ollama service is not running" error
**Solution**: 
- Start Ollama with `ollama serve`
- Check if port 11434 is available
- Verify Ollama installation

#### No Models Available
**Problem**: "No models available" message
**Solution**:
- Install a model: `ollama pull llama3.2:3b`
- Check available models: `ollama list`
- Verify model download completed

#### Slow Performance
**Problem**: Agent responses are very slow
**Solutions**:
- Use smaller models (llama3.2:1b instead of 7b)
- Reduce max_tokens in settings
- Check system resources (CPU/RAM usage)
- Disable unnecessary tools

#### Memory Issues
**Problem**: Out of memory errors
**Solutions**:
- Use smaller models
- Reduce conversation history length
- Restart Ollama service
- Check available system RAM

### Performance Optimization

#### Model Selection
- **llama3.2:1b**: Fastest, suitable for simple tasks
- **llama3.2:3b**: Good balance of speed and capability
- **llama3.2:7b**: Best quality, requires more resources

#### Configuration Tuning
- Lower temperature for more focused responses
- Reduce max_tokens for faster generation
- Adjust chunk_size for RAG queries
- Enable tool caching for repeated operations

## Configuration

### Environment Variables

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.2:3b

# Model Parameters
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1000
DEFAULT_TOP_P=0.9

# RAG Configuration
RAG_ENABLED=true
RAG_DOCUMENTS_PATH=./documents
RAG_MAX_RESULTS=3
```

### Agent Tools

#### Web Search Tool
- Provider: DuckDuckGo (default)
- Fallback: SearX instances
- Use case: Current information, facts, news

#### Terminal Tool
- Security: Command validation enabled
- Confirmation: Required for dangerous commands
- Use case: System administration, file operations

#### RAG Tool
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Supported formats: .md, .txt, .pdf
- Use case: Local knowledge base queries

## Best Practices

### Security

1. **Command Validation**: All terminal commands are validated
2. **User Confirmation**: Dangerous operations require approval
3. **Sandboxing**: Tools run with limited permissions
4. **Logging**: All activities are comprehensively logged

### Performance

1. **Model Selection**: Choose appropriate model for your use case
2. **Resource Monitoring**: Keep track of CPU and memory usage
3. **Caching**: Enable tool result caching when possible
4. **Batch Operations**: Group similar operations together

### Maintenance

1. **Regular Updates**: Keep Ollama and models updated
2. **Log Monitoring**: Review logs for errors and performance issues
3. **Index Refresh**: Periodically refresh the RAG document index
4. **Cleanup**: Remove old logs and cached files

## API Reference

### Agent Interface

The agent exposes these key methods:

- `process_query(query: str)`: Process a single query
- `process_query_stream(query: str)`: Stream processing steps
- `register_tool(tool: ToolInterface)`: Add new tools
- `update_model(model: str)`: Change the active model

### Tool Interface

All tools must implement:

- `execute(query: str, **kwargs)`: Main execution method
- `get_tool_info()`: Return tool metadata
- Tool validation and error handling

## Advanced Features

### Conversation Memory

The agent maintains conversation context:

- Sliding window approach (configurable size)
- Cross-turn reference capability
- Memory usage indicators in UI
- Automatic cleanup of old context

### Tool Orchestration

Multiple tools can be used in sequence:

- Conditional tool selection
- Result chaining between tools
- Error recovery and fallback
- User interaction for confirmations

### Monitoring and Logging

Comprehensive observability:

- Structured JSON logging
- Performance metrics collection
- Request correlation tracking
- Real-time log analysis tools
