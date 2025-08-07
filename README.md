# 🤖 Streamlit Ollama Chatbot

A minimalist, high-end chatbot interface built on Streamlit and Ollama, inspired by Dieter Rams' design principles and Apple's user experience philosophy.

## ✨ Features

- **Elegant Interface**: Clean, Apple-inspired design with intuitive navigation
- **Real-time Metrics**: Performance metrics displayed with each response (latency, token count)
- **Model Flexibility**: Easy model switching and parameter tuning via sidebar
- **Session Management**: Save, export, and manage chat sessions
- **Responsive Design**: Works seamlessly on desktop and mobile devices

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

```
agent-aiops/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── config/
│   ├── __init__.py
│   └── settings.py       # Configuration settings
├── services/
│   ├── __init__.py
│   └── ollama_service.py # Ollama API integration
└── utils/
    ├── __init__.py
    └── chat_utils.py     # Chat utility functions
```

## ⚙️ Configuration

### Default Settings

- **Model**: `llama3.2:3b`
- **Temperature**: `0.7`
- **Max Tokens**: `1000`
- **Top-p**: `0.9`

### Environment Variables

- `OLLAMA_BASE_URL`: Ollama API URL (default: `http://localhost:11434`)

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

### Adding New Models

1. Pull the model with Ollama:
   ```bash
   ollama pull <model-name>
   ```

2. The model will automatically appear in the sidebar selection

### Customizing Parameters

All model parameters can be adjusted in real-time via the sidebar:

- **Temperature**: Controls response randomness (0.0 - 2.0)
- **Max Tokens**: Maximum response length (100 - 4000)
- **Top-p**: Nucleus sampling parameter (0.1 - 1.0)

### Exporting Chat History

Click the "Export" button in the sidebar to download your chat history as a text file.

## 🐛 Troubleshooting

### Common Issues

1. **"Ollama service is not running"**
   - Start Ollama: `ollama serve`
   - Check if port 11434 is available

2. **"No models available"**
   - Install a model: `ollama pull llama3.2:3b`
   - Verify installation: `ollama list`

3. **Slow responses**
   - Check your hardware resources
   - Try a smaller model
   - Adjust max_tokens parameter

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support and questions, please open an issue in the repository.
