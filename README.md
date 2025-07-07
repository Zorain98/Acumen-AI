# 🤖 Acumen AI

> **Intelligent Conversational AI Platform powered by MCP Agents and Groq**

A sophisticated conversational AI web application built with Streamlit that leverages Model Context Protocol (MCP) agents and Groq's high-performance inference engine to deliver intelligent, context-aware interactions through a modern chat interface.

## 🚀 Features

### Core Capabilities
- **Natural Language Processing**: Powered by Llama-3.3-70b-versatile model via Groq API
- **MCP Agent Integration**: Advanced reasoning with Model Context Protocol agents
- **Persistent Memory**: Conversation history with memory management
- **Real-time Chat Interface**: Responsive web-based chat experience
- **Asynchronous Processing**: Non-blocking operations for optimal performance

### Advanced Features
- **Function Calling**: Enhanced tool usage with proper error handling
- **Debug Mode**: Comprehensive logging and error tracking
- **Session Management**: Automatic cleanup and resource management
- **Error Recovery**: Intelligent error handling with user-friendly messages
- **Rate Limit Handling**: Graceful degradation with retry mechanisms

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   MCP Agent     │────│   Groq API      │
│   Frontend      │    │   Processing    │    │   (Llama-3.3)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│  Configuration  │──────────────┘
                        │  Management     │
                        └─────────────────┘
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Web Interface)
- **AI/ML**: Groq API, Llama-3.3-70b-versatile
- **Agent Framework**: MCP (Model Context Protocol)
- **Language**: Python 3.8+
- **Async Processing**: asyncio
- **Environment Management**: python-dotenv

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key
- MCP configuration file (`browser_mcp.json`)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/acumen-ai.git
   cd acumen-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

4. **Configure MCP Agent**
   - Ensure `browser_mcp.json` configuration file is present
   - Modify settings as needed for your use case

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Getting Started

1. **Initialize Agent**: Click "Initialize Agent" in the sidebar
2. **Start Chatting**: Type your message in the chat input
3. **Explore Features**: Toggle debug mode for detailed logging
4. **Manage Memory**: Use "Clear Memory" to reset conversation history

### Configuration Options

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Model | llama-3.3-70b-versatile | Groq model for AI processing |
| Max Steps | 10 | Maximum reasoning steps for agent |
| Temperature | 0.1 | Response randomness control |
| Max Tokens | 4096 | Maximum response length |
| Timeout | 30s | Request timeout limit |
| Max Retries | 2 | Error retry attempts |

## 💻 Code Structure

```
acumen-ai/
├── app.py                 # Main Streamlit application
├── mcp_use.py            # MCP agent implementation
├── browser_mcp.json      # MCP configuration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── README.md            # This file
```

## 🔧 Key Components

### Agent Initialization
```python
async def initialize_agent():
    """Initialize the MCP agent and client"""
    # Loads environment variables
    # Configures Groq API connection
    # Sets up MCP client and agent
    # Returns initialized components
```

### Response Processing
```python
async def get_agent_response(user_input, agent):
    """Get response from the agent with enhanced error handling"""
    # Validates user input
    # Enhances prompts for better responses
    # Handles errors gracefully
    # Returns formatted response
```

### Session Management
- Automatic state persistence
- Memory management
- Resource cleanup
- Error state tracking

## 🎯 Use Cases

- **Customer Support**: Intelligent chatbot for customer queries
- **Research Assistant**: Context-aware research and analysis
- **Content Creation**: AI-powered writing and editing assistance
- **Data Analysis**: Natural language data querying and insights
- **Educational Tool**: Interactive learning and tutoring

## 🛡️ Error Handling

The application includes comprehensive error handling for:

- **API Rate Limits**: Graceful degradation with user notifications
- **Function Call Errors**: Intelligent retry mechanisms
- **Timeout Issues**: Request optimization and user guidance
- **Configuration Errors**: Clear error messages and troubleshooting
- **Memory Issues**: Automatic cleanup and resource management

## 🔍 Debug Mode

Enable debug mode for:
- Verbose logging of all operations
- Detailed error tracking and tracebacks
- Real-time performance monitoring
- Configuration validation
- Session state inspection

## 🔐 Security Considerations

- Environment variable protection for API keys
- Session isolation and cleanup
- Input validation and sanitization
- Error message sanitization
- Resource usage monitoring

## 📊 Performance Optimization

- **Asynchronous Processing**: Non-blocking operations
- **Memory Management**: Automatic cleanup protocols
- **Request Optimization**: 30-second timeout with retry logic
- **Response Caching**: Session-based conversation history
- **Resource Monitoring**: Automatic session cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for high-performance AI inference
- [Streamlit](https://streamlit.io/) for the web framework
- [MCP Protocol](https://modelcontextprotocol.io/) for agent architecture
- [Llama](https://llama.meta.com/) for the foundational language model

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

**Built with ❤️ using Streamlit • Powered by MCP Agent & Groq**
