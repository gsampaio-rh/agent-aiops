# Agent-AIOps Refactoring Plan

## Overview

This document outlines a comprehensive refactoring plan for the Agent-AIOps project to improve code organization, maintainability, and architectural quality while following software engineering best practices.

## Current State Analysis

### Strengths
✅ **Modular Structure**: Good separation of concerns with distinct `services/`, `ui/`, `utils/`, and `config/` directories  
✅ **Clean Main Entry Point**: Minimal `app.py` with clear orchestration (82 lines)  
✅ **Comprehensive Logging**: Enterprise-grade logging system with correlation tracking  
✅ **Configuration Management**: Centralized settings with environment variable support  
✅ **UI Components**: Well-organized Streamlit components following design principles  

### Issues Identified
❌ **Missing Core Architecture**: Missing `core/` directory with interfaces, models, and exceptions  
❌ **Import Dependencies**: Services import from missing `core.interfaces` and `core.models`  
❌ **Inconsistent Constants**: Constants spread across `config/settings.py` and `config/constants.py`  
❌ **Missing Abstractions**: No base classes or interfaces for extensibility  
❌ **Error Handling**: Exception classes referenced but not defined  
❌ **Testing Structure**: No visible test directory or testing framework setup  

## Refactoring Goals

1. **Fix Missing Dependencies**: Create missing core components
2. **Establish Clear Architecture**: Implement proper layered architecture
3. **Improve Modularity**: Better separation of concerns and dependency injection
4. **Enhance Maintainability**: Consistent patterns and error handling
5. **Add Testing Infrastructure**: Comprehensive testing setup
6. **Documentation**: Clear architectural documentation

## Proposed Architecture

```
agent-aiops/
├── core/                           # 🏗️ Core architecture components
│   ├── __init__.py
│   ├── interfaces/                 # 📋 Abstract interfaces
│   │   ├── __init__.py
│   │   ├── llm_service.py         # LLM service interface
│   │   ├── agent_service.py       # Agent service interface  
│   │   └── search_service.py      # Search service interface
│   ├── models/                     # 📊 Data models and DTOs
│   │   ├── __init__.py
│   │   ├── agent.py               # Agent-related models
│   │   ├── search.py              # Search models
│   │   └── chat.py                # Chat models
│   ├── exceptions/                 # 🚨 Custom exceptions
│   │   ├── __init__.py
│   │   ├── base.py                # Base exception classes
│   │   ├── service_exceptions.py  # Service-specific exceptions
│   │   └── agent_exceptions.py    # Agent-specific exceptions
│   └── validators/                 # ✅ Input validation
│       ├── __init__.py
│       ├── query_validator.py     # Query validation
│       └── config_validator.py    # Configuration validation
├── services/                       # 🔧 Business logic services
│   ├── __init__.py
│   ├── base_service.py            # Base service class
│   ├── ollama_service.py          # ✅ LLM service implementation
│   ├── agent_service.py           # ✅ Agent service implementation
│   └── search_service.py          # ✅ Search service implementation
├── config/                         # ⚙️ Configuration management
│   ├── __init__.py
│   ├── settings.py                # ✅ Environment-based settings
│   ├── constants.py               # ✅ Application constants
│   └── dependencies.py           # 🔄 Dependency injection setup
├── ui/                            # 🎨 User interface components
│   ├── __init__.py
│   ├── components.py              # ✅ Reusable UI components
│   ├── styles.py                  # ✅ CSS styling
│   └── streamlit_utils.py         # ✅ Streamlit utilities
├── utils/                         # 🛠️ Utility functions
│   ├── __init__.py
│   ├── logger.py                  # ✅ Logging utilities
│   ├── chat_utils.py              # ✅ Chat utilities
│   └── log_analyzer.py            # ✅ Log analysis
├── tests/                         # 🧪 Test suite
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── unit/                      # Unit tests
│   │   ├── test_services/
│   │   ├── test_core/
│   │   └── test_utils/
│   ├── integration/               # Integration tests
│   │   ├── test_api_integration/
│   │   └── test_agent_workflows/
│   └── fixtures/                  # Test data and fixtures
├── docs/                          # 📚 Documentation
│   ├── architecture.md           # Architecture overview
│   ├── api_reference.md          # API documentation
│   └── development_guide.md      # Development guidelines
├── scripts/                       # 🔧 Development scripts
│   ├── setup_dev.py              # Development environment setup
│   ├── run_tests.py              # Test runner
│   └── format_code.py            # Code formatting
├── app.py                         # ✅ Main application entry
├── run.py                         # ✅ Development server
├── requirements.txt               # ✅ Dependencies
├── requirements-dev.txt           # 🆕 Development dependencies
├── pyproject.toml                 # 🆕 Project configuration
└── README.md                      # ✅ Project documentation
```

## Implementation Plan

### Phase 1: Core Foundation (High Priority)

#### 1.1 Create Core Interfaces
```python
# core/interfaces/llm_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator

class LLMServiceInterface(ABC):
    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass
    
    @abstractmethod
    def chat_stream(self, model: str, messages: List[Dict], **kwargs) -> Iterator[Dict]:
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        pass
    
    @abstractmethod
    def validate_parameters(self, temperature: float, max_tokens: int, top_p: float) -> None:
        pass
```

#### 1.2 Create Core Models
```python
# core/models/agent.py
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import time

class StepType(Enum):
    THOUGHT = "thought"
    TOOL_SELECTION = "tool_selection"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"

@dataclass
class AgentStep:
    step_type: StepType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 1.3 Create Exception Hierarchy
```python
# core/exceptions/base.py
from enum import Enum
from typing import Optional, Dict, Any

class ErrorCodes(Enum):
    # Service errors
    OLLAMA_CONNECTION_FAILED = "OLLAMA_001"
    OLLAMA_REQUEST_FAILED = "OLLAMA_002"
    OLLAMA_MODEL_NOT_FOUND = "OLLAMA_003"
    
    # Agent errors
    AGENT_INIT_FAILED = "AGENT_001"
    AGENT_PROCESSING_FAILED = "AGENT_002"
    
    # Search errors
    SEARCH_PROVIDER_FAILED = "SEARCH_001"
    SEARCH_TIMEOUT = "SEARCH_002"
    
    # Validation errors
    INVALID_QUERY = "VALIDATION_001"
    INVALID_PARAMETERS = "VALIDATION_002"

class AgentAIOpsException(Exception):
    """Base exception for Agent-AIOps application."""
    
    def __init__(self, message: str, error_code: Optional[ErrorCodes] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = time.time()
```

### Phase 2: Service Layer Improvements (Medium Priority)

#### 2.1 Create Base Service Class
```python
# services/base_service.py
from abc import ABC
from typing import Any, Dict
from utils.logger import get_logger

class BaseService(ABC):
    """Base class for all services with common functionality."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def validate_query(self, query: str) -> None:
        """Validate query input."""
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        if len(query) > 10000:  # 10KB limit
            raise ValidationError("Query too long")
```

#### 2.2 Implement Dependency Injection
```python
# config/dependencies.py
from functools import lru_cache
from services.ollama_service import OllamaService
from services.search_service import WebSearchService
from services.agent_service import ReactAgent

class ServiceContainer:
    """Service container for dependency injection."""
    
    def __init__(self):
        self._services = {}
    
    @lru_cache()
    def get_ollama_service(self) -> OllamaService:
        if 'ollama' not in self._services:
            self._services['ollama'] = OllamaService()
        return self._services['ollama']
    
    @lru_cache()
    def get_search_service(self) -> WebSearchService:
        if 'search' not in self._services:
            self._services['search'] = WebSearchService()
        return self._services['search']
    
    @lru_cache()
    def get_agent_service(self) -> ReactAgent:
        if 'agent' not in self._services:
            self._services['agent'] = ReactAgent(
                llm_service=self.get_ollama_service(),
                search_service=self.get_search_service()
            )
        return self._services['agent']

# Global service container
container = ServiceContainer()
```

### Phase 3: Testing Infrastructure (Medium Priority)

#### 3.1 Setup Testing Framework
```python
# tests/conftest.py
import pytest
from unittest.mock import Mock
from services.ollama_service import OllamaService
from services.search_service import WebSearchService

@pytest.fixture
def mock_ollama_service():
    service = Mock(spec=OllamaService)
    service.get_available_models.return_value = ["llama3.2:3b", "codellama:7b"]
    service.health_check.return_value = True
    return service

@pytest.fixture
def mock_search_service():
    service = Mock(spec=WebSearchService)
    return service
```

#### 3.2 Add Development Dependencies
```txt
# requirements-dev.txt
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-mock>=3.11.0
pytest-cov>=4.1.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.3.0
```

#### 3.3 Project Configuration
```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-aiops"
version = "1.0.0"
description = "AI Agent for AIOps with Ollama and Streamlit"
requires-python = ">=3.8"

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=. --cov-report=html --cov-report=term-missing"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Phase 4: Configuration Consolidation (Low Priority)

#### 4.1 Merge Configuration Files
Consolidate settings from both `config/settings.py` and `config/constants.py` into a single, well-organized configuration system.

#### 4.2 Environment-Specific Configuration
```python
# config/environments.py
from enum import Enum
from dataclasses import dataclass

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

@dataclass
class EnvironmentConfig:
    debug: bool
    log_level: str
    enable_metrics: bool
    enable_agent_mode: bool
```

### Phase 5: Documentation and Development Tools (Low Priority)

#### 5.1 Architecture Documentation
- Create comprehensive architecture diagrams
- Document design patterns used
- API reference documentation

#### 5.2 Development Scripts
```python
# scripts/setup_dev.py
import subprocess
import sys

def setup_development_environment():
    """Setup development environment with pre-commit hooks."""
    commands = [
        "pip install -r requirements-dev.txt",
        "pre-commit install",
        "pre-commit run --all-files"
    ]
    
    for cmd in commands:
        subprocess.run(cmd.split(), check=True)
```

## Benefits of This Refactoring

### 🔧 **Technical Benefits**
- **Proper Dependency Management**: Clear interfaces and dependency injection
- **Error Handling**: Comprehensive exception hierarchy with error codes
- **Testing**: Full test coverage with mocks and fixtures  
- **Type Safety**: Better type hints and validation
- **Code Quality**: Consistent formatting and linting

### 🚀 **Architectural Benefits**
- **Modularity**: Clear separation of concerns
- **Extensibility**: Easy to add new services and features
- **Maintainability**: Well-organized codebase with clear patterns
- **Scalability**: Clean architecture that can grow with requirements

### 👥 **Development Benefits**
- **Developer Experience**: Clear structure and documentation
- **Onboarding**: New developers can understand the codebase quickly
- **Debugging**: Better error messages and logging
- **Collaboration**: Consistent code style and patterns

## Implementation Priority

### 🔴 **Critical (Do First)**
1. Create missing core interfaces and models
2. Fix import dependencies  
3. Implement exception hierarchy
4. Add basic input validation

### 🟡 **Important (Do Second)**  
1. Setup testing infrastructure
2. Add development dependencies
3. Create base service classes
4. Implement dependency injection

### 🟢 **Enhancement (Do Third)**
1. Consolidate configuration
2. Add comprehensive documentation  
3. Setup development scripts
4. Add pre-commit hooks

## Migration Strategy

### Step 1: Backward Compatibility
- Keep existing functionality working
- Add new components alongside old ones
- Gradual migration without breaking changes

### Step 2: Progressive Enhancement
- Replace old patterns incrementally
- Update one module at a time
- Maintain test coverage throughout

### Step 3: Cleanup
- Remove deprecated code
- Consolidate duplicate functionality
- Final documentation update

## Conclusion

This refactoring plan addresses the current architectural gaps while maintaining the project's strengths. The modular approach ensures that improvements can be implemented incrementally without disrupting the existing functionality.

The focus on clean architecture, proper error handling, and comprehensive testing will make the codebase more maintainable and extensible for future development.

---

**Next Steps**: Start with Phase 1 (Core Foundation) to fix the immediate import issues and establish a solid architectural foundation.
