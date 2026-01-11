# CLAUDE.md - AI Assistant Guide for Atomic Agents

This document provides a comprehensive guide for AI assistants working with the Atomic Agents codebase. It covers the project structure, architecture, development workflows, and key conventions to follow.

## Table of Contents

- [Project Overview](#project-overview)
- [Codebase Structure](#codebase-structure)
- [Core Concepts and Architecture](#core-concepts-and-architecture)
- [Development Workflows](#development-workflows)
- [Testing Requirements](#testing-requirements)
- [Code Quality Standards](#code-quality-standards)
- [Key Conventions](#key-conventions)
- [Common Patterns](#common-patterns)
- [Important Files and Locations](#important-files-and-locations)
- [Tips for AI Assistants](#tips-for-ai-assistants)

---

## Project Overview

**Atomic Agents** is a Python framework for building modular, predictable, and maintainable AI agent applications. The framework is built on **Instructor** and **Pydantic**, emphasizing:

- **Atomicity**: Single-purpose, composable components
- **Schema-First Design**: All I/O defined via Pydantic schemas
- **Type Safety**: Modern Python generics with type preservation
- **Provider Agnosticism**: Works with any LLM provider (OpenAI, Anthropic, Groq, Ollama, etc.)
- **Modularity**: Components can be easily swapped and reused

**Current Version**: 2.2.2
**Python Requirement**: >=3.12,<4.0
**License**: MIT
**Repository**: https://github.com/BrainBlend-AI/atomic-agents

---

## Codebase Structure

The repository uses a **monorepo structure** with four main components:

```
/home/user/atomic-agents/
├── atomic-agents/          # Core framework library
│   └── atomic_agents/      # Main package
│       ├── agents/         # AtomicAgent implementation
│       ├── base/           # Base classes (schemas, tools, resources, prompts)
│       ├── context/        # Chat history and system prompt generation
│       ├── connectors/     # External integrations (MCP)
│       └── utils/          # Utility functions
├── atomic-assembler/       # CLI tool for managing components
│   └── atomic_assembler/   # TUI application (Textual framework)
├── atomic-examples/        # 11+ complete example projects
│   ├── quickstart/         # Basic examples
│   ├── hooks-example/      # Hook system demonstration
│   ├── mcp-agent/          # MCP integration examples
│   ├── rag-chatbot/        # RAG implementation
│   └── ...                 # Other examples
├── atomic-forge/           # Collection of downloadable tools
│   └── tools/              # Individual tool packages
├── docs/                   # Sphinx documentation
├── guides/                 # Development guides
├── scripts/                # Build and utility scripts
└── .github/workflows/      # CI/CD workflows
```

### Core Module Organization

```
atomic_agents/
├── __init__.py                     # Main package exports
├── agents/
│   └── atomic_agent.py             # AtomicAgent[InputSchema, OutputSchema]
├── base/
│   ├── base_io_schema.py           # BaseIOSchema (all schemas inherit)
│   ├── base_tool.py                # BaseTool[InputSchema, OutputSchema]
│   ├── base_resource.py            # BaseResource[InputSchema, OutputSchema] (MCP)
│   └── base_prompt.py              # BasePrompt[InputSchema, OutputSchema] (MCP)
├── context/
│   ├── chat_history.py             # ChatHistory (conversation management)
│   └── system_prompt_generator.py  # SystemPromptGenerator + context providers
├── connectors/
│   └── mcp/                        # Model Context Protocol integration
│       ├── mcp_factory.py          # Dynamic tool/resource/prompt generation
│       ├── schema_transformer.py   # JSON Schema → Pydantic conversion
│       └── mcp_definition_service.py  # MCP service definitions
└── utils/
    └── format_tool_message.py      # Tool message formatting
```

---

## Core Concepts and Architecture

### 1. Generic Type Parameters

All core classes use **generic type parameters** for type safety:

```python
# Agents
AtomicAgent[InputSchema, OutputSchema]

# Tools
BaseTool[InputSchema, OutputSchema]

# Resources (MCP)
BaseResource[InputSchema, OutputSchema]

# Prompts (MCP)
BasePrompt[InputSchema, OutputSchema]
```

### 2. Base Classes

#### **BaseIOSchema** (`atomic_agents/base/base_io_schema.py`)
- All input/output schemas **must** inherit from `BaseIOSchema`
- **Requires** a docstring describing the schema's purpose
- Automatically generates JSON schema with title and description
- Provides rich console integration for pretty printing

```python
from atomic_agents import BaseIOSchema
from pydantic import Field

class CustomInputSchema(BaseIOSchema):
    """Input schema for custom agent."""  # Docstring required!
    query: str = Field(..., description="User's query")
    context: str = Field(default="", description="Additional context")
```

#### **AtomicAgent** (`atomic_agents/agents/atomic_agent.py`)
Generic agent class with four execution methods:

```python
agent = AtomicAgent[InputSchema, OutputSchema](config)

# Synchronous
result = agent.run(input_data)                    # Complete response
for partial in agent.run_stream(input_data):      # Streaming response
    print(partial)

# Asynchronous
result = await agent.run_async(input_data)        # Complete response
async for partial in agent.run_async_stream(input_data):  # Streaming
    print(partial)
```

**Key Methods**:
- `run()`: Synchronous execution, returns complete response
- `run_stream()`: Synchronous streaming (generator)
- `run_async()`: Async execution, returns complete response
- `run_async_stream()`: Async streaming (async generator)
- `register_context_provider(name, provider)`: Add dynamic context
- `unregister_context_provider(name)`: Remove context provider
- `reset_history()`: Clear conversation history

#### **BaseTool** (`atomic_agents/base/base_tool.py`)
Generic tool class for creating reusable components:

```python
from atomic_agents import BaseTool, BaseToolConfig, BaseIOSchema

class CalculatorTool(BaseTool[CalculatorInput, CalculatorOutput]):
    """Tool for mathematical calculations."""

    def __init__(self, config: BaseToolConfig = BaseToolConfig()):
        super().__init__(config)

    def run(self, params: CalculatorInput) -> CalculatorOutput:
        # Implementation
        return CalculatorOutput(result=...)
```

**Properties** (automatically derived from generic parameters):
- `input_schema`: Input schema class
- `output_schema`: Output schema class
- `tool_name`: Derived from class name
- `tool_description`: Derived from docstring

#### **SystemPromptGenerator** (`atomic_agents/context/system_prompt_generator.py`)
Structured system prompt generation with four sections:

```python
from atomic_agents.context import SystemPromptGenerator

system_prompt = SystemPromptGenerator(
    background=[
        "You are an expert Python developer.",
        "You specialize in clean, maintainable code."
    ],
    steps=[
        "Analyze the user's request.",
        "Design a solution using best practices.",
        "Implement the solution with proper error handling."
    ],
    output_instructions=[
        "Provide clear, well-documented code.",
        "Include type hints for all functions.",
        "Follow PEP 8 style guidelines."
    ],
    # Optional: context_providers for dynamic content
)
```

#### **ChatHistory** (`atomic_agents/context/chat_history.py`)
Turn-based conversation management:

```python
from atomic_agents.context import ChatHistory

history = ChatHistory(max_messages=100)  # Optional limit

# Access messages
messages = history.messages  # List of all messages

# Serialization
history.dump("conversation.json")       # Save to file
history = ChatHistory.load("conversation.json")  # Load from file

# Reset
history.clear()
```

**Features**:
- Turn ID tracking
- Multimodal support (text, images, PDFs, audio)
- Max message limit management
- JSON serialization/deserialization

### 3. Context Providers

Dynamic context injection into system prompts at runtime:

```python
from atomic_agents.context import BaseDynamicContextProvider

class SearchResultsProvider(BaseDynamicContextProvider):
    def __init__(self, title: str, search_results: list[str]):
        super().__init__(title=title)
        self.search_results = search_results

    def get_info(self) -> str:
        return "\n".join(self.search_results)

# Register with agent
provider = SearchResultsProvider("Search Results", ["Result 1", "Result 2"])
agent.register_context_provider("search_results", provider)
```

### 4. MCP (Model Context Protocol) Integration

The framework has extensive MCP support for dynamic tool/resource/prompt generation:

**Key Components**:
- **MCPFactory**: Dynamically creates BaseTool/BaseResource/BasePrompt from MCP servers
- **Transport Types**: STDIO, HTTP_STREAM (default), SSE
- **Schema Transformation**: Automatic JSON Schema → Pydantic conversion

```python
from atomic_agents.connectors.mcp import fetch_mcp_tools_async, MCPTransportType

# Fetch tools from MCP server
tools = await fetch_mcp_tools_async(
    endpoint="http://localhost:3000",
    transport_type=MCPTransportType.HTTP_STREAM
)
```

---

## Development Workflows

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BrainBlend-AI/atomic-agents.git
   cd atomic-agents
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Activate virtual environment**:
   ```bash
   poetry shell
   ```

### Code Quality Workflow

**ALWAYS** run these commands before committing:

1. **Format code** (Black):
   ```bash
   poetry run black atomic-agents atomic-assembler atomic-examples atomic-forge
   ```
   - Line length: **127 characters**
   - Configuration in `/home/user/atomic-agents/pyproject.toml`

2. **Lint code** (Flake8):
   ```bash
   poetry run flake8 --extend-exclude=.venv atomic-agents atomic-assembler atomic-examples atomic-forge
   ```
   - Max line length: **150 characters**
   - Max complexity: **10**
   - Ignores: E203, W293, W503
   - Configuration in `/home/user/atomic-agents/.flake8`

3. **Run tests**:
   ```bash
   poetry run pytest --cov=atomic_agents atomic-agents
   ```
   - Minimum coverage: **100% for new functionality**
   - Configuration in `/home/user/atomic-agents/pytest.ini`

### Git Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```
   - Follow conventional commit format
   - Write clear, descriptive messages

3. **Push to fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**:
   - Describe changes clearly
   - Reference related issues
   - Include test results
   - Update documentation if needed

---

## Testing Requirements

### Test Structure

```
atomic-agents/tests/
├── agents/
│   └── test_atomic_agent.py
├── base/
│   └── test_base_tool.py
├── connectors/mcp/
│   ├── test_mcp_factory.py
│   ├── test_schema_transformer.py
│   └── test_mcp_definition_service.py
├── context/
│   ├── test_chat_history.py
│   └── test_system_prompt_generator.py
└── utils/
    └── test_format_tool_message.py
```

### Testing Conventions

- **Framework**: pytest with pytest-cov and pytest-asyncio
- **Test files**: Must follow `test_*.py` pattern
- **Test functions**: Must start with `test_`
- **Test classes**: Must start with `Test`
- **Coverage requirement**: **100% for new functionality**
- **Async support**: Full support for async tests

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest --cov=atomic_agents atomic-agents

# Run specific test file
poetry run pytest atomic-agents/tests/agents/test_atomic_agent.py

# Generate HTML coverage report
poetry run pytest --cov=atomic_agents --cov-report=html atomic-agents
```

### Writing Tests

```python
import pytest
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from pydantic import Field

class TestInputSchema(BaseIOSchema):
    """Test input schema."""
    message: str = Field(..., description="Test message")

class TestOutputSchema(BaseIOSchema):
    """Test output schema."""
    response: str = Field(..., description="Test response")

def test_agent_run(mock_instructor, mock_history):
    """Test agent execution."""
    agent = AtomicAgent[TestInputSchema, TestOutputSchema](
        AgentConfig(
            client=mock_instructor,
            model="gpt-4o-mini",
            history=mock_history
        )
    )

    input_data = TestInputSchema(message="Hello")
    result = agent.run(input_data)

    assert isinstance(result, TestOutputSchema)
    assert result.response is not None

@pytest.mark.asyncio
async def test_agent_run_async(mock_async_instructor, mock_history):
    """Test async agent execution."""
    agent = AtomicAgent[TestInputSchema, TestOutputSchema](
        AgentConfig(
            client=mock_async_instructor,
            model="gpt-4o-mini",
            history=mock_history
        )
    )

    input_data = TestInputSchema(message="Hello")
    result = await agent.run_async(input_data)

    assert isinstance(result, TestOutputSchema)
```

---

## Code Quality Standards

### Python Style Guidelines

- **Follow PEP 8**: Standard Python style guide
- **Use type hints**: All function signatures must have type hints
- **Write docstrings**: Use Google-style docstrings for all public modules, functions, classes, and methods
- **Keep functions focused**: Single responsibility principle
- **Use meaningful names**: Clear, descriptive variable and function names

### Code Formatting (Black)

```python
# Configuration in pyproject.toml
[tool.black]
line-length = 127
```

**Apply formatting**:
```bash
poetry run black atomic-agents atomic-assembler atomic-examples atomic-forge
```

### Linting (Flake8)

```ini
# Configuration in .flake8
[flake8]
max-line-length = 150
max-complexity = 10
ignore = E203, W293, W503
```

**Run linter**:
```bash
poetry run flake8 --extend-exclude=.venv atomic-agents atomic-assembler atomic-examples atomic-forge
```

### Docstring Format (Google Style)

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.

    Longer description if needed, explaining the function's purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Examples:
        >>> example_function("test", 5)
        True
    """
    pass
```

---

## Key Conventions

### 1. Schema Design

**All schemas MUST**:
- Inherit from `BaseIOSchema`
- Include a descriptive docstring
- Use `Field()` with descriptions for all fields
- Follow the naming pattern: `{Component}{Input|Output}Schema`

```python
from atomic_agents import BaseIOSchema
from pydantic import Field

class WeatherQueryInputSchema(BaseIOSchema):
    """Input schema for weather query agent."""
    city: str = Field(..., description="Name of the city")
    units: str = Field(default="metric", description="Temperature units")

class WeatherQueryOutputSchema(BaseIOSchema):
    """Output schema for weather query agent."""
    temperature: float = Field(..., description="Current temperature")
    conditions: str = Field(..., description="Weather conditions")
```

### 2. Tool Creation

**Tool structure**:
```python
from atomic_agents import BaseTool, BaseToolConfig, BaseIOSchema
from pydantic import Field

class ToolNameInputSchema(BaseIOSchema):
    """Input schema for ToolName."""
    parameter: str = Field(..., description="Parameter description")

class ToolNameOutputSchema(BaseIOSchema):
    """Output schema for ToolName."""
    result: str = Field(..., description="Result description")

class ToolName(BaseTool[ToolNameInputSchema, ToolNameOutputSchema]):
    """Brief description of what the tool does.

    Longer description with usage details, examples, and any
    important notes about configuration or behavior.
    """

    def __init__(self, config: BaseToolConfig = BaseToolConfig()):
        super().__init__(config)
        # Initialize any tool-specific state

    def run(self, params: ToolNameInputSchema) -> ToolNameOutputSchema:
        """Execute the tool's main functionality.

        Args:
            params: Input parameters

        Returns:
            Tool execution results
        """
        # Implementation
        return ToolNameOutputSchema(result="...")
```

### 3. Agent Creation

**Agent structure**:
```python
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator, ChatHistory
from pydantic import Field
import instructor
from openai import OpenAI

# Define schemas
class AgentInputSchema(BaseIOSchema):
    """Input schema for the agent."""
    query: str = Field(..., description="User query")

class AgentOutputSchema(BaseIOSchema):
    """Output schema for the agent."""
    response: str = Field(..., description="Agent response")

# Create system prompt
system_prompt = SystemPromptGenerator(
    background=["Background information"],
    steps=["Processing steps"],
    output_instructions=["Output formatting rules"]
)

# Initialize client and agent
client = instructor.from_openai(OpenAI())
agent = AtomicAgent[AgentInputSchema, AgentOutputSchema](
    AgentConfig(
        client=client,
        model="gpt-4o-mini",
        system_prompt_generator=system_prompt,
        history=ChatHistory(),
        model_api_parameters={
            "temperature": 0.7,
            "max_tokens": 1000
        }
    )
)
```

### 4. Schema Chaining Pattern

Match output schema to tool input schema for seamless composition:

```python
from web_search_agent.tools.searxng_search import SearXNGSearchTool

# Agent output matches tool input
query_agent = AtomicAgent[UserInput, SearXNGSearchTool.input_schema](config)

# Can directly pass agent output to tool
query_result = query_agent.run(user_input)
search_result = search_tool.run(query_result)
```

### 5. Orchestrator Pattern

Use Union types for multi-tool selection:

```python
from typing import Union
from pydantic import Field

class OrchestratorOutputSchema(BaseIOSchema):
    """Output schema for orchestrator agent."""
    tool_parameters: Union[SearchToolInput, CalculatorToolInput] = Field(
        ..., description="Parameters for the selected tool"
    )

orchestrator = AtomicAgent[UserInput, OrchestratorOutputSchema](config)
result = orchestrator.run(user_input)

# Route to appropriate tool based on type
if isinstance(result.tool_parameters, SearchToolInput):
    output = search_tool.run(result.tool_parameters)
elif isinstance(result.tool_parameters, CalculatorToolInput):
    output = calculator_tool.run(result.tool_parameters)
```

---

## Common Patterns

### 1. Multimodal Input

```python
from atomic_agents.context import ChatHistory

history = ChatHistory()

# Add image to history
import instructor

input_with_image = instructor.Image.from_path("path/to/image.png")
history.add_user_turn(input_with_image)

# Agent will receive image in context
response = agent.run(input_data)
```

### 2. Context Provider for Search Results

```python
from atomic_agents.context import BaseDynamicContextProvider

class SearchResultsProvider(BaseDynamicContextProvider):
    def __init__(self, title: str, results: list[dict]):
        super().__init__(title=title)
        self.results = results

    def get_info(self) -> str:
        formatted = []
        for i, result in enumerate(self.results, 1):
            formatted.append(f"{i}. {result['title']}\n   {result['url']}\n   {result['snippet']}")
        return "\n\n".join(formatted)

# Register with agent
provider = SearchResultsProvider("Web Search Results", search_results)
agent.register_context_provider("search_results", provider)
```

### 3. Hook System

```python
from atomic_agents import AtomicAgent, AgentConfig

def on_error_hook(error: Exception, agent: AtomicAgent, *args, **kwargs):
    """Custom error handler."""
    print(f"Error occurred: {error}")
    # Log, retry, or handle error

def on_success_hook(result, agent: AtomicAgent, *args, **kwargs):
    """Custom success handler."""
    print(f"Agent completed successfully: {result}")

agent = AtomicAgent[InputSchema, OutputSchema](config)
agent.config.hooks = {
    "on_error": on_error_hook,
    "on_success": on_success_hook
}
```

### 4. Streaming Response Display

```python
# Synchronous streaming
for partial in agent.run_stream(input_data):
    print(partial.response, end='', flush=True)
print()  # Newline after complete

# Asynchronous streaming
async for partial in agent.run_async_stream(input_data):
    print(partial.response, end='', flush=True)
print()
```

---

## Important Files and Locations

### Core Framework Files
- `/home/user/atomic-agents/atomic-agents/atomic_agents/agents/atomic_agent.py` - Main agent implementation
- `/home/user/atomic-agents/atomic-agents/atomic_agents/base/base_io_schema.py` - Base schema class
- `/home/user/atomic-agents/atomic-agents/atomic_agents/base/base_tool.py` - Base tool class
- `/home/user/atomic-agents/atomic-agents/atomic_agents/context/system_prompt_generator.py` - Prompt generation
- `/home/user/atomic-agents/atomic-agents/atomic_agents/context/chat_history.py` - Conversation management

### Configuration Files
- `/home/user/atomic-agents/pyproject.toml` - Project dependencies and build config
- `/home/user/atomic-agents/.flake8` - Linting configuration
- `/home/user/atomic-agents/pytest.ini` - Test configuration

### Documentation
- `/home/user/atomic-agents/README.md` - Main project documentation
- `/home/user/atomic-agents/UPGRADE_DOC.md` - v2.0 upgrade guide
- `/home/user/atomic-agents/docs/contributing.md` - Contributing guidelines
- `/home/user/atomic-agents/guides/DEV_GUIDE.md` - Developer guide
- `/home/user/atomic-agents/docs/` - Sphinx documentation source

### CI/CD
- `/home/user/atomic-agents/.github/workflows/code-quality.yml` - Code quality checks
- `/home/user/atomic-agents/.github/workflows/docs.yml` - Documentation deployment

### Examples
- `/home/user/atomic-agents/atomic-examples/quickstart/` - Basic examples
- `/home/user/atomic-agents/atomic-examples/hooks-example/` - Hook system examples
- `/home/user/atomic-agents/atomic-examples/mcp-agent/` - MCP integration examples

### Tools
- `/home/user/atomic-agents/atomic-forge/tools/` - Individual tool implementations

---

## Tips for AI Assistants

### When Working on This Codebase

1. **Always use generic type parameters**:
   - `AtomicAgent[InputSchema, OutputSchema]`
   - `BaseTool[InputSchema, OutputSchema]`
   - Never use schemas as class attributes for tools/agents

2. **Schema requirements**:
   - ALL schemas must inherit from `BaseIOSchema`
   - ALL schemas must have a docstring
   - ALL fields must use `Field()` with descriptions

3. **Import paths** (v2.0+):
   - Use `from atomic_agents import ...` for base classes
   - Use `from atomic_agents.context import ...` for context components
   - Use `from atomic_agents.connectors.mcp import ...` for MCP
   - NEVER use `.lib` in import paths (that's v1.x)

4. **Testing is mandatory**:
   - 100% coverage requirement for new functionality
   - Test both sync and async methods
   - Use fixtures for mocking
   - Test edge cases and error conditions

5. **Code quality before commits**:
   - Run Black: `poetry run black atomic-agents atomic-assembler atomic-examples atomic-forge`
   - Run Flake8: `poetry run flake8 --extend-exclude=.venv atomic-agents atomic-assembler atomic-examples atomic-forge`
   - Run tests: `poetry run pytest --cov=atomic_agents atomic-agents`

6. **Documentation updates**:
   - Update docstrings when changing functionality
   - Update README.md for significant changes
   - Add examples for new features

7. **Naming conventions**:
   - Classes: PascalCase (e.g., `AtomicAgent`, `BaseTool`)
   - Functions/methods: snake_case (e.g., `run_async`, `get_info`)
   - Schemas: `{Component}{Input|Output}Schema` (e.g., `WeatherQueryInputSchema`)
   - Tools: `{Name}Tool` (e.g., `CalculatorTool`)

8. **Version compatibility**:
   - Python >=3.12 required
   - This is v2.0+ codebase (breaking changes from v1.x)
   - Check UPGRADE_DOC.md for migration details

9. **Common mistakes to avoid**:
   - Don't use `memory` (use `history`)
   - Don't use `BaseAgent` (use `AtomicAgent`)
   - Don't use `BaseAgentConfig` (use `AgentConfig`)
   - Don't put schemas in config (use type parameters)
   - Don't forget docstrings on schemas
   - Don't skip type hints

10. **When creating new tools**:
    - Follow the tool template in `/home/user/atomic-agents/atomic-forge/tools/`
    - Include comprehensive tests
    - Add README.md with usage examples
    - Create both input and output schemas
    - Document all dependencies

### Understanding the Architecture

The framework follows a **layered architecture**:

1. **Base Layer** (`base/`): Core abstractions (schemas, tools, resources, prompts)
2. **Agent Layer** (`agents/`): Agent implementation with execution logic
3. **Context Layer** (`context/`): Conversation management and prompt generation
4. **Connector Layer** (`connectors/`): External integrations (MCP)
5. **Utility Layer** (`utils/`): Helper functions

**Data flow**:
```
User Input → InputSchema → Agent → LLM → OutputSchema → User Output
                ↑                    ↑
          System Prompt        Tools/Resources
          (with context)
```

### Quick Reference Commands

```bash
# Setup
poetry install
poetry shell

# Code Quality
poetry run black atomic-agents atomic-assembler atomic-examples atomic-forge
poetry run flake8 --extend-exclude=.venv atomic-agents atomic-assembler atomic-examples atomic-forge
poetry run pytest --cov=atomic_agents atomic-agents

# Run CLI
poetry run atomic

# Build documentation
cd docs && poetry run sphinx-build -b html . _build/html

# Run specific example
cd atomic-examples/quickstart && poetry run python quickstart/1_0_basic_chatbot.py
```

### Resources for Learning

- **Main Docs**: https://brainblend-ai.github.io/atomic-agents/
- **Examples**: `/home/user/atomic-agents/atomic-examples/`
- **Contributing Guide**: `/home/user/atomic-agents/docs/contributing.md`
- **Developer Guide**: `/home/user/atomic-agents/guides/DEV_GUIDE.md`
- **Upgrade Guide**: `/home/user/atomic-agents/UPGRADE_DOC.md`
- **Medium Article**: https://ai.gopubby.com/want-to-build-ai-agents-c83ab4535411
- **Discord**: https://discord.gg/J3W9b5AZJR
- **Subreddit**: https://www.reddit.com/r/AtomicAgents/

---

## Summary

Atomic Agents is a production-ready framework for building modular AI applications with the same software engineering principles used in traditional development. When working with this codebase:

- **Follow the schema-first approach** with strict type safety
- **Use generic type parameters** for all agents and tools
- **Maintain 100% test coverage** for new functionality
- **Run code quality checks** before every commit
- **Document everything** with clear docstrings and examples
- **Follow established patterns** for consistency and maintainability

This ensures the codebase remains clean, predictable, and easy to maintain as it grows.

---

**Last Updated**: 2026-01-11
**Framework Version**: 2.2.2
**Python Requirement**: >=3.12,<4.0
