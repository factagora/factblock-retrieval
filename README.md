# GraphRAG Retrieval System

A production-ready retrieval system for graph-based retrieval augmented generation, designed for compliance domain applications. This system enables intelligent search and retrieval of regulatory documents, guidance, enforcement actions, and compliance topics using Neo4j graph database.

## Features

- **🔍 Graph-based retrieval**: Advanced search using Neo4j graph relationships
- **🏗️ Modular architecture**: Extensible design supporting multiple retrieval strategies
- **⚖️ Compliance domain focus**: Pre-built models for regulations, guidance, and enforcement
- **🚀 Production ready**: Comprehensive testing, configuration management, and error handling
- **📊 Rich relationships**: Connects related documents through meaningful graph relationships
- **🔧 Easy integration**: Clean API interface for seamless integration into existing systems

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd factblock-retrieval

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from src.retrieval import RetrievalModule
from src.config import load_config

# Load configuration (supports environment variables)
config = load_config()

# Initialize retrieval module
module = RetrievalModule('graphrag')
module.initialize(config.to_dict())

# Perform search
results = module.retrieve(
    query_text="GDPR data protection requirements",
    filters={'category': 'data_privacy'},
    limit=10
)

# Process results
for result in results:
    print(f"Score: {result.score:.2f}")
    print(f"Source: {result.source_type}")
    print(f"Content: {result.content[:200]}...")
    print("---")
```

### Docker Setup (Neo4j)

```bash
# Start Neo4j database
docker-compose up -d

# Load sample data
python examples/example_data_loader.py
```

## Project Structure

```
factblock-retrieval/
├── src/
│   ├── __init__.py
│   ├── retrieval/          # Retrieval algorithms and interfaces
│   │   ├── __init__.py     # RetrievalModule main API
│   │   ├── base.py         # Abstract base classes
│   │   └── graph_rag.py    # GraphRAG implementation
│   ├── database/           # Database connections
│   │   ├── __init__.py
│   │   └── neo4j_client.py # Neo4j client with pooling
│   ├── models/             # Pydantic data models
│   │   ├── __init__.py
│   │   └── data_models.py  # Compliance domain models
│   └── config.py           # Configuration management
├── examples/               # Usage examples and data loading
│   ├── __init__.py
│   └── example_data_loader.py
├── tests/                  # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py        # Test fixtures and configuration
│   ├── test_config.py     # Configuration tests
│   ├── test_models.py     # Data model tests
│   ├── test_retrieval.py  # Retrieval system tests
│   ├── test_data_loader.py # Data loader tests
│   ├── test_neo4j_client.py # Database client tests
│   └── test_integration.py # Integration tests
├── requirements.txt        # Python dependencies
├── pytest.ini            # Test configuration
├── docker-compose.yml     # Neo4j development setup
├── setup.py              # Package configuration
└── README.md             # This file
```

## Configuration

The system supports configuration through environment variables or direct configuration:

### Environment Variables

```bash
# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"

# Retrieval Configuration
export RETRIEVAL_DEFAULT_LIMIT=10
export RETRIEVAL_SCORE_THRESHOLD=0.7
export RETRIEVAL_EXPAND_HOPS=2
```

### Programmatic Configuration

```python
from src.config import AppConfig

# Create configuration
config = AppConfig()

# Or load with custom values
config = AppConfig()
config.neo4j.uri = "bolt://production:7687"
config.retrieval.default_limit = 20

# Use with retrieval module
module.initialize(config.to_dict())
```

## Data Models

The system includes pre-built models for the compliance domain:

- **FederalRegulation**: Federal regulations with citations and effective dates
- **AgencyGuidance**: Guidance documents from regulatory agencies  
- **EnforcementAction**: Enforcement actions and penalties
- **ComplianceTopic**: Cross-cutting compliance themes
- **Category**: Standardized categorization (financial, healthcare, data_privacy, etc.)

## API Reference

### RetrievalModule

The main interface for the retrieval system.

```python
from src.retrieval import RetrievalModule

# Initialize
module = RetrievalModule(retriever_type='graphrag')
module.initialize(config_dict)

# Check status
if module.is_initialized():
    # Perform retrieval
    results = module.retrieve(
        query_text="search query",
        filters={'category': 'data_privacy'},
        limit=10
    )
```

### RetrievalResult

Each result contains:

- `content`: Formatted content from the source document
- `metadata`: Additional information (node_id, node_type, etc.)
- `score`: Relevance score (0.0 to 1.0+)
- `source_type`: Type of source document

### Configuration Classes

- `AppConfig`: Main configuration class
- `Neo4jConfig`: Database connection settings
- `RetrievalConfig`: Retrieval algorithm parameters

## Testing

```bash
# Run all unit tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only integration tests (requires Neo4j)
pytest -m integration

# Run specific test file
pytest tests/test_retrieval.py -v
```

## Development

### Setting Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### Running Tests

The test suite includes:
- **Unit tests**: Fast tests with mocked dependencies
- **Integration tests**: Tests requiring actual Neo4j database
- **Data consistency tests**: Validation of sample data relationships

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Check style
flake8 src/ tests/ examples/

# Type checking
mypy src/
```

## Extending the System

### Adding New Retrieval Methods

1. Implement the `BaseRetriever` interface:

```python
from src.retrieval.base import BaseRetriever, RetrievalQuery, RetrievalResult

class CustomRetriever(BaseRetriever):
    def initialize(self, config):
        # Setup your retriever
        pass
        
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        # Implement your retrieval logic
        pass
```

2. Register in `RetrievalModule`:

```python
# In src/retrieval/__init__.py
if self.retriever_type == 'custom':
    self.retriever = CustomRetriever()
```

### Adding New Data Models

1. Define your model in `src/models/data_models.py`:

```python
class CustomDocument(BaseModel):
    title: str
    content: str
    category: Category
```

2. Update the data loader to include your new model type.

## Deployment

### Production Considerations

1. **Database Setup**: Use managed Neo4j (Neo4j Aura) or self-hosted cluster
2. **Environment Variables**: Configure all settings via environment variables
3. **Monitoring**: Implement logging and metrics collection
4. **Scaling**: Consider read replicas for high-traffic scenarios

### Docker Deployment

```bash
# Build production image (you'll need to create Dockerfile)
docker build -t graphrag-retrieval .

# Run with environment variables
docker run -e NEO4J_URI=bolt://production:7687 graphrag-retrieval
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License

## Support

For questions or issues, please open an issue on the repository.