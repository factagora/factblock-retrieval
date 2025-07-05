# GraphRAG Retrieval System - API Reference

This document provides comprehensive API documentation for the GraphRAG Retrieval System.

## Table of Contents

- [Core Classes](#core-classes)
- [Data Models](#data-models)
- [Configuration](#configuration)
- [Database Client](#database-client)
- [Examples](#examples)

## Core Classes

### RetrievalModule

The main interface for the retrieval system.

```python
from src.retrieval import RetrievalModule
```

#### Constructor

```python
RetrievalModule(retriever_type: str = 'graphrag')
```

**Parameters:**
- `retriever_type` (str): Type of retriever to use. Currently supports 'graphrag'.

#### Methods

##### initialize

```python
initialize(config: Dict[str, Any]) -> None
```

Initialize the retrieval module with configuration.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing:
  - `neo4j_uri` (str): Neo4j database URI
  - `neo4j_user` (str): Database username
  - `neo4j_password` (str): Database password
  - `retrieval_config` (Dict, optional): Additional retrieval parameters

**Raises:**
- `ValueError`: If retriever_type is not supported
- `RuntimeError`: If initialization fails (e.g., database connection)

**Example:**
```python
config = {
    'neo4j_uri': 'bolt://localhost:7687',
    'neo4j_user': 'neo4j',
    'neo4j_password': 'password123'
}
module.initialize(config)
```

##### retrieve

```python
retrieve(
    query_text: str, 
    filters: Optional[Dict] = None, 
    limit: int = 10
) -> List[RetrievalResult]
```

Perform retrieval based on query text and optional filters.

**Parameters:**
- `query_text` (str): Text query to search for
- `filters` (Dict, optional): Filter criteria:
  - `category` (str): Filter by document category
  - Additional filters supported by specific retrievers
- `limit` (int): Maximum number of results to return (default: 10)

**Returns:**
- `List[RetrievalResult]`: List of retrieval results ranked by relevance

**Raises:**
- `RuntimeError`: If module is not initialized

**Example:**
```python
results = module.retrieve(
    query_text="GDPR compliance requirements",
    filters={'category': 'data_privacy'},
    limit=5
)
```

##### is_initialized

```python
is_initialized() -> bool
```

Check if the module has been initialized.

**Returns:**
- `bool`: True if initialized, False otherwise

##### get_retriever_type

```python
get_retriever_type() -> str
```

Get the current retriever type.

**Returns:**
- `str`: The retriever type (e.g., 'graphrag')

### RetrievalQuery

Data class representing a retrieval query.

```python
from src.retrieval import RetrievalQuery
```

#### Attributes

- `query_text` (str): The search query text
- `filters` (Optional[Dict[str, Any]]): Optional filter criteria
- `limit` (int): Maximum number of results (default: 10)

#### Example

```python
query = RetrievalQuery(
    query_text="data privacy regulations",
    filters={'category': 'data_privacy'},
    limit=15
)
```

### RetrievalResult

Data class representing a single retrieval result.

```python
from src.retrieval import RetrievalResult
```

#### Attributes

- `content` (str): Formatted content from the source document
- `metadata` (Dict[str, Any]): Additional metadata about the result
- `score` (float): Relevance score (higher is more relevant)
- `source_type` (str): Type of source document (e.g., 'FederalRegulation')

#### Common Metadata Fields

- `node_id` (int): Database node identifier
- `node_type` (List[str]): Neo4j node labels
- `category` (str): Document category
- `citation` (str): Legal citation (for regulations)
- `agency` (str): Issuing agency (for guidance/enforcement)
- `is_expansion` (bool): True if result came from graph expansion

#### Example

```python
result = RetrievalResult(
    content="Name: GDPR | Description: General Data Protection Regulation",
    metadata={
        'node_id': 123,
        'category': 'data_privacy',
        'citation': 'Regulation (EU) 2016/679'
    },
    score=0.95,
    source_type="FederalRegulation"
)
```

### BaseRetriever

Abstract base class for implementing custom retrievers.

```python
from src.retrieval.base import BaseRetriever
```

#### Abstract Methods

##### initialize

```python
initialize(config: Dict[str, Any]) -> None
```

Initialize the retriever with configuration.

##### retrieve

```python
retrieve(query: RetrievalQuery) -> List[RetrievalResult]
```

Perform retrieval based on the query.

#### Example Implementation

```python
class CustomRetriever(BaseRetriever):
    def initialize(self, config: Dict[str, Any]) -> None:
        # Initialize your retriever
        self.config = config
        
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        # Implement your retrieval logic
        results = []
        # ... retrieval logic ...
        return results
```

### GraphRAGRetriever

Concrete implementation of BaseRetriever using Neo4j graph database.

```python
from src.retrieval import GraphRAGRetriever
```

#### Features

- Graph-based search using Neo4j Cypher queries
- Context expansion through graph relationships
- Relevance scoring based on text similarity and node types
- Category-based filtering

#### Scoring Algorithm

The GraphRAG retriever uses a multi-factor scoring system:

1. **Base Score**: Direct matches get higher base scores than expanded results
2. **Text Similarity**: Term overlap between query and document content
3. **Node Type Boost**: 
   - FederalRegulation: +0.2
   - AgencyGuidance: +0.15
   - EnforcementAction: +0.1
4. **Graph Expansion**: Related nodes included with lower scores

## Data Models

### Category

Enumeration of compliance categories.

```python
from src.models import Category
```

#### Values

- `FINANCIAL = "financial"`
- `HEALTHCARE = "healthcare"`
- `ENVIRONMENTAL = "environmental"`
- `DATA_PRIVACY = "data_privacy"`
- `OTHER = "other"`

### FederalRegulation

Model for federal regulations.

```python
from src.models import FederalRegulation
```

#### Fields

- `name` (str): Regulation name
- `citation` (str): Legal citation
- `description` (str): Regulation description
- `effective_date` (date): When regulation became effective
- `full_text` (Optional[str]): Complete regulation text
- `category` (Category): Regulation category

#### Example

```python
regulation = FederalRegulation(
    name="GDPR",
    citation="Regulation (EU) 2016/679",
    description="General Data Protection Regulation",
    effective_date=date(2018, 5, 25),
    category=Category.DATA_PRIVACY
)
```

### AgencyGuidance

Model for agency guidance documents.

```python
from src.models import AgencyGuidance
```

#### Fields

- `title` (str): Guidance document title
- `agency` (str): Issuing agency
- `date_issued` (date): Issue date
- `summary` (str): Document summary
- `reference_number` (str): Agency reference number
- `category` (Category): Document category

### EnforcementAction

Model for enforcement actions.

```python
from src.models import EnforcementAction
```

#### Fields

- `title` (str): Enforcement action title
- `agency` (str): Enforcing agency
- `date` (date): Action date
- `summary` (str): Action summary
- `docket_number` (str): Legal docket number
- `outcome` (str): Action outcome/penalty
- `category` (Category): Action category

### ComplianceTopic

Model for compliance topics.

```python
from src.models import ComplianceTopic
```

#### Fields

- `name` (str): Topic name
- `description` (str): Topic description
- `related_regulations` (List[str]): List of related regulation names
- `category` (Category): Topic category

## Configuration

### AppConfig

Main configuration class combining all settings.

```python
from src.config import AppConfig, load_config
```

#### Methods

##### to_dict

```python
to_dict() -> Dict[str, Any]
```

Convert configuration to dictionary format for module initialization.

##### get_neo4j_config

```python
get_neo4j_config() -> Dict[str, str]
```

Get Neo4j-specific configuration.

##### get_retrieval_config

```python
get_retrieval_config() -> Dict[str, Any]
```

Get retrieval-specific configuration.

##### validate_config

```python
validate_config() -> bool
```

Validate that configuration is complete and valid.

#### Example

```python
# Load from environment variables
config = load_config()

# Or create programmatically
config = AppConfig()
config.neo4j.uri = "bolt://localhost:7687"
config.retrieval.default_limit = 20

# Use with retrieval module
module.initialize(config.to_dict())
```

### Neo4jConfig

Neo4j database configuration.

#### Environment Variables

- `NEO4J_URI`: Database URI (default: "bolt://localhost:7687")
- `NEO4J_USER`: Username (default: "neo4j")
- `NEO4J_PASSWORD`: Password (default: "password123")

### RetrievalConfig

Retrieval algorithm configuration.

#### Environment Variables

- `RETRIEVAL_DEFAULT_LIMIT`: Default result limit (default: 10)
- `RETRIEVAL_EMBEDDING_MODEL`: Embedding model name
- `RETRIEVAL_SCORE_THRESHOLD`: Minimum score threshold (default: 0.7)
- `RETRIEVAL_EXPAND_HOPS`: Graph expansion depth (default: 2)

## Database Client

### Neo4jClient

Low-level Neo4j database client.

```python
from src.database.neo4j_client import Neo4jClient
```

#### Constructor

```python
Neo4jClient(uri: str, user: str, password: str)
```

#### Methods

##### verify_connectivity

```python
verify_connectivity() -> bool
```

Test database connection.

##### execute_query

```python
execute_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]
```

Execute read-only Cypher query.

##### execute_write_query

```python
execute_write_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]
```

Execute write Cypher query.

##### close

```python
close() -> None
```

Close database connection.

#### Example

```python
client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")

try:
    if client.verify_connectivity():
        results = client.execute_query(
            "MATCH (n:FederalRegulation) RETURN n.name as name LIMIT 5"
        )
        for result in results:
            print(result['name'])
finally:
    client.close()
```

## Error Handling

### Common Exceptions

- `RuntimeError`: Module not initialized or database connection failed
- `ValueError`: Invalid configuration or parameter values
- `ConnectionError`: Database connectivity issues
- `ValidationError`: Invalid data model values (from Pydantic)

### Best Practices

1. **Always check initialization**:
   ```python
   if not module.is_initialized():
       raise RuntimeError("Module must be initialized first")
   ```

2. **Handle connection failures gracefully**:
   ```python
   try:
       module.initialize(config)
   except RuntimeError as e:
       logger.error(f"Failed to initialize: {e}")
       # Fallback logic
   ```

3. **Validate configuration**:
   ```python
   config = load_config()
   if not config.validate_config():
       raise ValueError("Invalid configuration")
   ```

## Performance Considerations

### Optimization Tips

1. **Use appropriate limits**: Don't retrieve more results than needed
2. **Apply filters**: Category filters reduce search space
3. **Connection pooling**: Neo4j client uses connection pooling automatically
4. **Batch queries**: Process multiple queries efficiently
5. **Monitor performance**: Use logging to track query times

### Example Performance Monitoring

```python
import time
import logging

logger = logging.getLogger(__name__)

def timed_retrieval(module, query_text, **kwargs):
    start_time = time.time()
    results = module.retrieve(query_text, **kwargs)
    execution_time = time.time() - start_time
    
    logger.info(f"Query '{query_text}' took {execution_time:.3f}s, {len(results)} results")
    return results
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Basic retrieval operations
- `advanced_usage.py`: Custom retrievers, batch processing, performance monitoring
- `example_data_loader.py`: Loading sample data into Neo4j

For more examples and tutorials, refer to the main README.md file.