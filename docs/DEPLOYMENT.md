# GraphRAG Retrieval System - Deployment Guide

This guide covers deploying the GraphRAG Retrieval System in various environments, from development to production.

## Table of Contents

- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Development Deployment

### Prerequisites

- Python 3.9 or higher
- Neo4j 5.x database
- 4GB+ RAM (for Neo4j)

### Quick Setup

1. **Clone and install**:
   ```bash
   git clone <repository-url>
   cd factblock-retrieval
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Start Neo4j with Docker**:
   ```bash
   docker-compose up -d
   ```

3. **Load sample data**:
   ```bash
   python examples/example_data_loader.py
   ```

4. **Test the installation**:
   ```bash
   python examples/basic_usage.py
   ```

### Development Configuration

Create `.env` file in project root:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Retrieval Configuration
RETRIEVAL_DEFAULT_LIMIT=10
RETRIEVAL_SCORE_THRESHOLD=0.7
RETRIEVAL_EXPAND_HOPS=2

# Logging
LOG_LEVEL=INFO
```

## Production Deployment

### System Requirements

**Minimum Requirements:**
- 4 CPU cores
- 8GB RAM
- 50GB storage
- Network connectivity to Neo4j

**Recommended Requirements:**
- 8+ CPU cores
- 16GB+ RAM
- 100GB+ SSD storage
- Load balancer for high availability

### Production Configuration

#### Environment Variables

```bash
# Database (use managed service in production)
NEO4J_URI=bolt://production-cluster:7687
NEO4J_USER=prod_user
NEO4J_PASSWORD=${SECURE_PASSWORD}

# Retrieval Tuning
RETRIEVAL_DEFAULT_LIMIT=20
RETRIEVAL_SCORE_THRESHOLD=0.8
RETRIEVAL_EXPAND_HOPS=3

# Performance
NEO4J_POOL_SIZE=50
NEO4J_CONNECTION_TIMEOUT=30

# Security
SSL_ENABLED=true
SSL_CERT_PATH=/etc/ssl/certs/app.crt
SSL_KEY_PATH=/etc/ssl/private/app.key

# Monitoring
LOG_LEVEL=WARNING
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=30
```

#### Production Checklist

- [ ] Use managed Neo4j service (Neo4j Aura or self-hosted cluster)
- [ ] Enable SSL/TLS encryption
- [ ] Configure authentication and authorization
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Implement health checks
- [ ] Set up backup procedures
- [ ] Configure resource limits
- [ ] Enable connection pooling
- [ ] Test disaster recovery

### Security Configuration

#### Neo4j Security

```cypher
-- Create dedicated user for the application
CREATE USER graphrag_app SET PASSWORD 'secure_password' CHANGE NOT REQUIRED;

-- Grant minimal required permissions
GRANT ROLE reader TO graphrag_app;
GRANT TRAVERSE ON GRAPH * TO graphrag_app;
GRANT READ {*} ON GRAPH * TO graphrag_app;
```

#### Application Security

```python
# Use environment variables for secrets
import os
from src.config import AppConfig

config = AppConfig()
config.neo4j.password = os.getenv('NEO4J_PASSWORD')

# Validate SSL certificates
config.neo4j.uri = "neo4j+s://production-cluster:7687"
```

## Docker Deployment

### Development Docker Setup

The included `docker-compose.yml` provides a complete development environment:

```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.14.0
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password123
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  neo4j_data:
  neo4j_logs:
```

### Production Docker Setup

#### Dockerfile

Create `Dockerfile` for the application:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY setup.py .

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 graphrag
USER graphrag

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from src.config import load_config; from src.database.neo4j_client import Neo4jClient; \
         config = load_config(); client = Neo4jClient(**config.get_neo4j_config()); \
         client.verify_connectivity() or exit(1); client.close()"

# Default command
CMD ["python", "-m", "examples.basic_usage"]
```

#### Production Docker Compose

```yaml
version: '3.8'
services:
  graphrag-app:
    build: .
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - LOG_LEVEL=INFO
    depends_on:
      - neo4j
    restart: unless-stopped
    networks:
      - graphrag-network

  neo4j:
    image: neo4j:5.14.0-enterprise
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_dbms_memory_heap_initial__size=2G
      - NEO4J_dbms_memory_heap_max__size=4G
      - NEO4J_dbms_memory_pagecache_size=2G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_conf:/conf
    ports:
      - "7687:7687"
    restart: unless-stopped
    networks:
      - graphrag-network

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_conf:

networks:
  graphrag-network:
    driver: bridge
```

#### Build and Deploy

```bash
# Build the image
docker build -t graphrag-retrieval:latest .

# Run with environment file
docker-compose --env-file .env.prod up -d

# Check logs
docker-compose logs -f graphrag-app

# Scale the application
docker-compose up -d --scale graphrag-app=3
```

## Cloud Deployment

### AWS Deployment

#### Using ECS with Fargate

1. **Create ECR repository**:
   ```bash
   aws ecr create-repository --repository-name graphrag-retrieval
   ```

2. **Build and push image**:
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin \
     123456789012.dkr.ecr.us-east-1.amazonaws.com

   # Build and tag
   docker build -t graphrag-retrieval .
   docker tag graphrag-retrieval:latest \
     123456789012.dkr.ecr.us-east-1.amazonaws.com/graphrag-retrieval:latest

   # Push
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/graphrag-retrieval:latest
   ```

3. **Create ECS task definition**:
   ```json
   {
     "family": "graphrag-retrieval",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "512",
     "memory": "1024",
     "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "graphrag-app",
         "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/graphrag-retrieval:latest",
         "environment": [
           {"name": "NEO4J_URI", "value": "bolt://neo4j-cluster:7687"},
           {"name": "LOG_LEVEL", "value": "INFO"}
         ],
         "secrets": [
           {
             "name": "NEO4J_PASSWORD",
             "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:neo4j-password"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/graphrag-retrieval",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

#### Using Neo4j Aura

```python
# Configure for Neo4j Aura
import os

config = {
    'neo4j_uri': 'neo4j+s://xxxxx.databases.neo4j.io',
    'neo4j_user': 'neo4j',
    'neo4j_password': os.getenv('NEO4J_AURA_PASSWORD'),
    'retrieval_config': {
        'default_limit': 20,
        'score_threshold': 0.8
    }
}
```

### Google Cloud Platform

#### Using Cloud Run

1. **Build and push to Container Registry**:
   ```bash
   # Configure Docker for GCP
   gcloud auth configure-docker

   # Build and tag
   docker build -t gcr.io/PROJECT_ID/graphrag-retrieval .
   docker push gcr.io/PROJECT_ID/graphrag-retrieval
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy graphrag-retrieval \
     --image gcr.io/PROJECT_ID/graphrag-retrieval \
     --platform managed \
     --region us-central1 \
     --set-env-vars NEO4J_URI=bolt://neo4j-vm:7687 \
     --set-secrets NEO4J_PASSWORD=neo4j-password:latest \
     --memory 2Gi \
     --cpu 2 \
     --max-instances 10
   ```

### Azure Deployment

#### Using Container Instances

```bash
# Create resource group
az group create --name graphrag-rg --location eastus

# Deploy container
az container create \
  --resource-group graphrag-rg \
  --name graphrag-retrieval \
  --image myregistry.azurecr.io/graphrag-retrieval:latest \
  --cpu 2 \
  --memory 4 \
  --environment-variables \
    NEO4J_URI=bolt://neo4j-vm:7687 \
    LOG_LEVEL=INFO \
  --secure-environment-variables \
    NEO4J_PASSWORD=secure_password
```

## Monitoring and Maintenance

### Health Checks

Implement health check endpoints:

```python
# health_check.py
from src.config import load_config
from src.retrieval import RetrievalModule

def health_check():
    """Basic health check."""
    try:
        config = load_config()
        module = RetrievalModule('graphrag')
        module.initialize(config.to_dict())
        
        # Test basic retrieval
        results = module.retrieve("test", limit=1)
        
        return {
            'status': 'healthy',
            'database': 'connected',
            'retrieval': 'operational'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }
```

### Logging Configuration

```python
import logging
import os

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/graphrag.log')
    ]
)

# Add request ID for tracing
import uuid
logger = logging.getLogger(__name__)

def log_request(query_text, request_id=None):
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"Request {request_id}: Query '{query_text}'")
    return request_id
```

### Metrics Collection

```python
import time
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.query_count = 0
        self.total_time = 0.0
        self.error_count = 0
        self.category_stats = defaultdict(int)
    
    def record_query(self, execution_time, category=None, error=False):
        self.query_count += 1
        self.total_time += execution_time
        
        if error:
            self.error_count += 1
        
        if category:
            self.category_stats[category] += 1
    
    def get_stats(self):
        return {
            'total_queries': self.query_count,
            'avg_response_time': self.total_time / max(self.query_count, 1),
            'error_rate': self.error_count / max(self.query_count, 1),
            'category_breakdown': dict(self.category_stats)
        }
```

### Database Maintenance

#### Regular Maintenance Tasks

```cypher
-- Clean up old log entries (if using Neo4j for logging)
MATCH (log:LogEntry) 
WHERE log.timestamp < datetime() - duration('P7D')
DELETE log;

-- Update database statistics
CALL db.stats.collect();

-- Check database constraints
CALL db.constraints;

-- Monitor memory usage
CALL dbms.queryJmx("java.lang:type=Memory");
```

#### Backup Procedures

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/neo4j"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
neo4j-admin backup \
  --backup-dir="$BACKUP_DIR" \
  --name="graphrag_$DATE" \
  --from=bolt://localhost:7687

# Compress backup
tar -czf "$BACKUP_DIR/graphrag_$DATE.tar.gz" \
  "$BACKUP_DIR/graphrag_$DATE"

# Clean up old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
```

## Troubleshooting

### Common Issues

#### Connection Problems

**Issue**: Cannot connect to Neo4j
```
RuntimeError: Failed to establish Neo4j connection
```

**Solutions**:
1. Check Neo4j is running: `docker ps` or `systemctl status neo4j`
2. Verify connection parameters: URI, username, password
3. Check firewall settings for port 7687
4. Review Neo4j logs: `docker logs <neo4j-container>`

#### Performance Issues

**Issue**: Slow query performance

**Solutions**:
1. Check database indexes:
   ```cypher
   CALL db.indexes;
   ```

2. Create missing indexes:
   ```cypher
   CREATE INDEX FOR (n:FederalRegulation) ON (n.category);
   CREATE INDEX FOR (n:AgencyGuidance) ON (n.category);
   CREATE TEXT INDEX FOR (n:FederalRegulation) ON (n.description);
   ```

3. Monitor query performance:
   ```cypher
   CALL dbms.listQueries();
   ```

#### Memory Issues

**Issue**: Out of memory errors

**Solutions**:
1. Increase Neo4j heap size:
   ```bash
   NEO4J_dbms_memory_heap_initial__size=2G
   NEO4J_dbms_memory_heap_max__size=4G
   ```

2. Reduce query result limits
3. Implement pagination for large result sets

### Debugging Tips

1. **Enable debug logging**:
   ```python
   import logging
   logging.getLogger('src.retrieval').setLevel(logging.DEBUG)
   ```

2. **Test database connection**:
   ```python
   from src.database.neo4j_client import Neo4jClient
   
   client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
   print(client.verify_connectivity())
   print(client.get_database_info())
   ```

3. **Validate configuration**:
   ```python
   from src.config import load_config
   
   config = load_config()
   print(config.validate_config())
   print(config)
   ```

### Performance Tuning

#### Neo4j Optimization

```bash
# Increase page cache for better read performance
NEO4J_dbms_memory_pagecache_size=4G

# Optimize for read-heavy workloads
NEO4J_dbms_tx_log_rotation_retention_policy=3 days

# Enable query logging for performance analysis
NEO4J_dbms_logs_query_enabled=true
NEO4J_dbms_logs_query_threshold=100ms
```

#### Application Optimization

```python
# Connection pooling configuration
config = {
    'neo4j_uri': 'bolt://localhost:7687',
    'neo4j_user': 'neo4j',
    'neo4j_password': 'password',
    'pool_size': 50,  # Increase for high concurrency
    'connection_timeout': 30,
    'retrieval_config': {
        'default_limit': 10,  # Reasonable default
        'score_threshold': 0.7,  # Filter low-quality results
        'expand_hops': 2  # Limit graph traversal depth
    }
}
```

For additional support, refer to the main documentation or open an issue on the project repository.