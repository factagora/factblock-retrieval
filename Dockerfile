# Multi-stage build to reduce final image size
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Set working directory
WORKDIR /app

# Create non-root user for security first
RUN useradd --create-home --shell /bin/bash app

# Copy Python packages from builder stage to app user location
COPY --from=builder /root/.local /home/app/.local
RUN chown -R app:app /home/app/.local

# Copy application code
COPY . .

# Copy and set permissions for wait script
COPY wait-for-neo4j.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/wait-for-neo4j.sh

# Set ownership and switch to app user
RUN chown -R app:app /app
USER app

# Set PATH to include user's local bin
ENV PATH="/home/app/.local/bin:$PATH"

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start command with Neo4j wait
CMD ["/usr/local/bin/wait-for-neo4j.sh", "neo4j", "7687", "uvicorn", "src.api.graphrag_fact_check:app", "--host", "0.0.0.0", "--port", "8001"]