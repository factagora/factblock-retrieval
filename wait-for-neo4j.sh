#!/bin/bash
# Wait for Neo4j to be ready before starting the API

set -e

host="$1"
port="$2"
shift 2
cmd="$@"

echo "Waiting for Neo4j at $host:$port..."

# Wait up to 300 seconds (5 minutes) for Neo4j to be ready
timeout=300
count=0

while ! nc -z "$host" "$port"; do
  sleep 5
  count=$((count + 5))
  if [ $count -ge $timeout ]; then
    echo "Timeout waiting for Neo4j at $host:$port"
    exit 1
  fi
  echo "Still waiting for Neo4j... ($count/$timeout seconds)"
done

echo "Neo4j is ready at $host:$port"

# Additional HTTP check
echo "Checking Neo4j HTTP endpoint..."
for i in {1..30}; do
  if curl -s -f "http://$host:7474" > /dev/null 2>&1; then
    echo "Neo4j HTTP endpoint is ready"
    break
  fi
  echo "Waiting for Neo4j HTTP endpoint... ($i/30)"
  sleep 5
done

echo "Starting API..."
exec $cmd