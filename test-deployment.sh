#!/bin/bash

# Test script for local Docker deployment

echo "🧪 Testing GraphRAG Fact-Check API deployment..."

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker Compose."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.production .env
    echo "📝 Please edit .env with your Azure OpenAI credentials before running again."
    exit 1
fi

echo "🔄 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

echo "🔍 Checking service status..."
docker-compose ps

echo "🏥 Testing health endpoints..."
echo "Neo4j health check:"
if curl -f http://localhost:7474 &>/dev/null; then
    echo "✅ Neo4j is responding"
else
    echo "❌ Neo4j not responding"
fi

echo "API health check:"
if curl -f http://localhost:8001/health &>/dev/null; then
    echo "✅ API is responding"
    curl -s http://localhost:8001/health | python3 -m json.tool
else
    echo "❌ API not responding"
fi

echo "🔧 Debug info:"
curl -s http://localhost:8001/debug | python3 -m json.tool

echo "📝 Test fact-checking endpoint:"
curl -X POST http://localhost:8001/fact-check-graphrag \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The company reported a 25% increase in revenue last quarter according to their financial statements.",
    "compliance_focus": ["financial"],
    "max_evidence": 3
  }' | python3 -m json.tool

echo "✅ Deployment test complete!"
echo ""
echo "🌐 Services available at:"
echo "   - API: http://localhost:8001"
echo "   - Neo4j Browser: http://localhost:7474"
echo "   - API Docs: http://localhost:8001/docs"
echo ""
echo "📋 To stop services: docker-compose down"