#!/bin/bash

# GraphRAG Fact-Check API Deployment Script
# This script handles local Docker deployment and Azure VM deployment

set -e

echo "🚀 GraphRAG Fact-Check API Deployment"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "📝 Please edit .env file with your actual Azure OpenAI credentials:"
        echo "   - AZURE_OPENAI_ENDPOINT"
        echo "   - AZURE_OPENAI_API_KEY"
        echo "   - AZURE_OPENAI_DEPLOYMENT"
        echo ""
        echo "Run this script again after updating .env file."
        exit 1
    else
        echo "❌ .env.template file not found. Please create .env file manually."
        exit 1
    fi
fi

# Check if essential environment variables are set
if grep -q "your_actual_api_key_here" .env; then
    echo "❌ Please update .env file with your actual Azure OpenAI API key"
    exit 1
fi

echo "✅ Environment configuration validated"

# Build and start services
echo "🔨 Building and starting services..."
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check Neo4j
if curl -f http://localhost:7474 &> /dev/null; then
    echo "✅ Neo4j is running"
else
    echo "❌ Neo4j is not responding"
    docker-compose logs neo4j
    exit 1
fi

# Check API
if curl -f http://localhost:8001/health &> /dev/null; then
    echo "✅ API is running"
else
    echo "❌ API is not responding"
    docker-compose logs api
    exit 1
fi

# Display service information
echo ""
echo "🎉 Deployment successful!"
echo "========================"
echo "📊 Service URLs:"
echo "   • API Endpoint:    http://localhost:8001"
echo "   • API Health:      http://localhost:8001/health"
echo "   • API Debug:       http://localhost:8001/debug"
echo "   • API Docs:        http://localhost:8001/docs"
echo "   • Neo4j Browser:   http://localhost:7474"
echo ""
echo "🧪 Test the API:"
echo "   curl -s http://localhost:8001/health | jq"
echo ""
echo "📋 Management Commands:"
echo "   • View logs:       docker-compose logs -f"
echo "   • Stop services:   docker-compose down"
echo "   • Restart:         docker-compose restart"
echo "   • Update:          ./deploy.sh"
echo ""
echo "🔧 Troubleshooting:"
echo "   • Check logs:      docker-compose logs api"
echo "   • Check status:    docker-compose ps"
echo "   • Shell access:    docker-compose exec api bash"