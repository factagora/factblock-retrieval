# Cross-Project Database Access Guide

## Overview

This document explains how the **factblock-collection** project can access and update the production Neo4j database deployed by the **factblock-retrieval** project.

## 🏗️ **Architecture Setup**

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  factblock-         │    │    Azure VM         │    │  factblock-         │
│  collection         │────│    Neo4j Server     │────│  retrieval          │
│                     │    │                     │    │                     │
│ • Data Collection   │    │ • Neo4j Database    │    │ • GraphRAG API      │
│ • Data Processing   │    │ • Port 7687 (Bolt)  │    │ • Fact Checking     │
│ • Production Deploy │    │ • Port 7474 (HTTP)  │    │ • Smart Routing     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         WRITES                    SHARED                      READS
```

## 🔧 **Configuration Files Created**

### For factblock-collection project:

1. **`.env.production`** - Production environment configuration
2. **`config/database.production.json`** - Database connection settings
3. **`scripts/connect_to_production.py`** - Connection testing script
4. **`scripts/deploy_to_production.py`** - Production deployment script
5. **`docs/PRODUCTION_DEPLOYMENT.md`** - Detailed deployment guide

### For factblock-retrieval project:

1. **Updated `docker-compose.yml`** - Enabled external Neo4j connections

## 📋 **Quick Setup Guide**

### Step 1: Get Your Azure VM IP

From your factblock-retrieval deployment, get the Azure VM IP address:
- Check GitHub Actions deployment logs
- Or SSH to VM: `curl ifconfig.me`
- Or Azure Portal: Virtual Machines → Overview

### Step 2: Configure factblock-collection

1. **Navigate to factblock-collection project**:
   ```bash
   cd /Users/randybaek/workspace/factblock-collection
   ```

2. **Update production environment**:
   ```bash
   # Edit .env.production
   NEO4J_URI=bolt://YOUR_AZURE_VM_IP:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   NEO4J_DATABASE=neo4j
   ```

### Step 3: Test Connection

```bash
cd /Users/randybaek/workspace/factblock-collection
python scripts/connect_to_production.py
```

Expected output:
```
✅ Connection successful!
📊 Database info:
   Nodes: 122
   Relationships: 250
   FactBlocks: 120
```

### Step 4: Deploy Data to Production

```bash
cd /Users/randybaek/workspace/factblock-collection
python scripts/deploy_to_production.py
```

## 🔒 **Security Configuration**

### Neo4j Server (factblock-retrieval)
- ✅ **External access enabled**: Neo4j listens on all interfaces
- ✅ **Port exposure**: 7687 (Bolt) and 7474 (HTTP) exposed
- ✅ **Authentication**: Username/password protection
- ✅ **Network security**: Protected by Azure NSG rules

### Client Access (factblock-collection)
- ✅ **Secure connection**: Uses Bolt protocol with authentication
- ✅ **Environment isolation**: Separate production config
- ✅ **Connection validation**: Tests before deployment
- ✅ **Error handling**: Graceful failure and rollback

## 🚀 **Deployment Workflow**

### Option 1: Manual Deployment
```bash
# In factblock-collection project
python scripts/deploy_to_production.py
```

### Option 2: Automated GitHub Actions
Add to factblock-collection `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production Neo4j
on:
  push:
    paths: ['data/processed/enhanced_knowledge_graph_dataset.json']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Deploy to Production
      env:
        NEO4J_URI: bolt://${{ secrets.AZURE_VM_IP }}:7687
        NEO4J_USER: neo4j
        NEO4J_PASSWORD: password
      run: python scripts/deploy_to_production.py
```

## 📊 **Data Flow Process**

1. **factblock-collection** collects and processes data
2. **Enhanced dataset** is generated with FactBlocks and relationships
3. **Production deployment** uploads data to shared Neo4j server
4. **factblock-retrieval API** automatically reads updated data
5. **GraphRAG queries** use the latest FactBlock information
6. **Users** get fact-checking results based on fresh data

## 🔍 **Monitoring & Validation**

### Check Production Data
```bash
# Test connection
python scripts/connect_to_production.py

# Verify API uses new data
curl http://your-vm-ip:8001/health

# Test fact-checking with Korean data
curl -X POST http://your-vm-ip:8001/fact-check-graphrag \
  -H "Content-Type: application/json" \
  -d '{
    "text": "미국 연준이 2022년 기준금리를 7차례 인상했다",
    "max_evidence": 3
  }'
```

### Monitor Performance
- **Database growth**: Track node/relationship count
- **API response times**: Monitor fact-checking performance
- **Memory usage**: Watch Neo4j resource consumption
- **Error rates**: Track connection and query failures

## 🚨 **Troubleshooting**

### Connection Issues
- **Check VM status**: Ensure Azure VM is running
- **Verify ports**: Confirm 7687 is open in Azure NSG
- **Test locally**: Use Neo4j browser at `http://vm-ip:7474`
- **Check credentials**: Verify username/password

### Deployment Issues
- **Dataset exists**: Ensure enhanced_knowledge_graph_dataset.json exists
- **Parsing errors**: Check dataset format and structure
- **Memory limits**: Monitor Neo4j memory during large imports
- **Network timeouts**: Increase timeout for large datasets

### API Integration Issues
- **Cache refresh**: Restart GraphRAG API after data updates
- **Index updates**: Ensure Neo4j indexes are current
- **Query optimization**: Monitor query performance
- **Data consistency**: Verify relationships are properly connected

## 📈 **Next Steps**

1. **Set up monitoring**: Track database health and API performance
2. **Automate deployment**: Create CI/CD pipeline for data updates
3. **Scale infrastructure**: Consider clustering for high availability
4. **Enhance security**: Implement SSL and IP restrictions
5. **Backup strategy**: Regular snapshots and data archival

---

Your cross-project database access is now fully configured! The factblock-collection project can seamlessly update the production Neo4j database that powers the GraphRAG fact-checking API. 🚀