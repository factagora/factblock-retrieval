# GraphRAG Fact-Check API - Automated Deployment Guide

## Overview

The GraphRAG Fact-Check API is deployed automatically using GitHub Actions CI/CD pipeline to an Azure VM. This guide covers the automated deployment process and how to use it.

## 🚀 **Automated Deployment Features**

✅ **Continuous Integration**: Automated testing on every push  
✅ **Continuous Deployment**: Automatic deployment to Azure VM  
✅ **Health Checks**: Validates deployment success  
✅ **Automatic Rollback**: Rolls back if deployment fails  
✅ **Backup System**: Creates backups before deployment  

## 📋 **How to Deploy**

### Simple Deployment (Recommended)

1. **Make your changes** to the codebase
2. **Commit and push** to the main branch:
   ```bash
   git add .
   git commit -m "your commit message"
   git push origin main
   ```
3. **Monitor deployment** in GitHub Actions tab
4. **Access your deployed API** at `http://your-vm-ip:8001`

That's it! The deployment is fully automated.

## 🔧 **Initial Setup (One-time only)**

If you haven't set up the deployment yet, you need to configure GitHub Secrets:

### Required GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions:

- **`AZURE_VM_IP`**: Your Azure VM public IP address
- **`AZURE_VM_USER`**: SSH username (usually `azureuser`)
- **`AZURE_VM_SSH_KEY`**: Your private SSH key content
- **`AZURE_OPENAI_ENDPOINT`**: Your Azure OpenAI endpoint
- **`AZURE_OPENAI_API_KEY`**: Your Azure OpenAI API key
- **`AZURE_OPENAI_DEPLOYMENT`**: Your deployment name (e.g., `gpt-4o`)

### Azure VM Requirements

Your Azure VM should have:
- **OS**: Ubuntu 20.04 LTS or later
- **Size**: Standard_B2ms (2 vCPUs, 8 GB RAM) minimum
- **Ports**: 22 (SSH), 8001 (API), 7474 (Neo4j) open in Network Security Group
- **Docker**: Installed (handled by cloud-init script)

## 📊 **Deployment Process**

When you push to `main` branch, GitHub Actions automatically:

1. **🧪 Tests** the API with automated test suite
2. **🚀 Deploys** to Azure VM via SSH
3. **📦 Builds** fresh Docker containers
4. **🔄 Restarts** services gracefully
5. **🏥 Validates** deployment health
6. **📝 Reports** deployment status

## 🔍 **Monitoring Deployment**

### GitHub Actions Dashboard
- Go to your repository → **Actions** tab
- Click on the latest workflow run
- Monitor progress in real-time

### Deployment Status
- **API Health**: `http://your-vm-ip:8001/health`
- **API Documentation**: `http://your-vm-ip:8001/docs`
- **Neo4j Browser**: `http://your-vm-ip:7474`

### Check Deployment Logs
```bash
ssh azureuser@your-vm-ip
cd /opt/factblock-retrieval
sudo docker-compose logs -f
```

## 🌐 **Service URLs**

Once deployed, access your services at:
- **API Endpoint**: `http://your-vm-ip:8001`
- **API Health Check**: `http://your-vm-ip:8001/health`
- **Interactive API Docs**: `http://your-vm-ip:8001/docs`
- **Example Texts**: `http://your-vm-ip:8001/example-texts`
- **Neo4j Browser**: `http://your-vm-ip:7474`

## ⚡ **Testing Your Deployment**

### Quick Health Check
```bash
curl http://your-vm-ip:8001/health
```

### Test Korean Fact-Checking (Works with your actual data)
```bash
curl -X POST http://your-vm-ip:8001/fact-check-graphrag \
  -H "Content-Type: application/json" \
  -d '{
    "text": "미국 연준이 2022년 기준금리를 7차례 인상했다",
    "max_evidence": 3,
    "compliance_focus": ["financial"]
  }'
```

### Test English Fact-Checking (LLM-based)
```bash
curl -X POST http://your-vm-ip:8001/fact-check-graphrag \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Federal Reserve raised interest rates multiple times in 2022",
    "max_evidence": 3
  }'
```

## 🔧 **Manual Management (Optional)**

If you need to manually manage the deployment:

### SSH Access
```bash
ssh azureuser@your-vm-ip
cd /opt/factblock-retrieval
```

### Service Management
```bash
# View service status
sudo docker-compose ps

# View logs
sudo docker-compose logs -f api
sudo docker-compose logs -f neo4j

# Restart services
sudo docker-compose restart

# Stop services
sudo docker-compose down
```

## 🚨 **Troubleshooting**

### Deployment Fails in GitHub Actions
1. **Check Secrets**: Ensure all GitHub Secrets are properly configured
2. **Check VM Status**: Verify Azure VM is running
3. **Check SSH Key**: Ensure SSH key format is correct (include headers/footers)
4. **Check Network**: Verify ports 8001, 7474, 22 are open in Azure NSG

### API Not Responding
1. **Check health endpoint**: `curl http://your-vm-ip:8001/health`
2. **Check logs**: `sudo docker-compose logs api`
3. **Check containers**: `sudo docker-compose ps`
4. **Restart services**: `sudo docker-compose restart`

### GraphRAG Not Finding Evidence
- **Korean queries work best** with your current Neo4j data
- **English queries** rely on LLM analysis only
- Test with: `"미국 연준이 2022년 기준금리를 7차례 인상했다"`

## 📈 **Production Considerations**

### Performance Optimization
- **VM Size**: Consider upgrading to Standard_D2s_v3 for production
- **Memory**: Monitor Neo4j memory usage
- **Storage**: Use Premium SSD for better performance

### Security
- **Firewall**: Restrict source IPs in Azure NSG
- **HTTPS**: Set up SSL certificates for production
- **Secrets**: Use Azure Key Vault for sensitive data

### Monitoring
- **Azure Monitor**: Set up VM and application monitoring
- **Alerts**: Configure alerts for service failures
- **Backups**: Regular snapshots of VM and data

## 🎯 **Next Steps**

Your GraphRAG Fact-Check API is now fully automated! To deploy new features:

1. **Develop locally** and test your changes
2. **Commit and push** to main branch
3. **Monitor deployment** in GitHub Actions
4. **Test the deployed API** with your new features

The system now supports:
- ✅ **Hybrid fact-checking** (GraphRAG + LLM)
- ✅ **Korean language support** for your Neo4j data
- ✅ **Smart query routing** for optimal performance
- ✅ **Real-world examples** based on your actual data
- ✅ **Automated deployment** with rollback capability