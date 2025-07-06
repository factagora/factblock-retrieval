# GraphRAG Fact-Check API - Azure VM Deployment Guide

## Overview

This guide covers deploying the GraphRAG Fact-Check API (Neo4j + FastAPI) to an Azure VM using Docker Compose.

## Prerequisites

- Azure subscription with available credits
- Azure CLI installed locally
- Your Azure OpenAI credentials

## Deployment Options

### Option 1: Azure Portal (Recommended for beginners)

1. **Create VM in Azure Portal:**
   - Go to Azure Portal → Virtual Machines → Create
   - Choose Ubuntu 20.04 LTS
   - Size: Standard_B2ms (2 vCPUs, 8 GB RAM) minimum
   - Authentication: SSH public key
   - Inbound ports: HTTP (80), HTTPS (443), SSH (22), Custom (8001, 7474)

2. **Upload cloud-init script:**
   - In "Advanced" tab, paste contents of `azure-cloud-init.yml`

3. **Create and wait for deployment**

### Option 2: Azure CLI (Advanced)

```bash
# Create resource group
az group create --name factblock-rg --location eastus

# Create VM with cloud-init
az vm create \
  --resource-group factblock-rg \
  --name factblock-vm \
  --image UbuntuLTS \
  --size Standard_B2ms \
  --admin-username ubuntu \
  --generate-ssh-keys \
  --custom-data azure-cloud-init.yml

# Open ports
az vm open-port --resource-group factblock-rg --name factblock-vm --port 8001 --priority 1000
az vm open-port --resource-group factblock-rg --name factblock-vm --port 7474 --priority 1001

# Get public IP
az vm show -d --resource-group factblock-rg --name factblock-vm --query publicIps -o tsv
```

## Post-Deployment Setup

### 1. SSH into the VM

```bash
ssh ubuntu@YOUR_VM_IP
```

### 2. Upload Project Files

**Option A: Direct upload (if you have the code locally)**
```bash
# From your local machine
scp -r . ubuntu@YOUR_VM_IP:/opt/factblock-retrieval/
```

**Option B: Git clone (if code is in repository)**
```bash
# On the VM
cd /opt/factblock-retrieval
git clone YOUR_REPO_URL .
```

### 3. Configure Environment Variables

```bash
cd /opt/factblock-retrieval
cp .env.production .env
nano .env
```

Update with your actual values:
```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your_actual_api_key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### 4. Start Services

```bash
# Start the application
sudo systemctl start factblock-retrieval

# Enable auto-start on boot
sudo systemctl enable factblock-retrieval

# Check status
sudo systemctl status factblock-retrieval
```

### 5. Verify Deployment

```bash
# Check containers
sudo docker-compose ps

# Check logs
sudo docker-compose logs -f

# Test API health
curl http://localhost:8001/health

# Test from outside
curl http://YOUR_VM_IP:8001/health
```

## Service URLs

- **API Endpoint:** `http://YOUR_VM_IP:8001`
- **Neo4j Browser:** `http://YOUR_VM_IP:7474`
- **Health Check:** `http://YOUR_VM_IP:8001/health`
- **Debug Info:** `http://YOUR_VM_IP:8001/debug`
- **API Docs:** `http://YOUR_VM_IP:8001/docs`

## Management Commands

```bash
# View logs
sudo docker-compose logs -f api
sudo docker-compose logs -f neo4j

# Restart services
sudo docker-compose restart

# Stop services
sudo docker-compose down

# Update deployment
cd /opt/factblock-retrieval
git pull  # if using git
sudo docker-compose down
sudo docker-compose build --no-cache
sudo docker-compose up -d

# Check resource usage
docker stats
```

## Troubleshooting

### Common Issues

1. **Port not accessible externally:**
   ```bash
   # Check Azure NSG rules in portal
   # Ensure ports 8001 and 7474 are open
   ```

2. **Neo4j not starting:**
   ```bash
   sudo docker-compose logs neo4j
   # Check memory allocation in docker-compose.yml
   ```

3. **API can't connect to Neo4j:**
   ```bash
   sudo docker-compose logs api
   # Check NEO4J_URI in environment variables
   ```

4. **Azure OpenAI errors:**
   ```bash
   curl http://localhost:8001/debug
   # Check API keys and endpoint configuration
   ```

### Performance Tuning

For production workloads, consider:

1. **Upgrade VM size:** Standard_D2s_v3 or larger
2. **Add managed disks:** For Neo4j data persistence
3. **Configure backup:** Regular snapshots of VM/data
4. **Set up monitoring:** Azure Monitor for VM metrics
5. **Load balancer:** If scaling to multiple VMs

## Security Considerations

1. **Restrict access:** Use Azure NSG to limit source IPs
2. **HTTPS:** Set up SSL/TLS certificates
3. **Firewall:** Configure UFW on the VM
4. **Updates:** Regularly update VM and containers
5. **Secrets:** Use Azure Key Vault for sensitive data

## Cost Optimization

- **Schedule shutdown:** Use Azure Automation for non-24/7 workloads
- **Right-size VM:** Monitor usage and adjust size accordingly
- **Reserved instances:** For long-term deployments
- **Storage optimization:** Use appropriate disk types

## Next Steps

After successful deployment:
1. Test the hybrid fact-checking API
2. Set up monitoring and alerting
3. Configure automated backups
4. Implement CI/CD pipeline
5. Proceed with Task #11 (integrate with other services)