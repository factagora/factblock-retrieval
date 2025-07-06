# GitHub Actions CI/CD Setup Guide

## Overview

This guide helps you configure GitHub Actions for automated testing and deployment of the GraphRAG Fact-Check API to your Azure VM.

## Required GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions, then add these secrets:

### 1. Azure VM Connection
- **`AZURE_VM_IP`**: Your Azure VM public IP address
  ```
  20.81.43.138
  ```

- **`AZURE_VM_USER`**: SSH username for your Azure VM
  ```
  azureuser
  ```

- **`AZURE_VM_SSH_KEY`**: Your private SSH key content
  ```
  -----BEGIN RSA PRIVATE KEY-----
  [Your full SSH private key content from factblock-retriever-vm_key.pem]
  -----END RSA PRIVATE KEY-----
  ```

### 2. Azure OpenAI Configuration
- **`AZURE_OPENAI_ENDPOINT`**: Your Azure OpenAI endpoint
  ```
  https://factagora-aoai-eastus.openai.azure.com/
  ```

- **`AZURE_OPENAI_API_KEY`**: Your Azure OpenAI API key
  ```
  8f3c930c3785429e9393c55d2aae952b
  ```

- **`AZURE_OPENAI_DEPLOYMENT`**: Your deployment name
  ```
  gpt-4o
  ```

## How to Set Up Secrets

1. **Navigate to Secrets**:
   - Go to your GitHub repository
   - Click **Settings** tab
   - Click **Secrets and variables** → **Actions**

2. **Add each secret**:
   - Click **New repository secret**
   - Enter the name exactly as shown above
   - Paste the corresponding value
   - Click **Add secret**

## SSH Key Setup

To get your SSH private key content:

```bash
# Display your private key (copy the entire output)
cat ~/.ssh/factblock-retriever-vm_key.pem
```

**Important**: Copy the entire key including the header and footer lines.

## Workflow Features

The GitHub Actions workflow provides:

### ✅ **Automated Testing**
- Tests API endpoints
- Validates response formats
- Checks service health
- Runs on every push and PR

### 🚀 **Automated Deployment**
- Only deploys from `main` branch
- Creates backup before deployment
- Builds fresh Docker images
- Restarts services gracefully
- Validates deployment health

### 🔄 **Automatic Rollback**
- Triggers if deployment fails
- Restores from backup
- Ensures service availability

### 📊 **Deployment Validation**
- Internal health checks
- External accessibility tests
- Service status reporting

## Workflow Triggers

The workflow runs on:
- **Push to main**: Full test → deploy → validate
- **Pull request**: Test only (no deployment)
- **Manual trigger**: Available in Actions tab

## Monitoring Deployments

1. **View workflow runs**:
   - Go to **Actions** tab in your repository
   - Click on the latest workflow run

2. **Check deployment status**:
   - Visit: `http://your-vm-ip:8001/health`
   - API docs: `http://your-vm-ip:8001/docs`

3. **View logs**:
   - SSH to VM: `ssh -i key.pem azureuser@vm-ip`
   - Check logs: `cd /opt/factblock-retrieval && sudo docker-compose logs -f`

## Testing the Setup

After configuring secrets:

1. **Make a small change** to trigger deployment:
   ```bash
   # Make a small change and push
   git add .
   git commit -m "test: trigger CI/CD pipeline"
   git push origin main
   ```

2. **Monitor the workflow**:
   - Check GitHub Actions tab
   - Watch for green checkmarks ✅

3. **Verify deployment**:
   - Test API: `curl http://your-vm-ip:8001/health`
   - Check examples: `curl http://your-vm-ip:8001/example-texts`

## Troubleshooting

### Common Issues:

1. **SSH connection fails**:
   - Verify `AZURE_VM_SSH_KEY` contains full key with headers
   - Check `AZURE_VM_IP` and `AZURE_VM_USER` are correct

2. **Deployment fails**:
   - Check Azure VM is running
   - Verify ports 8001, 7474 are open in NSG
   - Check VM has sufficient disk space

3. **Health checks fail**:
   - Services may need more time to start
   - Check Docker and Docker Compose are working on VM

### Manual Recovery:

If deployment fails, you can manually rollback:

```bash
ssh -i ~/.ssh/factblock-retriever-vm_key.pem azureuser@your-vm-ip
cd /opt/factblock-retrieval

# List available backups
ls -la src_backup_*

# Restore from most recent backup
sudo rm -rf src
sudo mv src_backup_YYYYMMDD_HHMMSS src
sudo docker-compose restart api
```

## Security Best Practices

- ✅ All secrets stored securely in GitHub
- ✅ SSH keys with proper permissions
- ✅ No secrets in code repository
- ✅ Automated backups before deployment
- ✅ Health checks and rollback capability

## Next Steps

After setup:
1. Test the pipeline with a small change
2. Monitor first few deployments
3. Set up branch protection rules
4. Consider staging environment for larger changes

Your GraphRAG API now has professional-grade CI/CD! 🚀