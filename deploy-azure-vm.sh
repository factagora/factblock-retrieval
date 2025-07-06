#!/bin/bash

# Azure VM Deployment Script for GraphRAG Fact-Check API
# This script sets up Docker and deploys the application

set -e

echo "🚀 Setting up GraphRAG Fact-Check API on Azure VM..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add current user to docker group
sudo usermod -aG docker $USER

# Install Git
echo "📁 Installing Git..."
sudo apt-get install -y git

# Create application directory
echo "📂 Setting up application directory..."
sudo mkdir -p /opt/factblock-retrieval
sudo chown $USER:$USER /opt/factblock-retrieval
cd /opt/factblock-retrieval

# Clone repository (replace with your actual repo URL)
echo "⬇️ Cloning repository..."
# git clone https://github.com/your-username/factblock-retrieval.git .

echo "📝 Setting up environment variables..."
echo "Please create .env file with your Azure OpenAI credentials:"
echo "cp .env.production .env"
echo "# Then edit .env with your actual values"

# Create systemd service for auto-start
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/factblock-retrieval.service > /dev/null <<EOF
[Unit]
Description=GraphRAG Fact-Check API
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=true
WorkingDirectory=/opt/factblock-retrieval
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable factblock-retrieval.service

# Create update script
echo "📋 Creating update script..."
tee update-deployment.sh > /dev/null <<EOF
#!/bin/bash
echo "🔄 Updating GraphRAG Fact-Check API..."
cd /opt/factblock-retrieval
git pull
docker-compose down
docker-compose build --no-cache
docker-compose up -d
echo "✅ Update complete!"
EOF
chmod +x update-deployment.sh

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy your code to /opt/factblock-retrieval/"
echo "2. Create .env file: cp .env.production .env"
echo "3. Edit .env with your Azure OpenAI credentials"
echo "4. Start services: sudo systemctl start factblock-retrieval"
echo "5. Check status: docker-compose ps"
echo "6. View logs: docker-compose logs -f"
echo ""
echo "🌐 Your API will be available at: http://$(curl -s ifconfig.me):8001"
echo "📊 Neo4j browser at: http://$(curl -s ifconfig.me):7474"