# 🚀 Production Deployment Guide

This guide provides step-by-step instructions for deploying the Sentiment Analysis API in production environments.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **RAM**: 4GB minimum, 8GB+ recommended
- **CPU**: 2+ cores recommended
- **Storage**: 10GB+ free space
- **Network**: Internet access for API dependencies

### Software Requirements
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: For cloning the repository
- **curl**: For health checks and testing

## 🛠️ Quick Deployment

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd sentiment_analysis

# Make deployment script executable
chmod +x docker-deploy.sh

# Deploy everything with one command
./docker-deploy.sh deploy
```

### 2. Verify Deployment
```bash
# Check service status
./docker-deploy.sh status

# Perform health checks
./docker-deploy.sh health

# View logs
./docker-deploy.sh logs
```

### 3. Access the Application
- **API**: http://localhost:5000
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:5000/api/status
- **Health Check**: http://localhost:5000/api/health

## 🔧 Manual Deployment

### Step 1: Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (required)
nano .env
```

**Important Environment Variables:**
```bash
# Production settings
FLASK_ENV=production
API_DEBUG=False
LOG_LEVEL=INFO

# YouTube API (optional)
YOUTUBE_API_KEY=your_actual_api_key

# Performance tuning
MAX_WORKERS=4
PRELOAD_MODELS=True
CACHE_ENABLED=True
```

### Step 2: Build Images
```bash
# Build all Docker images
docker-compose build

# Or build individually
docker build -t sentiment-analysis-api .
docker build -t sentiment-analysis-frontend ./web
```

### Step 3: Start Services
```bash
# Start core services
docker-compose up -d

# Start with monitoring (optional)
docker-compose --profile monitoring up -d

# Start with production profile
docker-compose --profile production up -d
```

### Step 4: Initialize Models
```bash
# Train initial models (if not present)
docker-compose exec sentiment-api python train_sentiment_model.py

# Or copy pre-trained models
docker cp ./models/. sentiment-api:/app/models/
```

## 🌐 Production Configurations

### Load Balancer Setup (Nginx)
```bash
# Enable Nginx reverse proxy
docker-compose --profile production up -d

# Custom Nginx configuration
cp nginx.conf /etc/nginx/sites-available/sentiment-analysis
ln -s /etc/nginx/sites-available/sentiment-analysis /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

### SSL/HTTPS Configuration
```bash
# Generate SSL certificates (Let's Encrypt)
certbot --nginx -d your-domain.com

# Or use custom certificates
mkdir -p ssl
cp your-cert.pem ssl/cert.pem
cp your-key.pem ssl/key.pem
```

### Database Setup (Optional)
```bash
# Enable database persistence
echo "DATABASE_ENABLED=True" >> .env
echo "DATABASE_URL=postgresql://user:pass@db:5432/sentiment" >> .env

# Add PostgreSQL to docker-compose.yml
# (See example in docker-compose.yml)
```

## 📊 Monitoring Setup

### Enable Monitoring Stack
```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access monitoring dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001 (admin/admin)
```

### Custom Metrics
```bash
# Add custom metrics endpoint to API
# Metrics available at: http://localhost:5000/api/metrics

# Configure Prometheus scraping
# Edit monitoring/prometheus.yml
```

## 🔄 Maintenance Operations

### Backup and Restore
```bash
# Create backup
./docker-deploy.sh backup

# Restore from backup
docker cp backups/20240101_120000/models/. sentiment-api:/app/models/
docker-compose restart sentiment-api
```

### Model Updates
```bash
# Train new model
docker-compose exec sentiment-api python train_sentiment_model.py

# Deploy new model
docker-compose exec sentiment-api python -c "
from services.sentiment_service import SentimentAnalysisService
service = SentimentAnalysisService()
service.load_model('enhanced', '/app/models/new_model.pkl')
"
```

### Log Management
```bash
# View logs
docker-compose logs -f sentiment-api

# Rotate logs
docker-compose exec sentiment-api logrotate /etc/logrotate.conf

# Export logs
docker cp sentiment-api:/app/logs/. ./exported-logs/
```

### Scaling
```bash
# Scale API instances
docker-compose up -d --scale sentiment-api=3

# Update load balancer configuration
# Edit nginx.conf upstream block
```

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Port conflicts** | Change ports in .env file |
| **Memory errors** | Increase Docker memory limit |
| **Model loading fails** | Check model files and permissions |
| **API timeouts** | Increase timeout values in config |
| **Database connection** | Verify database credentials and network |

### Debug Commands
```bash
# Check container status
docker-compose ps

# Inspect container
docker inspect sentiment-api

# Execute commands in container
docker-compose exec sentiment-api bash

# Check resource usage
docker stats

# View container logs
docker logs sentiment-api --tail 100
```

### Performance Tuning
```bash
# Optimize for production
export MAX_WORKERS=8
export WORKER_TIMEOUT=300
export PRELOAD_MODELS=True

# Monitor performance
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Profile API endpoints
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:5000/api/predict
```

## 🔒 Security Considerations

### Production Security Checklist
- [ ] Change default passwords
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Set up API rate limiting
- [ ] Enable request logging
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Network segmentation

### Environment Security
```bash
# Secure environment variables
chmod 600 .env

# Use Docker secrets for sensitive data
echo "your_api_key" | docker secret create youtube_api_key -

# Enable container security scanning
docker scan sentiment-analysis-api
```

## 📈 Performance Optimization

### Resource Allocation
```yaml
# docker-compose.yml resource limits
services:
  sentiment-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Caching Strategy
```bash
# Enable Redis caching
CACHE_ENABLED=True
CACHE_TYPE=redis
CACHE_DEFAULT_TTL=3600

# Monitor cache performance
docker-compose exec redis redis-cli info stats
```

### Database Optimization
```bash
# PostgreSQL tuning
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

## 🌍 Cloud Deployment

### AWS Deployment
```bash
# Deploy to ECS
aws ecs create-cluster --cluster-name sentiment-analysis
aws ecs register-task-definition --cli-input-json file://aws-task-definition.json

# Deploy to EKS
kubectl apply -f k8s-deployment.yaml
```

### Google Cloud Deployment
```bash
# Deploy to Cloud Run
gcloud run deploy sentiment-api \
  --image gcr.io/PROJECT-ID/sentiment-analysis \
  --platform managed \
  --region us-central1
```

### Azure Deployment
```bash
# Deploy to Container Instances
az container create \
  --resource-group sentiment-rg \
  --name sentiment-api \
  --image sentiment-analysis:latest
```

## 📞 Support

For deployment issues:
1. Check the troubleshooting section
2. Review container logs
3. Verify environment configuration
4. Test with minimal configuration
5. Create an issue with deployment details

---

**Happy Deploying! 🚀**
