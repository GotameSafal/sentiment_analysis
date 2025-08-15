# 🚀 Sentiment Analysis API v2.0

A professional, modular REST API for sentiment analysis with YouTube integration. Built with clean architecture and separation of concerns.

## 🏗️ **Project Architecture**

```
sentiment_analysis/
├── api/                    # API layer (routes, endpoints)
│   ├── __init__.py
│   └── routes.py          # Clean API endpoints
├── models/                 # Model definitions and interfaces
│   ├── __init__.py
│   └── base_model.py      # Abstract base classes
├── services/              # Business logic layer
│   ├── __init__.py
│   ├── sentiment_service.py  # Core sentiment analysis
│   └── youtube_service.py    # YouTube integration
├── config/                # Configuration management
│   ├── __init__.py
│   └── settings.py        # Centralized configuration
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── data_loader.py     # Data loading utilities
├── tests/                 # Test suite
│   ├── __init__.py
│   └── test_api.py        # API tests
├── app.py                 # Main Flask application
├── requirements.txt       # Dependencies
└── README_v2.md          # This documentation
```

## ✨ **Key Features**

### **🎯 Core Functionality**
- **Single & Batch Text Analysis** - Analyze individual texts or batches up to 100
- **YouTube Video Analysis** - Extract and analyze video comments
- **Sentiment Trends** - Track sentiment changes over time
- **Multi-Video Comparison** - Compare sentiment across multiple videos
- **Model Training** - Train custom models via API
- **Performance Metrics** - Detailed model evaluation

### **🏗️ Architecture Benefits**
- **Modular Design** - Clean separation of concerns
- **Scalable Structure** - Easy to extend and maintain
- **Configuration Management** - Centralized settings
- **Error Handling** - Comprehensive error responses
- **Testing Framework** - Built-in test suite
- **Documentation** - Self-documenting API

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Clone and setup
cd sentiment_analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **2. Run the API**
```bash
# Development mode
python app.py

# Production mode
gunicorn -w 4 -b 0.0.0.0:5000 app:create_app()
```

### **3. Test the API**
```bash
# Health check
curl http://localhost:5000/api/health

# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!", "model_name": "basic"}'
```

## 📚 **API Documentation**

### **Base URL**: `http://localhost:5000`

### **Endpoints Overview**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/status` | API status and configuration |
| POST | `/api/predict` | Single text analysis |
| POST | `/api/predict/batch` | Batch text analysis |
| POST | `/api/youtube/analyze` | YouTube video analysis |
| POST | `/api/youtube/trends` | Sentiment trends analysis |
| POST | `/api/youtube/compare` | Multi-video comparison |
| POST | `/api/train` | Train new model |
| GET | `/api/models` | List available models |
| POST | `/api/models/metrics` | Model performance metrics |

### **Example Usage**

#### **Single Text Analysis**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is absolutely fantastic!",
    "model_name": "basic"
  }'
```

**Response:**
```json
{
  "text": "This product is absolutely fantastic!",
  "sentiment": "positive",
  "confidence": 0.892,
  "probabilities": {
    "negative": 0.045,
    "neutral": 0.063,
    "positive": 0.892
  },
  "model_name": "basic",
  "timestamp": "2024-07-29T15:30:45.123456"
}
```

#### **Batch Analysis**
```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Great product!", "Terrible service", "It'\''s okay"],
    "model_name": "basic"
  }'
```

#### **YouTube Analysis**
```bash
curl -X POST http://localhost:5000/api/youtube/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "api_key": "YOUR_YOUTUBE_API_KEY",
    "max_comments": 100,
    "model_name": "basic"
  }'
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=5000
export API_DEBUG=False

# Model Configuration
export DEFAULT_MODEL_TYPE=basic
export MAX_FEATURES=10000

# YouTube Configuration
export YOUTUBE_API_KEY=your_api_key_here
export YOUTUBE_DEFAULT_MAX_COMMENTS=100

# Logging
export LOG_LEVEL=INFO
export LOG_TO_FILE=True
```

### **Configuration Files**
- `config/settings.py` - Centralized configuration management
- Environment-specific configs (development, production, testing)

## 🧪 **Testing**

### **Run Tests**
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m unittest tests.test_api.TestAPI.test_health_check

# Run with coverage
python -m pytest tests/ --cov=.
```

### **API Testing**
```bash
# Test all endpoints
python tests/test_api.py
```

## 🎯 **Model Management**

### **Available Models**
- **Basic Model** - Fast, lightweight sentiment analysis
- **Enhanced Model** - Advanced features with ensemble methods
- **Custom Models** - Train your own models via API

### **Training New Models**
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Great!", "Terrible!", "Okay"],
    "labels": ["positive", "negative", "neutral"],
    "model_name": "basic",
    "model_type": "logistic"
  }'
```

### **Model Metrics**
```bash
curl -X POST http://localhost:5000/api/models/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "basic",
    "test_texts": ["Great!", "Bad!"],
    "test_labels": ["positive", "negative"]
  }'
```

## 🚀 **Deployment**

### **Development**
```bash
python app.py
```

### **Production with Gunicorn**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:create_app()
```

### **Docker Deployment**
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:create_app()"]
```

### **Cloud Deployment**
- **AWS**: Elastic Beanstalk, ECS, Lambda
- **Google Cloud**: App Engine, Cloud Run
- **Azure**: App Service, Container Instances
- **Heroku**: Direct deployment

## 📊 **Performance**

### **Benchmarks**
- **Single prediction**: ~50-100ms
- **Batch (10 texts)**: ~200-500ms
- **YouTube analysis (100 comments)**: ~5-15 seconds
- **Model training**: ~2-10 minutes (depending on size)

### **Scalability**
- **Horizontal scaling**: Multiple workers with Gunicorn
- **Load balancing**: Nginx, HAProxy
- **Caching**: Redis, Memcached support
- **Database**: PostgreSQL, MongoDB support

## 🔒 **Security**

### **Current Features**
- CORS configuration
- Request validation
- Error handling
- Input sanitization

### **Future Enhancements**
- API key authentication
- Rate limiting
- Request logging
- Security headers

## 🛠️ **Development**

### **Adding New Features**
1. **Models**: Extend `models/base_model.py`
2. **Services**: Add business logic in `services/`
3. **API**: Add endpoints in `api/routes.py`
4. **Config**: Update `config/settings.py`
5. **Tests**: Add tests in `tests/`

### **Code Style**
- Follow PEP 8
- Use type hints
- Document functions
- Write tests for new features

## 📈 **Monitoring**

### **Health Checks**
- `/api/health` - Basic health check
- `/api/status` - Detailed status information

### **Logging**
- Structured logging with timestamps
- Configurable log levels
- File and console output
- Error tracking

### **Metrics**
- Request/response times
- Model accuracy
- Error rates
- Resource usage

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

## 📄 **License**

MIT License - see LICENSE file for details.

## 🆘 **Support**

- **Documentation**: Check `/api/status` endpoint
- **Issues**: GitHub issues
- **API Help**: Visit `http://localhost:5000` for interactive docs

---

**Built with ❤️ using Flask, scikit-learn, and modern Python practices.**
