# 🚀 Sentiment Analysis REST API Documentation

A Flask-based REST API for sentiment analysis with YouTube integration. **No authentication required.**

## 🌐 Base URL
```
http://localhost:5000
```

## 📋 API Endpoints

### 1. Health Check
**GET** `/api/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-07-29T15:30:45.123456",
  "version": "1.0.0"
}
```

### 2. API Status
**GET** `/api/status`

Get API status and model information.

**Response:**
```json
{
  "status": "running",
  "timestamp": "2024-07-29T15:30:45.123456",
  "models": {
    "basic_model": true,
    "enhanced_model": true,
    "available_models": [
      {
        "name": "sentiment_model.pkl",
        "size": 630000,
        "modified": "2024-07-29T15:30:45.123456"
      }
    ]
  },
  "endpoints": { ... }
}
```

### 3. Single Text Sentiment Analysis
**POST** `/api/predict`

Analyze sentiment of a single text.

**Request Body:**
```json
{
  "text": "This movie is absolutely amazing!",
  "use_enhanced": false
}
```

**Response:**
```json
{
  "text": "This movie is absolutely amazing!",
  "sentiment": "positive",
  "confidence": 0.892,
  "probabilities": {
    "negative": 0.045,
    "neutral": 0.063,
    "positive": 0.892
  },
  "model_used": "basic",
  "timestamp": "2024-07-29T15:30:45.123456"
}
```

### 4. Batch Text Analysis
**POST** `/api/predict/batch`

Analyze sentiment of multiple texts (max 100).

**Request Body:**
```json
{
  "texts": [
    "This is great!",
    "I hate this movie",
    "It's okay, nothing special"
  ],
  "use_enhanced": false
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "text": "This is great!",
      "sentiment": "positive",
      "confidence": 0.856,
      "probabilities": { ... }
    },
    {
      "index": 1,
      "text": "I hate this movie",
      "sentiment": "negative",
      "confidence": 0.743,
      "probabilities": { ... }
    }
  ],
  "model_used": "basic",
  "timestamp": "2024-07-29T15:30:45.123456",
  "total_processed": 3
}
```

### 5. YouTube Video Analysis
**POST** `/api/youtube/analyze`

Analyze sentiment of YouTube video comments.

**Request Body:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "api_key": "YOUR_YOUTUBE_API_KEY",
  "max_comments": 100,
  "use_enhanced": false
}
```

**Response:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "total_comments": 95,
  "sentiment_counts": {
    "positive": 60,
    "neutral": 20,
    "negative": 15
  },
  "sentiment_percentages": {
    "positive": 63.16,
    "neutral": 21.05,
    "negative": 15.79
  },
  "comments": [
    {
      "author": "User123",
      "text": "Great video!",
      "like_count": 5,
      "published_at": "2024-07-29T10:30:00Z",
      "sentiment": "positive",
      "confidence": 0.892,
      "probabilities": { ... }
    }
  ],
  "model_used": "basic",
  "timestamp": "2024-07-29T15:30:45.123456"
}
```

### 6. Train New Model
**POST** `/api/train`

Train a new sentiment model.

**Request Body:**
```json
{
  "model_type": "basic",
  "dataset_size": 1000
}
```

**Response:**
```json
{
  "status": "success",
  "model_type": "basic",
  "accuracy": 0.678,
  "model_file": "trained_model_20240729_153045.pkl",
  "dataset_size": 1000,
  "timestamp": "2024-07-29T15:30:45.123456"
}
```

### 7. List Available Models
**GET** `/api/models`

List all available trained models.

**Response:**
```json
{
  "models": {
    "basic_model": true,
    "enhanced_model": true,
    "available_models": [
      {
        "name": "sentiment_model.pkl",
        "size": 630000,
        "modified": "2024-07-29T15:30:45.123456"
      }
    ]
  },
  "timestamp": "2024-07-29T15:30:45.123456"
}
```

## 🔧 Usage Examples

### Python Requests
```python
import requests

# Single text analysis
response = requests.post('http://localhost:5000/api/predict', json={
    'text': 'This movie is fantastic!',
    'use_enhanced': False
})
result = response.json()
print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")

# Batch analysis
response = requests.post('http://localhost:5000/api/predict/batch', json={
    'texts': ['Great!', 'Terrible!', 'Okay'],
    'use_enhanced': False
})
results = response.json()
for item in results['results']:
    print(f"Text: {item['text']} -> {item['sentiment']}")

# YouTube analysis
response = requests.post('http://localhost:5000/api/youtube/analyze', json={
    'video_url': 'https://www.youtube.com/watch?v=VIDEO_ID',
    'api_key': 'YOUR_API_KEY',
    'max_comments': 50
})
analysis = response.json()
print(f"Total comments: {analysis['total_comments']}")
print(f"Positive: {analysis['sentiment_percentages']['positive']:.1f}%")
```

### cURL Examples
```bash
# Health check
curl http://localhost:5000/api/health

# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!", "use_enhanced": false}'

# Batch prediction
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Bad!", "Okay"], "use_enhanced": false}'

# YouTube analysis
curl -X POST http://localhost:5000/api/youtube/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "api_key": "YOUR_API_KEY",
    "max_comments": 100
  }'
```

### JavaScript/Fetch
```javascript
// Single text analysis
const response = await fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'This is wonderful!',
    use_enhanced: false
  })
});
const result = await response.json();
console.log(`Sentiment: ${result.sentiment} (${result.confidence})`);

// Batch analysis
const batchResponse = await fetch('http://localhost:5000/api/predict/batch', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    texts: ['Amazing!', 'Terrible!', 'Average'],
    use_enhanced: false
  })
});
const batchResult = await batchResponse.json();
batchResult.results.forEach(item => {
  console.log(`${item.text} -> ${item.sentiment}`);
});
```

## 🚀 Running the API

### Development Mode
```bash
# Activate environment
source venv/bin/activate

# Run API
python sentiment_api.py
```

### Production Mode (with Gunicorn)
```bash
# Activate environment
source venv/bin/activate

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 sentiment_api:app
```

## 📊 Response Codes

- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (no comments found)
- **500**: Internal Server Error
- **501**: Not Implemented

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
export API_PORT=5000
```

### Model Selection
- Set `use_enhanced: true` for higher accuracy (if available)
- Set `use_enhanced: false` for faster processing

## 🎯 Features

- ✅ **No Authentication Required**
- ✅ **CORS Enabled** (cross-origin requests)
- ✅ **Batch Processing** (up to 100 texts)
- ✅ **YouTube Integration** (with API key)
- ✅ **Multiple Models** (basic and enhanced)
- ✅ **Real-time Training** (basic models)
- ✅ **Health Monitoring**
- ✅ **Comprehensive Error Handling**

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found**: Train a model first using `/api/train`
2. **YouTube API errors**: Check your API key and quota
3. **Memory issues**: Reduce batch size or max_comments
4. **Port conflicts**: Change port in `app.run(port=5001)`

The API is now ready for production use! 🚀
