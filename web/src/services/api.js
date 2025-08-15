import axios from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    // Handle different error types
    if (error.code === 'ECONNABORTED') {
      error.message = 'Request timeout - please try again';
    } else if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      error.message = data?.detail || data?.message || `Server error: ${status}`;
    } else if (error.request) {
      // Request was made but no response received
      error.message = 'Unable to connect to the server. Please check if the API is running.';
    }
    
    return Promise.reject(error);
  }
);

// Utility functions for sentiment analysis
export const getSentimentColor = (sentiment) => {
  switch (sentiment?.toLowerCase()) {
    case 'positive':
      return 'text-green-600';
    case 'negative':
      return 'text-red-600';
    case 'neutral':
      return 'text-yellow-600';
    default:
      return 'text-gray-600';
  }
};

export const getSentimentIcon = (sentiment) => {
  switch (sentiment?.toLowerCase()) {
    case 'positive':
      return '😊';
    case 'negative':
      return '😞';
    case 'neutral':
      return '😐';
    default:
      return '❓';
  }
};

export const formatConfidence = (confidence) => {
  if (typeof confidence !== 'number') return 'N/A';
  return `${(confidence * 100).toFixed(1)}%`;
};

export const getConfidenceColor = (confidence) => {
  if (typeof confidence !== 'number') return 'text-gray-500';
  if (confidence >= 0.8) return 'text-green-600';
  if (confidence >= 0.6) return 'text-yellow-600';
  return 'text-red-600';
};

// API Service Class
class SentimentAPI {
  // Health and Status
  async healthCheck() {
    const response = await api.get('/api/health');
    return response.data;
  }

  async getStatus() {
    const response = await api.get('/api/status');
    return response.data;
  }

  // Sentiment Analysis
  async predictSingle(text, modelName = 'basic') {
    const response = await api.post('/api/predict', {
      text,
      model_name: modelName,
    });
    return response.data;
  }

  async predictBatch(texts, modelName = 'basic') {
    const response = await api.post('/api/predict/batch', {
      texts,
      model_name: modelName,
    });
    return response.data;
  }

  // YouTube Analysis
  async analyzeYouTubeVideo(videoUrl, maxComments = 100, modelName = 'basic') {
    const response = await api.post('/api/youtube/analyze', {
      video_url: videoUrl,
      max_comments: maxComments,
      model_name: modelName,
    });
    return response.data;
  }

  async getYouTubeTrends(videoUrl, maxComments = 200, modelName = 'basic') {
    const response = await api.post('/api/youtube/trends', {
      video_url: videoUrl,
      max_comments: maxComments,
      model_name: modelName,
    });
    return response.data;
  }

  async compareYouTubeVideos(videoUrls, maxCommentsPerVideo = 100, modelName = 'basic') {
    const response = await api.post('/api/youtube/compare', {
      video_urls: videoUrls,
      max_comments_per_video: maxCommentsPerVideo,
      model_name: modelName,
    });
    return response.data;
  }

  // Model Management
  async listModels() {
    const response = await api.get('/api/models');
    return response.data;
  }

  async trainModel(texts, labels, modelName = 'basic', modelType = 'logistic') {
    const response = await api.post('/api/train', {
      texts,
      labels,
      model_name: modelName,
      model_type: modelType,
    });
    return response.data;
  }

  async getModelMetrics(modelName, testTexts, testLabels) {
    const response = await api.post('/api/models/metrics', {
      model_name: modelName,
      test_texts: testTexts,
      test_labels: testLabels,
    });
    return response.data;
  }
}

// Create and export singleton instance
const sentimentAPI = new SentimentAPI();
export default sentimentAPI;

// Export individual methods for convenience
export const {
  healthCheck,
  getStatus,
  predictSingle,
  predictBatch,
  analyzeYouTubeVideo,
  getYouTubeTrends,
  compareYouTubeVideos,
  listModels,
  trainModel,
  getModelMetrics,
} = sentimentAPI;
