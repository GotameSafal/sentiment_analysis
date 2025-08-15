#!/usr/bin/env python3
"""
API Routes for Sentiment Analysis
Clean separation of API endpoints from business logic.
"""

from flask import request, jsonify
from flask_restful import Resource
from typing import Dict, Any
import logging
from datetime import datetime

from services.sentiment_service import SentimentAnalysisService
from services.youtube_service import YouTubeAnalysisService
from config.settings import APIConfig, YouTubeConfig

logger = logging.getLogger(__name__)


class BaseResource(Resource):
    """Base resource with common functionality."""
    
    def __init__(self):
        self.sentiment_service = SentimentAnalysisService()
        self.youtube_service = None
    
    def _get_youtube_service(self):
        """Get or create YouTube service with environment API key."""
        if not self.youtube_service:
            self.youtube_service = YouTubeAnalysisService(self.sentiment_service)
        return self.youtube_service
    
    def _validate_request_data(self, required_fields: list) -> Dict[str, Any]:
        """Validate request data and return parsed JSON."""
        try:
            data = request.get_json()
            if not data:
                return {'error': 'Request body must be valid JSON'}, 400
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return {'error': f'Missing required fields: {missing_fields}'}, 400
            
            return data, 200
            
        except Exception as e:
            return {'error': f'Invalid JSON: {str(e)}'}, 400
    
    def _handle_service_error(self, e: Exception) -> tuple:
        """Handle service layer errors consistently."""
        logger.error(f"Service error: {e}")
        
        if isinstance(e, ValueError):
            return {'error': str(e)}, 400
        elif isinstance(e, FileNotFoundError):
            return {'error': 'Model not found'}, 404
        else:
            return {'error': 'Internal server error'}, 500


class HealthCheckResource(BaseResource):
    """Health check endpoint."""
    
    def get(self):
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0'
        }


class StatusResource(BaseResource):
    """API status and information."""
    
    def get(self):
        try:
            models_info = self.sentiment_service.list_models()
            
            return {
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'models': models_info,
                'configuration': {
                    'max_batch_size': APIConfig.MAX_BATCH_SIZE,
                    'max_youtube_comments': APIConfig.MAX_YOUTUBE_COMMENTS,
                    'rate_limiting_enabled': APIConfig.RATE_LIMIT_ENABLED
                },
                'endpoints': {
                    'health': '/api/health',
                    'status': '/api/status',
                    'predict': '/api/predict',
                    'batch_predict': '/api/predict/batch',
                    'youtube_analyze': '/api/youtube/analyze',
                    'youtube_trends': '/api/youtube/trends',
                    'youtube_compare': '/api/youtube/compare',
                    'train': '/api/train',
                    'models': '/api/models'
                }
            }
        except Exception as e:
            return self._handle_service_error(e)


class PredictResource(BaseResource):
    """Single text sentiment prediction."""
    
    def post(self):
        try:
            data, status_code = self._validate_request_data(['text'])
            if status_code != 200:
                return data, status_code
            
            text = data['text']
            model_name = data.get('model_name')
            
            if not isinstance(text, str) or not text.strip():
                return {'error': 'Text must be a non-empty string'}, 400
            
            result = self.sentiment_service.predict_sentiment(text, model_name)
            return result
            
        except Exception as e:
            return self._handle_service_error(e)


class BatchPredictResource(BaseResource):
    """Batch text sentiment prediction."""
    
    def post(self):
        try:
            data, status_code = self._validate_request_data(['texts'])
            if status_code != 200:
                return data, status_code
            
            texts = data['texts']
            model_name = data.get('model_name')
            
            if not isinstance(texts, list):
                return {'error': 'texts must be a list'}, 400
            
            if len(texts) == 0:
                return {'error': 'texts list cannot be empty'}, 400
            
            if len(texts) > APIConfig.MAX_BATCH_SIZE:
                return {'error': f'Maximum {APIConfig.MAX_BATCH_SIZE} texts allowed per batch'}, 400
            
            # Validate each text
            for i, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    return {'error': f'Text at index {i} must be a non-empty string'}, 400
            
            results = self.sentiment_service.predict_batch(texts, model_name)
            
            return {
                'results': results,
                'total_processed': len(results),
                'model_used': model_name or 'default',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._handle_service_error(e)


class YouTubeAnalyzeResource(BaseResource):
    """YouTube video comment analysis."""

    def post(self):
        try:
            data, status_code = self._validate_request_data(['video_url'])
            if status_code != 200:
                return data, status_code

            video_url = data['video_url']
            max_comments = data.get('max_comments', YouTubeConfig.DEFAULT_MAX_COMMENTS)
            model_name = data.get('model_name')

            if not isinstance(video_url, str) or not video_url.strip():
                return {'error': 'video_url must be a non-empty string'}, 400

            if max_comments > APIConfig.MAX_YOUTUBE_COMMENTS:
                return {'error': f'Maximum {APIConfig.MAX_YOUTUBE_COMMENTS} comments allowed'}, 400

            # Check if YouTube API is configured
            if not YouTubeConfig.is_configured():
                return {
                    'error': 'YouTube API not configured',
                    'message': 'Please set YOUTUBE_API_KEY environment variable'
                }, 503

            youtube_service = self._get_youtube_service()
            result = youtube_service.analyze_video_comments(video_url, max_comments, model_name)

            return result

        except Exception as e:
            return self._handle_service_error(e)


class YouTubeTrendsResource(BaseResource):
    """YouTube video sentiment trends analysis."""

    def post(self):
        try:
            data, status_code = self._validate_request_data(['video_url'])
            if status_code != 200:
                return data, status_code

            video_url = data['video_url']
            max_comments = data.get('max_comments', 200)
            model_name = data.get('model_name')

            if max_comments > APIConfig.MAX_YOUTUBE_COMMENTS:
                max_comments = APIConfig.MAX_YOUTUBE_COMMENTS

            # Check if YouTube API is configured
            if not YouTubeConfig.is_configured():
                return {
                    'error': 'YouTube API not configured',
                    'message': 'Please set YOUTUBE_API_KEY environment variable'
                }, 503

            youtube_service = self._get_youtube_service()
            result = youtube_service.get_video_sentiment_trends(video_url, max_comments, model_name)

            return result

        except Exception as e:
            return self._handle_service_error(e)


class YouTubeCompareResource(BaseResource):
    """Compare sentiment across multiple YouTube videos."""

    def post(self):
        try:
            data, status_code = self._validate_request_data(['video_urls'])
            if status_code != 200:
                return data, status_code

            video_urls = data['video_urls']
            max_comments_per_video = data.get('max_comments_per_video', 100)
            model_name = data.get('model_name')

            if not isinstance(video_urls, list) or len(video_urls) == 0:
                return {'error': 'video_urls must be a non-empty list'}, 400

            if len(video_urls) > 5:
                return {'error': 'Maximum 5 videos can be compared at once'}, 400

            # Check if YouTube API is configured
            if not YouTubeConfig.is_configured():
                return {
                    'error': 'YouTube API not configured',
                    'message': 'Please set YOUTUBE_API_KEY environment variable'
                }, 503

            youtube_service = self._get_youtube_service()
            result = youtube_service.compare_videos(video_urls, max_comments_per_video, model_name)

            return result

        except Exception as e:
            return self._handle_service_error(e)


class TrainModelResource(BaseResource):
    """Train a new sentiment model."""
    
    def post(self):
        try:
            data, status_code = self._validate_request_data(['texts', 'labels'])
            if status_code != 200:
                return data, status_code
            
            texts = data['texts']
            labels = data['labels']
            model_name = data.get('model_name', 'basic')
            
            if not isinstance(texts, list) or not isinstance(labels, list):
                return {'error': 'texts and labels must be lists'}, 400
            
            if len(texts) != len(labels):
                return {'error': 'texts and labels must have the same length'}, 400
            
            if len(texts) == 0:
                return {'error': 'texts and labels cannot be empty'}, 400
            
            # Validate model name
            available_models = self.sentiment_service.list_models()
            if model_name not in available_models:
                return {'error': f'Model {model_name} not available. Available models: {list(available_models.keys())}'}, 400
            
            # Train model
            training_params = {
                'test_size': data.get('test_size', 0.2),
                'random_state': data.get('random_state', 42),
                'model_type': data.get('model_type', 'logistic')
            }
            
            result = self.sentiment_service.train_model(texts, labels, model_name, **training_params)
            
            return {
                'status': 'success',
                'model_name': model_name,
                'training_result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._handle_service_error(e)


class ModelsResource(BaseResource):
    """List available models and their information."""
    
    def get(self):
        try:
            models_info = self.sentiment_service.list_models()
            
            return {
                'models': models_info,
                'total_models': len(models_info),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._handle_service_error(e)


class ModelMetricsResource(BaseResource):
    """Get model performance metrics."""
    
    def post(self):
        try:
            data, status_code = self._validate_request_data(['model_name', 'test_texts', 'test_labels'])
            if status_code != 200:
                return data, status_code
            
            model_name = data['model_name']
            test_texts = data['test_texts']
            test_labels = data['test_labels']
            
            if not isinstance(test_texts, list) or not isinstance(test_labels, list):
                return {'error': 'test_texts and test_labels must be lists'}, 400
            
            if len(test_texts) != len(test_labels):
                return {'error': 'test_texts and test_labels must have the same length'}, 400
            
            metrics = self.sentiment_service.get_model_metrics(model_name, test_texts, test_labels)
            
            return {
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._handle_service_error(e)
