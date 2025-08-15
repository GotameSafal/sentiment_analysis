#!/usr/bin/env python3
"""
Sentiment Analysis REST API
A Flask-based REST API for sentiment analysis with no authentication required.

Features:
- Single text sentiment analysis
- Batch text analysis
- YouTube video comment analysis
- Model training endpoints
- Health check and status endpoints

Usage:
    python sentiment_api.py
    
API Endpoints:
    GET  /api/health                    - Health check
    GET  /api/status                    - API status and model info
    POST /api/predict                   - Analyze single text
    POST /api/predict/batch             - Analyze multiple texts
    POST /api/youtube/analyze           - Analyze YouTube video
    POST /api/train                     - Train new model
    GET  /api/models                    - List available models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import os
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any
import logging

# Import our sentiment analysis modules
from sentiment_analyzer import SentimentAnalyzer, SentimentDataset
from youtube_comment_extractor import YouTubeCommentExtractor
from enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
api = Api(app)

# Global variables for loaded models
sentiment_analyzer = None
enhanced_analyzer = None
youtube_extractor = None

def load_models():
    """Load available sentiment models."""
    global sentiment_analyzer, enhanced_analyzer
    
    try:
        # Load basic sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer()
        
        # Try to load existing model
        model_files = [
            'sentiment_model.pkl',
            'enhanced_sentiment_model.pkl',
            'ultimate_sentiment_model.pkl'
        ]
        
        model_loaded = False
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    sentiment_analyzer.load_model(model_file)
                    model_loaded = True
                    logger.info(f"Loaded model: {model_file}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_file}: {e}")
        
        if not model_loaded:
            logger.warning("No pre-trained model found. Training basic model...")
            # Train a basic model if none exists
            dataset = SentimentDataset()
            df = dataset.create_sample_dataset(size=1000)
            sentiment_analyzer.train_model(df, model_type='logistic')
            sentiment_analyzer.save_model('sentiment_model.pkl')
            logger.info("Basic model trained and saved")
        
        # Try to load enhanced analyzer
        try:
            enhanced_analyzer = EnhancedSentimentAnalyzer()
            if os.path.exists('enhanced_sentiment_model.pkl'):
                enhanced_analyzer.load_enhanced_model('enhanced_sentiment_model.pkl')
                logger.info("Enhanced model loaded")
        except Exception as e:
            logger.warning(f"Enhanced model not available: {e}")
            enhanced_analyzer = None
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def get_model_info():
    """Get information about loaded models."""
    info = {
        'basic_model': sentiment_analyzer is not None and sentiment_analyzer.is_trained,
        'enhanced_model': enhanced_analyzer is not None and enhanced_analyzer.is_trained,
        'available_models': []
    }
    
    # Check for available model files
    model_files = [
        'sentiment_model.pkl',
        'enhanced_sentiment_model.pkl',
        'ultimate_sentiment_model.pkl'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            info['available_models'].append({
                'name': model_file,
                'size': os.path.getsize(model_file),
                'modified': datetime.fromtimestamp(os.path.getmtime(model_file)).isoformat()
            })
    
    return info

class HealthCheck(Resource):
    """Health check endpoint."""
    
    def get(self):
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }

class APIStatus(Resource):
    """API status and model information."""
    
    def get(self):
        model_info = get_model_info()
        return {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'models': model_info,
            'endpoints': {
                'health': '/api/health',
                'status': '/api/status',
                'predict': '/api/predict',
                'batch_predict': '/api/predict/batch',
                'youtube_analyze': '/api/youtube/analyze',
                'train': '/api/train',
                'models': '/api/models'
            }
        }

class PredictSentiment(Resource):
    """Single text sentiment prediction."""
    
    def post(self):
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return {'error': 'Missing required field: text'}, 400
            
            text = data['text']
            use_enhanced = data.get('use_enhanced', False)
            
            if not text or not isinstance(text, str):
                return {'error': 'Text must be a non-empty string'}, 400
            
            # Choose model
            if use_enhanced and enhanced_analyzer and enhanced_analyzer.is_trained:
                result = enhanced_analyzer.predict_sentiment(text)
                model_used = 'enhanced'
            elif sentiment_analyzer and sentiment_analyzer.is_trained:
                result = sentiment_analyzer.predict_sentiment(text)
                model_used = 'basic'
            else:
                return {'error': 'No trained model available'}, 500
            
            # Add metadata
            result['model_used'] = model_used
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            return {'error': str(e)}, 500

class BatchPredict(Resource):
    """Batch text sentiment prediction."""
    
    def post(self):
        try:
            data = request.get_json()
            
            if not data or 'texts' not in data:
                return {'error': 'Missing required field: texts'}, 400
            
            texts = data['texts']
            use_enhanced = data.get('use_enhanced', False)
            
            if not isinstance(texts, list) or len(texts) == 0:
                return {'error': 'texts must be a non-empty list'}, 400
            
            if len(texts) > 100:
                return {'error': 'Maximum 100 texts allowed per batch'}, 400
            
            # Choose model
            if use_enhanced and enhanced_analyzer and enhanced_analyzer.is_trained:
                analyzer = enhanced_analyzer
                model_used = 'enhanced'
            elif sentiment_analyzer and sentiment_analyzer.is_trained:
                analyzer = sentiment_analyzer
                model_used = 'basic'
            else:
                return {'error': 'No trained model available'}, 500
            
            # Process batch
            results = []
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    results.append({
                        'index': i,
                        'error': 'Text must be a string'
                    })
                    continue
                
                try:
                    result = analyzer.predict_sentiment(text)
                    result['index'] = i
                    results.append(result)
                except Exception as e:
                    results.append({
                        'index': i,
                        'error': str(e)
                    })
            
            return {
                'results': results,
                'model_used': model_used,
                'timestamp': datetime.now().isoformat(),
                'total_processed': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error in batch predict: {e}")
            return {'error': str(e)}, 500

class YouTubeAnalyze(Resource):
    """YouTube video comment analysis."""
    
    def post(self):
        try:
            data = request.get_json()
            
            if not data or 'video_url' not in data:
                return {'error': 'Missing required field: video_url'}, 400
            
            video_url = data['video_url']
            api_key = data.get('api_key')
            max_comments = data.get('max_comments', 100)
            use_enhanced = data.get('use_enhanced', False)
            
            if not api_key:
                return {'error': 'YouTube API key required'}, 400
            
            if max_comments > 500:
                return {'error': 'Maximum 500 comments allowed'}, 400
            
            # Initialize YouTube extractor
            try:
                extractor = YouTubeCommentExtractor(api_key)
            except Exception as e:
                return {'error': f'Failed to initialize YouTube extractor: {e}'}, 400
            
            # Extract comments
            comments = extractor.extract_comments_from_url(video_url, max_comments, save_to_file=False)
            
            if not comments:
                return {'error': 'No comments found or extraction failed'}, 404
            
            # Choose model
            if use_enhanced and enhanced_analyzer and enhanced_analyzer.is_trained:
                analyzer = enhanced_analyzer
                model_used = 'enhanced'
            elif sentiment_analyzer and sentiment_analyzer.is_trained:
                analyzer = sentiment_analyzer
                model_used = 'basic'
            else:
                return {'error': 'No trained model available'}, 500
            
            # Analyze comments
            analyzed_comments = []
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            for comment in comments:
                try:
                    result = analyzer.predict_sentiment(comment['text'])
                    
                    analyzed_comment = {
                        'author': comment['author'],
                        'text': comment['text'],
                        'like_count': comment['like_count'],
                        'published_at': comment['published_at'],
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'probabilities': result['probabilities']
                    }
                    
                    analyzed_comments.append(analyzed_comment)
                    sentiment_counts[result['sentiment']] += 1
                    
                except Exception as e:
                    logger.warning(f"Error analyzing comment: {e}")
            
            # Calculate statistics
            total_comments = len(analyzed_comments)
            sentiment_percentages = {
                sentiment: (count / total_comments * 100) if total_comments > 0 else 0
                for sentiment, count in sentiment_counts.items()
            }
            
            return {
                'video_url': video_url,
                'total_comments': total_comments,
                'sentiment_counts': sentiment_counts,
                'sentiment_percentages': sentiment_percentages,
                'comments': analyzed_comments,
                'model_used': model_used,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in YouTube analyze: {e}")
            return {'error': str(e)}, 500

class TrainModel(Resource):
    """Train a new sentiment model."""
    
    def post(self):
        try:
            data = request.get_json()
            
            model_type = data.get('model_type', 'basic')
            dataset_size = data.get('dataset_size', 1000)
            
            if model_type not in ['basic', 'enhanced']:
                return {'error': 'model_type must be "basic" or "enhanced"'}, 400
            
            if dataset_size > 5000:
                return {'error': 'Maximum dataset size is 5000'}, 400
            
            # Train model
            if model_type == 'basic':
                analyzer = SentimentAnalyzer()
                dataset = SentimentDataset()
                df = dataset.create_sample_dataset(size=dataset_size)
                results = analyzer.train_model(df, model_type='logistic')
                
                # Save model
                model_filename = f'trained_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                analyzer.save_model(model_filename)
                
                return {
                    'status': 'success',
                    'model_type': 'basic',
                    'accuracy': results['accuracy'],
                    'model_file': model_filename,
                    'dataset_size': dataset_size,
                    'timestamp': datetime.now().isoformat()
                }
            
            else:  # enhanced
                return {'error': 'Enhanced model training not implemented in API yet'}, 501
            
        except Exception as e:
            logger.error(f"Error in train model: {e}")
            return {'error': str(e)}, 500

class ListModels(Resource):
    """List available models."""
    
    def get(self):
        try:
            model_info = get_model_info()
            return {
                'models': model_info,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {'error': str(e)}, 500

# Register API routes
api.add_resource(HealthCheck, '/api/health')
api.add_resource(APIStatus, '/api/status')
api.add_resource(PredictSentiment, '/api/predict')
api.add_resource(BatchPredict, '/api/predict/batch')
api.add_resource(YouTubeAnalyze, '/api/youtube/analyze')
api.add_resource(TrainModel, '/api/train')
api.add_resource(ListModels, '/api/models')

@app.route('/')
def index():
    """API documentation."""
    return jsonify({
        'name': 'Sentiment Analysis API',
        'version': '1.0.0',
        'description': 'REST API for sentiment analysis with YouTube integration',
        'endpoints': {
            'GET /api/health': 'Health check',
            'GET /api/status': 'API status and model info',
            'POST /api/predict': 'Analyze single text sentiment',
            'POST /api/predict/batch': 'Analyze multiple texts',
            'POST /api/youtube/analyze': 'Analyze YouTube video comments',
            'POST /api/train': 'Train new model',
            'GET /api/models': 'List available models'
        },
        'documentation': 'See README.md for detailed API usage'
    })

if __name__ == '__main__':
    print("🚀 Starting Sentiment Analysis API...")
    
    # Load models
    if load_models():
        print("✅ Models loaded successfully")
    else:
        print("⚠️  Warning: Some models failed to load")
    
    print("🌐 API running at: http://localhost:5000")
    print("📚 API documentation: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/api/health")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
