#!/usr/bin/env python3
"""
Main Flask Application
Clean, organized Flask app with proper structure and configuration.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS
from flask_restful import Api
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import APIConfig, LoggingConfig, get_config
from api.routes import (
    HealthCheckResource,
    StatusResource,
    PredictResource,
    BatchPredictResource,
    YouTubeAnalyzeResource,
    YouTubeTrendsResource,
    YouTubeCompareResource,
    TrainModelResource,
    ModelsResource,
    ModelMetricsResource
)

# Configure logging
def setup_logging():
    """Setup application logging."""
    log_level = getattr(logging, LoggingConfig.LEVEL.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if LoggingConfig.LOG_TO_FILE:
        LoggingConfig.LOG_FILE_PATH.parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=LoggingConfig.FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LoggingConfig.LOG_FILE_PATH) if LoggingConfig.LOG_TO_FILE else logging.NullHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    config = get_config()
    app.config.from_object(config)
    
    # Setup CORS
    CORS(app, origins=APIConfig.CORS_ORIGINS)
    
    # Create API instance
    api = Api(app)
    
    # Register routes
    api.add_resource(HealthCheckResource, '/api/health')
    api.add_resource(StatusResource, '/api/status')
    api.add_resource(PredictResource, '/api/predict')
    api.add_resource(BatchPredictResource, '/api/predict/batch')
    api.add_resource(YouTubeAnalyzeResource, '/api/youtube/analyze')
    api.add_resource(YouTubeTrendsResource, '/api/youtube/trends')
    api.add_resource(YouTubeCompareResource, '/api/youtube/compare')
    api.add_resource(TrainModelResource, '/api/train')
    api.add_resource(ModelsResource, '/api/models')
    api.add_resource(ModelMetricsResource, '/api/models/metrics')
    
    # Root endpoint with API documentation
    @app.route('/')
    def index():
        return jsonify({
            'name': 'Sentiment Analysis API',
            'version': '2.0.0',
            'description': 'Professional REST API for sentiment analysis with YouTube integration',
            'architecture': 'Modular design with separated concerns',
            'features': [
                'Single and batch text analysis',
                'YouTube video comment analysis',
                'Sentiment trends analysis',
                'Multi-video comparison',
                'Model training and metrics',
                'Clean API architecture'
            ],
            'endpoints': {
                'health': {
                    'method': 'GET',
                    'url': '/api/health',
                    'description': 'Health check'
                },
                'status': {
                    'method': 'GET',
                    'url': '/api/status',
                    'description': 'API status and configuration'
                },
                'predict': {
                    'method': 'POST',
                    'url': '/api/predict',
                    'description': 'Analyze single text sentiment'
                },
                'batch_predict': {
                    'method': 'POST',
                    'url': '/api/predict/batch',
                    'description': 'Analyze multiple texts sentiment'
                },
                'youtube_analyze': {
                    'method': 'POST',
                    'url': '/api/youtube/analyze',
                    'description': 'Analyze YouTube video comments'
                },
                'youtube_trends': {
                    'method': 'POST',
                    'url': '/api/youtube/trends',
                    'description': 'Analyze sentiment trends over time'
                },
                'youtube_compare': {
                    'method': 'POST',
                    'url': '/api/youtube/compare',
                    'description': 'Compare sentiment across multiple videos'
                },
                'train': {
                    'method': 'POST',
                    'url': '/api/train',
                    'description': 'Train new sentiment model'
                },
                'models': {
                    'method': 'GET',
                    'url': '/api/models',
                    'description': 'List available models'
                },
                'model_metrics': {
                    'method': 'POST',
                    'url': '/api/models/metrics',
                    'description': 'Get model performance metrics'
                }
            },
            'documentation': {
                'api_docs': '/api/status',
                'health_check': '/api/health',
                'github': 'https://github.com/your-repo/sentiment-analysis-api'
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'message': 'The requested endpoint does not exist',
            'available_endpoints': '/api/status'
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'error': 'Method not allowed',
            'message': 'The HTTP method is not allowed for this endpoint'
        }), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    return app

def initialize_models():
    """Initialize and load models on startup."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import here to avoid circular imports
        from services.sentiment_service import SentimentAnalysisService
        from config.settings import ModelConfig
        
        logger.info("Initializing sentiment analysis service...")
        service = SentimentAnalysisService()
        
        # Try to load existing models
        model_files = [
            ('basic', ModelConfig.DEFAULT_MODEL_PATH),
            ('enhanced', ModelConfig.ENHANCED_MODEL_PATH),
            ('ultimate', ModelConfig.ULTIMATE_MODEL_PATH)
        ]
        
        models_loaded = 0
        for model_name, model_path in model_files:
            if model_path.exists():
                try:
                    success = service.load_model(model_name, str(model_path))
                    if success:
                        models_loaded += 1
                        logger.info(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} model: {e}")
        
        if models_loaded == 0:
            logger.warning("No pre-trained models found. Models will be trained on first use.")
        else:
            logger.info(f"Successfully loaded {models_loaded} models")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return False

def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Sentiment Analysis API v2.0")
    logger.info("=" * 50)
    
    # Initialize models
    if initialize_models():
        logger.info("✅ Models initialized successfully")
    else:
        logger.warning("⚠️  Model initialization had issues")
    
    # Create Flask app
    app = create_app()
    
    # Print startup information
    logger.info(f"🌐 API will run on: http://{APIConfig.HOST}:{APIConfig.PORT}")
    logger.info(f"📚 API documentation: http://{APIConfig.HOST}:{APIConfig.PORT}")
    logger.info(f"🔍 Health check: http://{APIConfig.HOST}:{APIConfig.PORT}/api/health")
    logger.info(f"🐛 Debug mode: {APIConfig.DEBUG}")
    
    # Run the application
    try:
        app.run(
            host=APIConfig.HOST,
            port=APIConfig.PORT,
            debug=APIConfig.DEBUG,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
