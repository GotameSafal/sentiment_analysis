#!/usr/bin/env python3
"""
Configuration settings for Sentiment Analysis API
Centralized configuration management for the entire application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Configuration
class APIConfig:
    """API-specific configuration."""
    HOST = os.getenv('API_HOST', '0.0.0.0')
    PORT = int(os.getenv('API_PORT', 5000))
    DEBUG = os.getenv('API_DEBUG', 'True').lower() == 'true'
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'False').lower() == 'true'
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 100))
    
    # Request limits
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 100))
    MAX_YOUTUBE_COMMENTS = int(os.getenv('MAX_YOUTUBE_COMMENTS', 500))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 120))

# Model Configuration
class ModelConfig:
    """Model-specific configuration."""
    # Model paths
    MODELS_DIR = BASE_DIR / 'models'
    DEFAULT_MODEL_PATH = MODELS_DIR / 'sentiment_model.pkl'
    ENHANCED_MODEL_PATH = MODELS_DIR / 'enhanced_sentiment_model.pkl'
    ULTIMATE_MODEL_PATH = MODELS_DIR / 'ultimate_sentiment_model.pkl'
    
    # Model settings
    DEFAULT_MODEL_TYPE = os.getenv('DEFAULT_MODEL_TYPE', 'basic')
    AUTO_LOAD_BEST_MODEL = os.getenv('AUTO_LOAD_BEST_MODEL', 'True').lower() == 'true'
    
    # Training settings
    DEFAULT_DATASET_SIZE = int(os.getenv('DEFAULT_DATASET_SIZE', 1000))
    MAX_TRAINING_SIZE = int(os.getenv('MAX_TRAINING_SIZE', 10000))
    
    # Feature extraction
    MAX_FEATURES = int(os.getenv('MAX_FEATURES', 10000))
    NGRAM_RANGE = (1, 2)  # Can be configured via environment if needed
    MIN_DF = int(os.getenv('MIN_DF', 2))
    MAX_DF = float(os.getenv('MAX_DF', 0.95))

# YouTube Configuration
class YouTubeConfig:
    """YouTube API configuration."""
    API_KEY = os.getenv('YOUTUBE_API_KEY', '')
    DEFAULT_MAX_COMMENTS = int(os.getenv('YOUTUBE_DEFAULT_MAX_COMMENTS', 100))
    REQUEST_TIMEOUT = int(os.getenv('YOUTUBE_REQUEST_TIMEOUT', 30))

    # Validation
    @classmethod
    def is_configured(cls):
        """Check if YouTube API is properly configured."""
        return bool(cls.API_KEY and cls.API_KEY != 'your_youtube_api_key_here')
    
    # Rate limiting for YouTube API
    REQUESTS_PER_SECOND = float(os.getenv('YOUTUBE_REQUESTS_PER_SECOND', 1.0))
    QUOTA_LIMIT_PER_DAY = int(os.getenv('YOUTUBE_QUOTA_LIMIT', 10000))

# Logging Configuration
class LoggingConfig:
    """Logging configuration."""
    LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File logging
    LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'False').lower() == 'true'
    LOG_FILE_PATH = BASE_DIR / 'logs' / 'sentiment_api.log'
    MAX_LOG_SIZE = int(os.getenv('MAX_LOG_SIZE', 10 * 1024 * 1024))  # 10MB
    BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 5))

# Database Configuration (for future use)
class DatabaseConfig:
    """Database configuration for storing results/analytics."""
    ENABLED = os.getenv('DATABASE_ENABLED', 'False').lower() == 'true'
    URL = os.getenv('DATABASE_URL', 'sqlite:///sentiment_analysis.db')
    
    # Connection settings
    POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 5))
    MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', 10))
    POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', 30))

# Cache Configuration
class CacheConfig:
    """Caching configuration."""
    ENABLED = os.getenv('CACHE_ENABLED', 'False').lower() == 'true'
    TYPE = os.getenv('CACHE_TYPE', 'memory')  # memory, redis, memcached
    
    # Redis settings (if using Redis)
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # Cache TTL
    DEFAULT_TTL = int(os.getenv('CACHE_DEFAULT_TTL', 3600))  # 1 hour
    MODEL_PREDICTION_TTL = int(os.getenv('CACHE_PREDICTION_TTL', 300))  # 5 minutes

# Security Configuration
class SecurityConfig:
    """Security configuration."""
    # API Keys (for future authentication)
    API_KEY_REQUIRED = os.getenv('API_KEY_REQUIRED', 'False').lower() == 'true'
    VALID_API_KEYS = os.getenv('VALID_API_KEYS', '').split(',') if os.getenv('VALID_API_KEYS') else []
    
    # Request validation
    MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', 1024 * 1024))  # 1MB
    ALLOWED_CONTENT_TYPES = ['application/json']
    
    # CORS security
    CORS_MAX_AGE = int(os.getenv('CORS_MAX_AGE', 86400))  # 24 hours

# Performance Configuration
class PerformanceConfig:
    """Performance and optimization settings."""
    # Threading
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
    WORKER_TIMEOUT = int(os.getenv('WORKER_TIMEOUT', 120))
    
    # Memory management
    MAX_MEMORY_USAGE = int(os.getenv('MAX_MEMORY_USAGE', 1024 * 1024 * 1024))  # 1GB
    GARBAGE_COLLECTION_THRESHOLD = int(os.getenv('GC_THRESHOLD', 100))
    
    # Model loading
    PRELOAD_MODELS = os.getenv('PRELOAD_MODELS', 'True').lower() == 'true'
    MODEL_CACHE_SIZE = int(os.getenv('MODEL_CACHE_SIZE', 3))

# Environment-specific configurations
class DevelopmentConfig:
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'

class ProductionConfig:
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'INFO'
    RATE_LIMIT_ENABLED = True

class TestingConfig:
    """Testing environment configuration."""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'

# Configuration factory
def get_config():
    """Get configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Export all configurations
__all__ = [
    'APIConfig',
    'ModelConfig', 
    'YouTubeConfig',
    'LoggingConfig',
    'DatabaseConfig',
    'CacheConfig',
    'SecurityConfig',
    'PerformanceConfig',
    'get_config',
    'BASE_DIR'
]
