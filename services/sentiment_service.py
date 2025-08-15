#!/usr/bin/env python3
"""
Sentiment Analysis Service
Core business logic for sentiment analysis operations.
Separated from API layer for better maintainability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from models.base_model import BaseSentimentModel, ModelMetrics
from config.settings import ModelConfig

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not found, using empty set")
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_process(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize and process text."""
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback if NLTK punkt tokenizer not available
            tokens = text.split()
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize tokens
        try:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        except LookupError:
            # Fallback if WordNet not available
            pass
        
        return tokens
    
    def preprocess_text(self, text: str, join_tokens: bool = True) -> str:
        """Complete text preprocessing pipeline."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and process
        tokens = self.tokenize_and_process(cleaned_text)
        
        # Return as string or list of tokens
        return ' '.join(tokens) if join_tokens else tokens


class BasicSentimentModel(BaseSentimentModel):
    """Basic sentiment analysis model implementation."""
    
    def __init__(self, model_name: str = "basic_sentiment_model"):
        super().__init__(model_name)
        self.preprocessor = TextPreprocessor()
        
    def train(self, texts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
        """Train the basic sentiment model."""
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
        
        logger.info(f"Training {self.model_name} with {len(texts)} samples")
        
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        
        # Encode labels
        encoded_labels = [self.label_encoder.get(label.lower(), 1) for label in labels]  # Default to neutral
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, encoded_labels, 
            test_size=kwargs.get('test_size', 0.2), 
            random_state=kwargs.get('random_state', 42),
            stratify=encoded_labels
        )
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=kwargs.get('max_features', ModelConfig.MAX_FEATURES),
            ngram_range=kwargs.get('ngram_range', ModelConfig.NGRAM_RANGE),
            min_df=kwargs.get('min_df', ModelConfig.MIN_DF),
            max_df=kwargs.get('max_df', ModelConfig.MAX_DF),
            stop_words='english'
        )
        
        # Fit vectorizer and transform data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        model_type = kwargs.get('model_type', 'logistic')
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='linear',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Fit model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store training info
        self.training_info = {
            'model_type': model_type,
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train_vec.shape[1],
            'parameters': kwargs
        }
        
        self._is_trained = True
        self.last_trained_at = datetime.now()
        
        logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_type': model_type
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        if not self.is_trained():
            raise ValueError("Model is not trained")
        
        if not self._validate_input(text):
            raise ValueError("Invalid input text")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        
        return self._prepare_prediction_result(text, prediction, probabilities)


class EnhancedSentimentModel(BaseSentimentModel):
    """Enhanced sentiment analysis model with advanced features."""
    
    def __init__(self, model_name: str = "enhanced_sentiment_model"):
        super().__init__(model_name)
        self.preprocessor = TextPreprocessor()
        self.ensemble_model = None
        
    def _extract_advanced_features(self, texts: List[str]) -> np.ndarray:
        """Extract advanced text features."""
        features = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            # Basic statistics
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Punctuation features
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_count = sum(1 for c in text if c.isupper())
            caps_ratio = caps_count / max(char_count, 1)
            
            # Sentiment indicators
            positive_words = ['good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'like', 'best']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting']
            
            positive_count = sum(1 for word in positive_words if word in text.lower())
            negative_count = sum(1 for word in negative_words if word in text.lower())
            
            # Average word length
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            feature_vector = [
                char_count, word_count, sentence_count,
                exclamation_count, question_count, caps_ratio,
                positive_count, negative_count, avg_word_length
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, texts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
        """Train the enhanced sentiment model."""
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
        
        logger.info(f"Training {self.model_name} with {len(texts)} samples")
        
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        
        # Encode labels
        encoded_labels = [self.label_encoder.get(label.lower(), 1) for label in labels]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, encoded_labels,
            test_size=kwargs.get('test_size', 0.2),
            random_state=kwargs.get('random_state', 42),
            stratify=encoded_labels
        )
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=kwargs.get('max_features', 15000),
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True
        )
        
        # Fit vectorizer and transform data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Extract advanced features
        X_train_advanced = self._extract_advanced_features([texts[i] for i in range(len(texts)) if i < len(X_train)])
        X_test_advanced = self._extract_advanced_features([texts[i] for i in range(len(texts)) if i >= len(X_train)])
        
        # Combine features
        from scipy.sparse import hstack
        X_train_combined = hstack([X_train_tfidf, X_train_advanced])
        X_test_combined = hstack([X_test_tfidf, X_test_advanced])
        
        # Create ensemble model
        estimators = [
            ('logistic', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
            ('svm', SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ]
        
        self.ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
        self.model = self.ensemble_model  # For compatibility
        
        # Train ensemble
        self.ensemble_model.fit(X_train_combined, y_train)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store training info
        self.training_info = {
            'model_type': 'enhanced_ensemble',
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'tfidf_features': X_train_tfidf.shape[1],
            'advanced_features': X_train_advanced.shape[1],
            'total_features': X_train_combined.shape[1],
            'estimators': [name for name, _ in estimators]
        }
        
        self._is_trained = True
        self.last_trained_at = datetime.now()
        
        logger.info(f"Enhanced model trained successfully. Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_type': 'enhanced_ensemble'
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment using enhanced model."""
        if not self.is_trained():
            raise ValueError("Model is not trained")
        
        if not self._validate_input(text):
            raise ValueError("Invalid input text")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # TF-IDF features
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Advanced features
        text_advanced = self._extract_advanced_features([text])
        
        # Combine features
        from scipy.sparse import hstack
        text_combined = hstack([text_tfidf, text_advanced])
        
        # Predict
        prediction = self.ensemble_model.predict(text_combined)[0]
        probabilities = self.ensemble_model.predict_proba(text_combined)[0]
        
        return self._prepare_prediction_result(text, prediction, probabilities)


class SentimentAnalysisService:
    """Main service class for sentiment analysis operations."""
    
    def __init__(self):
        self.models: Dict[str, BaseSentimentModel] = {}
        self.default_model_name = 'basic'
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models."""
        # Basic model
        basic_model = BasicSentimentModel('basic')
        self.models['basic'] = basic_model
        
        # Enhanced model
        enhanced_model = EnhancedSentimentModel('enhanced')
        self.models['enhanced'] = enhanced_model
        
        logger.info("Sentiment analysis service initialized")
    
    def get_model(self, model_name: Optional[str] = None) -> BaseSentimentModel:
        """Get a model by name."""
        if model_name is None:
            model_name = self.default_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.models[model_name]
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models."""
        return {
            name: model.get_model_info()
            for name, model in self.models.items()
        }
    
    def predict_sentiment(self, text: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        model = self.get_model(model_name)
        
        if not model.is_trained():
            raise ValueError(f"Model '{model.model_name}' is not trained")
        
        return model.predict(text)
    
    def predict_batch(self, texts: List[str], model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts."""
        model = self.get_model(model_name)
        
        if not model.is_trained():
            raise ValueError(f"Model '{model.model_name}' is not trained")
        
        return model.predict_batch(texts)
    
    def train_model(self, texts: List[str], labels: List[str], model_name: str = 'basic', **kwargs) -> Dict[str, Any]:
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        return model.train(texts, labels, **kwargs)
    
    def save_model(self, model_name: str, filepath: str) -> bool:
        """Save a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        return model.save_model(filepath)
    
    def load_model(self, model_name: str, filepath: str) -> bool:
        """Load a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        return model.load_model(filepath)
    
    def get_model_metrics(self, model_name: str, test_texts: List[str], test_labels: List[str]) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        model = self.get_model(model_name)
        
        if not model.is_trained():
            raise ValueError(f"Model '{model_name}' is not trained")
        
        # Get predictions
        predictions = model.predict_batch(test_texts)
        predicted_labels = [pred['sentiment'] for pred in predictions]
        
        # Calculate metrics
        accuracy = ModelMetrics.calculate_accuracy(test_labels, predicted_labels)
        detailed_metrics = ModelMetrics.calculate_precision_recall_f1(test_labels, predicted_labels)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'detailed_metrics': detailed_metrics,
            'test_samples': len(test_texts)
        }
