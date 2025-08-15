#!/usr/bin/env python3
"""
Base Model Classes for Sentiment Analysis
Abstract base classes and interfaces for sentiment analysis models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pickle
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SentimentModelInterface(ABC):
    """Abstract interface for sentiment analysis models."""
    
    @abstractmethod
    def train(self, texts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
        """Train the model with given texts and labels."""
        pass
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to file."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from file."""
        pass
    
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the model is trained and ready for predictions."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class BaseSentimentModel(SentimentModelInterface):
    """Base implementation of sentiment analysis model."""
    
    def __init__(self, model_name: str = "base_model"):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.label_encoder = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self._is_trained = False
        self.training_info = {}
        self.created_at = datetime.now()
        self.last_trained_at = None
        
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': self.model_name,
            'is_trained': self.is_trained(),
            'created_at': self.created_at.isoformat(),
            'last_trained_at': self.last_trained_at.isoformat() if self.last_trained_at else None,
            'training_info': self.training_info,
            'label_encoder': self.label_encoder,
            'model_type': type(self.model).__name__ if self.model else None
        }
    
    def _validate_input(self, text: Union[str, List[str]]) -> bool:
        """Validate input text(s)."""
        if isinstance(text, str):
            return len(text.strip()) > 0
        elif isinstance(text, list):
            return len(text) > 0 and all(isinstance(t, str) and len(t.strip()) > 0 for t in text)
        return False
    
    def _prepare_prediction_result(self, text: str, prediction: int, probabilities: List[float]) -> Dict[str, Any]:
        """Prepare standardized prediction result."""
        sentiment = self.label_decoder.get(prediction, 'unknown')
        confidence = max(probabilities) if probabilities else 0.0
        
        prob_dict = {
            'negative': probabilities[0] if len(probabilities) > 0 else 0.0,
            'neutral': probabilities[1] if len(probabilities) > 1 else 0.0,
            'positive': probabilities[2] if len(probabilities) > 2 else 0.0
        }
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save model to file."""
        try:
            if not self.is_trained():
                logger.error("Cannot save untrained model")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'label_decoder': self.label_decoder,
                'model_name': self.model_name,
                'training_info': self.training_info,
                'created_at': self.created_at,
                'last_trained_at': self.last_trained_at,
                'version': '1.0'
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from file."""
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load model components
            self.model = model_data.get('model')
            self.vectorizer = model_data.get('vectorizer')
            self.label_encoder = model_data.get('label_encoder', self.label_encoder)
            self.label_decoder = model_data.get('label_decoder', self.label_decoder)
            self.model_name = model_data.get('model_name', self.model_name)
            self.training_info = model_data.get('training_info', {})
            self.created_at = model_data.get('created_at', self.created_at)
            self.last_trained_at = model_data.get('last_trained_at')
            
            self._is_trained = self.model is not None
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Default batch prediction implementation."""
        if not self.is_trained():
            raise ValueError("Model is not trained")
        
        if not self._validate_input(texts):
            raise ValueError("Invalid input texts")
        
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text '{text[:50]}...': {e}")
                results.append({
                    'text': text,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results


class ModelManager:
    """Manages multiple sentiment analysis models."""
    
    def __init__(self):
        self.models: Dict[str, BaseSentimentModel] = {}
        self.default_model: Optional[str] = None
        
    def register_model(self, name: str, model: BaseSentimentModel, set_as_default: bool = False):
        """Register a model with the manager."""
        self.models[name] = model
        if set_as_default or self.default_model is None:
            self.default_model = name
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: Optional[str] = None) -> Optional[BaseSentimentModel]:
        """Get a model by name or return default."""
        if name is None:
            name = self.default_model
        return self.models.get(name)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models and their info."""
        return {
            name: model.get_model_info() 
            for name, model in self.models.items()
        }
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the manager."""
        if name in self.models:
            del self.models[name]
            if self.default_model == name:
                self.default_model = next(iter(self.models.keys())) if self.models else None
            logger.info(f"Removed model: {name}")
            return True
        return False
    
    def set_default_model(self, name: str) -> bool:
        """Set the default model."""
        if name in self.models:
            self.default_model = name
            logger.info(f"Set default model: {name}")
            return True
        return False


# Global model manager instance
model_manager = ModelManager()


class ModelMetrics:
    """Model performance metrics and evaluation."""
    
    @staticmethod
    def calculate_accuracy(y_true: List[str], y_pred: List[str]) -> float:
        """Calculate accuracy score."""
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted labels must have same length")
        
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true) if len(y_true) > 0 else 0.0
    
    @staticmethod
    def calculate_precision_recall_f1(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and F1 score for each class."""
        from collections import defaultdict
        
        # Count true positives, false positives, false negatives
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                tp[true] += 1
            else:
                fp[pred] += 1
                fn[true] += 1
        
        # Calculate metrics for each class
        metrics = {}
        for label in set(y_true + y_pred):
            precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return metrics
