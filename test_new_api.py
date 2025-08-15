#!/usr/bin/env python3
"""
Test Client for New Structured API
Demonstrates the clean architecture and functionality.
"""

import requests
import json
import time
from typing import Dict, Any


class StructuredAPIClient:
    """Client for testing the new structured API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get API status."""
        try:
            response = self.session.get(f"{self.base_url}/api/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def train_model(self, texts: list, labels: list, model_name: str = "basic") -> Dict[str, Any]:
        """Train a model."""
        try:
            data = {
                "texts": texts,
                "labels": labels,
                "model_name": model_name,
                "model_type": "logistic"
            }
            response = self.session.post(f"{self.base_url}/api/train", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_single(self, text: str, model_name: str = "basic") -> Dict[str, Any]:
        """Predict sentiment for single text."""
        try:
            data = {
                "text": text,
                "model_name": model_name
            }
            response = self.session.post(f"{self.base_url}/api/predict", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, texts: list, model_name: str = "basic") -> Dict[str, Any]:
        """Predict sentiment for multiple texts."""
        try:
            data = {
                "texts": texts,
                "model_name": model_name
            }
            response = self.session.post(f"{self.base_url}/api/predict/batch", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def test_structured_api():
    """Test the new structured API."""
    print("🚀 Testing New Structured Sentiment Analysis API v2.0")
    print("=" * 60)
    
    client = StructuredAPIClient()
    
    # Test 1: Health Check
    print("\n1. 🔍 Health Check")
    health = client.health_check()
    if "error" in health:
        print(f"❌ Health check failed: {health['error']}")
        return
    else:
        print(f"✅ API is healthy (v{health.get('version', 'unknown')})")
    
    # Test 2: Status Check
    print("\n2. 📊 API Status")
    status = client.get_status()
    if "error" in status:
        print(f"❌ Status check failed: {status['error']}")
    else:
        print(f"✅ API Status: {status['status']}")
        print(f"   Version: {status['version']}")
        print(f"   Available models: {list(status['models'].keys())}")
        print(f"   Max batch size: {status['configuration']['max_batch_size']}")
    
    # Test 3: Train a Model
    print("\n3. 🤖 Model Training")
    training_texts = [
        "This is absolutely amazing!", "I love this product!", "Fantastic quality!",
        "Terrible experience", "I hate this", "Worst product ever",
        "It's okay", "Average quality", "Nothing special", "Decent enough"
    ]
    training_labels = [
        "positive", "positive", "positive",
        "negative", "negative", "negative", 
        "neutral", "neutral", "neutral", "neutral"
    ]
    
    print("Training basic model with sample data...")
    train_result = client.train_model(training_texts, training_labels, "basic")
    
    if "error" in train_result:
        print(f"❌ Training failed: {train_result['error']}")
    else:
        print(f"✅ Model trained successfully!")
        print(f"   Accuracy: {train_result['training_result']['accuracy']:.3f}")
        print(f"   Training samples: {train_result['training_result']['training_samples']}")
        print(f"   Test samples: {train_result['training_result']['test_samples']}")
    
    # Test 4: Single Prediction
    print("\n4. 🎯 Single Text Prediction")
    test_texts = [
        "This product is absolutely incredible!",
        "Worst purchase I've ever made",
        "It's fine, nothing special"
    ]
    
    for text in test_texts:
        result = client.predict_single(text, "basic")
        if "error" in result:
            print(f"❌ Prediction failed: {result['error']}")
        else:
            print(f"Text: \"{text[:40]}...\"")
            print(f"   Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
    
    # Test 5: Batch Prediction
    print("\n5. 📦 Batch Prediction")
    batch_texts = [
        "Excellent service!",
        "Poor quality product",
        "Average experience",
        "Outstanding performance!",
        "Not worth the money"
    ]
    
    batch_result = client.predict_batch(batch_texts, "basic")
    if "error" in batch_result:
        print(f"❌ Batch prediction failed: {batch_result['error']}")
    else:
        print(f"✅ Processed {batch_result['total_processed']} texts")
        for item in batch_result['results']:
            if 'error' not in item:
                print(f"   \"{item['text'][:30]}...\" → {item['sentiment']} ({item['confidence']:.3f})")
    
    # Test 6: List Models
    print("\n6. 📋 Model Information")
    models = client.list_models()
    if "error" in models:
        print(f"❌ Model listing failed: {models['error']}")
    else:
        print(f"✅ Available models: {models['total_models']}")
        for name, info in models['models'].items():
            status_icon = "✅" if info['is_trained'] else "⏳"
            print(f"   {status_icon} {name}: trained={info['is_trained']}")
    
    print("\n🎉 Structured API testing completed!")
    print("\n📚 Architecture Benefits Demonstrated:")
    print("   ✅ Clean separation of concerns")
    print("   ✅ Modular design with services")
    print("   ✅ Comprehensive error handling")
    print("   ✅ Structured configuration")
    print("   ✅ Professional API responses")


def demonstrate_architecture():
    """Demonstrate the clean architecture."""
    print("\n🏗️ **NEW STRUCTURED ARCHITECTURE**")
    print("=" * 50)
    
    print("""
📁 Project Structure:
├── api/                    # 🌐 API Layer (Clean endpoints)
│   └── routes.py          # RESTful routes with validation
├── services/              # 💼 Business Logic Layer  
│   ├── sentiment_service.py  # Core sentiment analysis
│   └── youtube_service.py    # YouTube integration
├── models/                # 🤖 Model Layer
│   └── base_model.py      # Abstract model interfaces
├── config/                # ⚙️ Configuration Management
│   └── settings.py        # Centralized configuration
├── utils/                 # 🛠️ Utility Functions
│   └── data_loader.py     # Data loading utilities
├── tests/                 # 🧪 Test Suite
│   └── test_api.py        # Comprehensive API tests
└── app.py                 # 🚀 Main Flask Application

🎯 **Key Improvements:**
✅ Separation of Concerns    - API, Business Logic, Models separated
✅ Modular Design           - Easy to extend and maintain
✅ Configuration Management - Centralized settings
✅ Error Handling          - Comprehensive error responses
✅ Testing Framework       - Built-in test suite
✅ Documentation          - Self-documenting API
✅ Scalability            - Clean architecture for growth
✅ Maintainability        - Professional code organization
    """)


if __name__ == "__main__":
    demonstrate_architecture()
    test_structured_api()
