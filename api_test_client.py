#!/usr/bin/env python3
"""
Sentiment Analysis API Test Client
Test and demonstrate the REST API functionality.
"""

import requests
import json
import time
from typing import Dict, List, Any


class SentimentAPIClient:
    """Client for testing the Sentiment Analysis API."""
    
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
        """Get API status and model info."""
        try:
            response = self.session.get(f"{self.base_url}/api/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_single(self, text: str, use_enhanced: bool = False) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        try:
            data = {
                "text": text,
                "use_enhanced": use_enhanced
            }
            response = self.session.post(f"{self.base_url}/api/predict", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, texts: List[str], use_enhanced: bool = False) -> Dict[str, Any]:
        """Predict sentiment for multiple texts."""
        try:
            data = {
                "texts": texts,
                "use_enhanced": use_enhanced
            }
            response = self.session.post(f"{self.base_url}/api/predict/batch", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_youtube(self, video_url: str, api_key: str, max_comments: int = 50, use_enhanced: bool = False) -> Dict[str, Any]:
        """Analyze YouTube video comments."""
        try:
            data = {
                "video_url": video_url,
                "api_key": api_key,
                "max_comments": max_comments,
                "use_enhanced": use_enhanced
            }
            response = self.session.post(f"{self.base_url}/api/youtube/analyze", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def train_model(self, model_type: str = "basic", dataset_size: int = 1000) -> Dict[str, Any]:
        """Train a new model."""
        try:
            data = {
                "model_type": model_type,
                "dataset_size": dataset_size
            }
            response = self.session.post(f"{self.base_url}/api/train", json=data)
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


def test_api():
    """Test all API endpoints."""
    print("🚀 Testing Sentiment Analysis API")
    print("=" * 50)
    
    client = SentimentAPIClient()
    
    # Test 1: Health Check
    print("\n1. 🔍 Health Check")
    health = client.health_check()
    if "error" in health:
        print(f"❌ Health check failed: {health['error']}")
        return
    else:
        print(f"✅ API is healthy: {health['status']}")
    
    # Test 2: Status Check
    print("\n2. 📊 Status Check")
    status = client.get_status()
    if "error" in status:
        print(f"❌ Status check failed: {status['error']}")
    else:
        print(f"✅ API status: {status['status']}")
        print(f"   Basic model: {status['models']['basic_model']}")
        print(f"   Enhanced model: {status['models']['enhanced_model']}")
        print(f"   Available models: {len(status['models']['available_models'])}")
    
    # Test 3: Single Text Prediction
    print("\n3. 🎯 Single Text Prediction")
    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it!",
        "This is the worst film I've ever seen. Complete waste of time.",
        "It was okay, nothing special but not terrible either."
    ]
    
    for text in test_texts:
        result = client.predict_single(text)
        if "error" in result:
            print(f"❌ Prediction failed: {result['error']}")
        else:
            print(f"Text: \"{text[:50]}...\"")
            print(f"   Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            print(f"   Probabilities: {result['probabilities']}")
            print()
    
    # Test 4: Batch Prediction
    print("\n4. 📦 Batch Prediction")
    batch_texts = [
        "Amazing product, highly recommend!",
        "Terrible service, very disappointed",
        "Average quality, nothing special",
        "Outstanding performance!",
        "Could be better"
    ]
    
    batch_result = client.predict_batch(batch_texts)
    if "error" in batch_result:
        print(f"❌ Batch prediction failed: {batch_result['error']}")
    else:
        print(f"✅ Processed {batch_result['total_processed']} texts")
        print(f"   Model used: {batch_result['model_used']}")
        
        for item in batch_result['results']:
            if 'error' not in item:
                print(f"   \"{item['text'][:30]}...\" → {item['sentiment']} ({item['confidence']:.3f})")
    
    # Test 5: Model Listing
    print("\n5. 📋 List Models")
    models = client.list_models()
    if "error" in models:
        print(f"❌ Model listing failed: {models['error']}")
    else:
        print(f"✅ Available models:")
        for model in models['models']['available_models']:
            print(f"   {model['name']} ({model['size']} bytes)")
    
    # Test 6: YouTube Analysis (optional - requires API key)
    print("\n6. 🎬 YouTube Analysis (Optional)")
    youtube_api_key = input("Enter YouTube API key (or press Enter to skip): ").strip()
    
    if youtube_api_key:
        video_url = input("Enter YouTube video URL: ").strip()
        if video_url:
            print("Analyzing YouTube video...")
            youtube_result = client.analyze_youtube(video_url, youtube_api_key, max_comments=20)
            
            if "error" in youtube_result:
                print(f"❌ YouTube analysis failed: {youtube_result['error']}")
            else:
                print(f"✅ Analyzed {youtube_result['total_comments']} comments")
                print(f"   Sentiment distribution:")
                for sentiment, percentage in youtube_result['sentiment_percentages'].items():
                    print(f"     {sentiment}: {percentage:.1f}%")
                
                # Show a few sample comments
                print(f"   Sample comments:")
                for comment in youtube_result['comments'][:3]:
                    print(f"     \"{comment['text'][:50]}...\" → {comment['sentiment']}")
    else:
        print("⏭️  Skipping YouTube analysis (no API key provided)")
    
    # Test 7: Model Training (optional)
    print("\n7. 🤖 Model Training (Optional)")
    train_new = input("Train a new model? (y/n): ").strip().lower()
    
    if train_new == 'y':
        print("Training new model...")
        train_result = client.train_model(model_type="basic", dataset_size=500)
        
        if "error" in train_result:
            print(f"❌ Model training failed: {train_result['error']}")
        else:
            print(f"✅ Model trained successfully!")
            print(f"   Accuracy: {train_result['accuracy']:.3f}")
            print(f"   Model file: {train_result['model_file']}")
    else:
        print("⏭️  Skipping model training")
    
    print("\n🎉 API testing completed!")


def demo_api_usage():
    """Demonstrate practical API usage."""
    print("🎯 Sentiment Analysis API Demo")
    print("=" * 40)
    
    client = SentimentAPIClient()
    
    # Check if API is running
    health = client.health_check()
    if "error" in health:
        print(f"❌ API not available: {health['error']}")
        print("Make sure to run: python sentiment_api.py")
        return
    
    print("✅ API is running!")
    
    # Demo 1: Product Review Analysis
    print("\n📝 Demo 1: Product Review Analysis")
    reviews = [
        "This product exceeded my expectations! Amazing quality and fast shipping.",
        "Terrible quality, broke after one day. Don't waste your money.",
        "It's okay, does what it's supposed to do but nothing special.",
        "Outstanding customer service and great product. Highly recommended!",
        "Average product, overpriced for what you get."
    ]
    
    batch_result = client.predict_batch(reviews)
    if "error" not in batch_result:
        positive_count = sum(1 for r in batch_result['results'] if r.get('sentiment') == 'positive')
        negative_count = sum(1 for r in batch_result['results'] if r.get('sentiment') == 'negative')
        neutral_count = sum(1 for r in batch_result['results'] if r.get('sentiment') == 'neutral')
        
        print(f"📊 Review Analysis Results:")
        print(f"   Positive: {positive_count}/{len(reviews)} ({positive_count/len(reviews)*100:.1f}%)")
        print(f"   Negative: {negative_count}/{len(reviews)} ({negative_count/len(reviews)*100:.1f}%)")
        print(f"   Neutral: {neutral_count}/{len(reviews)} ({neutral_count/len(reviews)*100:.1f}%)")
    
    # Demo 2: Social Media Monitoring
    print("\n📱 Demo 2: Social Media Monitoring")
    social_posts = [
        "Just watched the new movie, absolutely loved it! #MovieNight",
        "Worst customer service ever. Never shopping here again. #Disappointed",
        "New restaurant in town is pretty good, worth trying",
        "Can't believe how amazing this concert was! Best night ever! 🎵",
        "Meh, another boring day at work"
    ]
    
    for post in social_posts:
        result = client.predict_single(post)
        if "error" not in result:
            emoji = "😊" if result['sentiment'] == 'positive' else "😞" if result['sentiment'] == 'negative' else "😐"
            print(f"   {emoji} \"{post[:40]}...\" → {result['sentiment']} ({result['confidence']:.2f})")
    
    print("\n✨ Demo completed! The API is ready for production use.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_api_usage()
    else:
        test_api()
