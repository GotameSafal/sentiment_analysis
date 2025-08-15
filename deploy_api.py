#!/usr/bin/env python3
"""
Production Deployment Script for Sentiment Analysis API
Handles model preparation, API deployment, and health monitoring.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path


class APIDeployer:
    """Handle API deployment and management."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.venv_path = self.base_dir / "venv"
        self.api_url = "http://localhost:5000"
        
    def check_environment(self):
        """Check if environment is properly set up."""
        print("🔍 Checking environment...")
        
        # Check virtual environment
        if not self.venv_path.exists():
            print("❌ Virtual environment not found")
            return False
        
        # Check Python version
        try:
            result = subprocess.run([
                str(self.venv_path / "bin" / "python"), "--version"
            ], capture_output=True, text=True)
            print(f"✅ Python version: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ Python check failed: {e}")
            return False
        
        # Check required packages
        required_packages = ['flask', 'scikit-learn', 'pandas', 'numpy']
        try:
            for package in required_packages:
                result = subprocess.run([
                    str(self.venv_path / "bin" / "pip"), "show", package
                ], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ Package {package} not installed")
                    return False
            print("✅ All required packages installed")
        except Exception as e:
            print(f"❌ Package check failed: {e}")
            return False
        
        return True
    
    def prepare_models(self):
        """Prepare models for production."""
        print("🤖 Preparing models...")
        
        # Check if any model exists
        model_files = [
            "sentiment_model.pkl",
            "enhanced_sentiment_model.pkl",
            "ultimate_sentiment_model.pkl"
        ]
        
        existing_models = [f for f in model_files if os.path.exists(f)]
        
        if existing_models:
            print(f"✅ Found existing models: {existing_models}")
            return True
        
        # Train a basic model if none exists
        print("⚠️  No models found. Training basic model...")
        try:
            result = subprocess.run([
                str(self.venv_path / "bin" / "python"),
                "train_with_youtube_dataset.py"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Basic model trained successfully")
                return True
            else:
                print(f"❌ Model training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Model training timed out")
            return False
        except Exception as e:
            print(f"❌ Model training error: {e}")
            return False
    
    def start_api_development(self):
        """Start API in development mode."""
        print("🚀 Starting API in development mode...")
        
        try:
            # Start the API
            process = subprocess.Popen([
                str(self.venv_path / "bin" / "python"),
                "sentiment_api.py"
            ])
            
            # Wait for API to start
            print("⏳ Waiting for API to start...")
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{self.api_url}/api/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ API started successfully!")
                        print(f"🌐 API running at: {self.api_url}")
                        print(f"📚 Documentation: {self.api_url}")
                        print(f"🔍 Health check: {self.api_url}/api/health")
                        return process
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
            
            print("❌ API failed to start within 30 seconds")
            process.terminate()
            return None
            
        except Exception as e:
            print(f"❌ Failed to start API: {e}")
            return None
    
    def start_api_production(self, workers=4, port=5000):
        """Start API in production mode with Gunicorn."""
        print(f"🚀 Starting API in production mode (workers: {workers}, port: {port})...")
        
        try:
            # Start with Gunicorn
            process = subprocess.Popen([
                str(self.venv_path / "bin" / "gunicorn"),
                "-w", str(workers),
                "-b", f"0.0.0.0:{port}",
                "--timeout", "120",
                "--keep-alive", "5",
                "--max-requests", "1000",
                "--max-requests-jitter", "100",
                "sentiment_api:app"
            ])
            
            # Wait for API to start
            print("⏳ Waiting for API to start...")
            api_url = f"http://localhost:{port}"
            
            for i in range(30):
                try:
                    response = requests.get(f"{api_url}/api/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ Production API started successfully!")
                        print(f"🌐 API running at: {api_url}")
                        print(f"👥 Workers: {workers}")
                        print(f"🔧 Production mode with Gunicorn")
                        return process
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
            
            print("❌ Production API failed to start within 30 seconds")
            process.terminate()
            return None
            
        except Exception as e:
            print(f"❌ Failed to start production API: {e}")
            return None
    
    def test_api(self):
        """Test API endpoints."""
        print("🧪 Testing API endpoints...")
        
        try:
            # Test health check
            response = requests.get(f"{self.api_url}/api/health")
            if response.status_code == 200:
                print("✅ Health check: OK")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
            
            # Test status
            response = requests.get(f"{self.api_url}/api/status")
            if response.status_code == 200:
                print("✅ Status endpoint: OK")
            else:
                print(f"❌ Status endpoint failed: {response.status_code}")
                return False
            
            # Test prediction
            response = requests.post(f"{self.api_url}/api/predict", json={
                "text": "This is a test message for sentiment analysis",
                "use_enhanced": False
            })
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction endpoint: OK (sentiment: {result.get('sentiment', 'unknown')})")
            else:
                print(f"❌ Prediction endpoint failed: {response.status_code}")
                return False
            
            # Test batch prediction
            response = requests.post(f"{self.api_url}/api/predict/batch", json={
                "texts": ["Great!", "Terrible!", "Okay"],
                "use_enhanced": False
            })
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch prediction endpoint: OK (processed: {result.get('total_processed', 0)})")
            else:
                print(f"❌ Batch prediction endpoint failed: {response.status_code}")
                return False
            
            print("✅ All API tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ API testing failed: {e}")
            return False
    
    def deploy_development(self):
        """Deploy API in development mode."""
        print("🚀 DEVELOPMENT DEPLOYMENT")
        print("=" * 50)
        
        if not self.check_environment():
            return False
        
        if not self.prepare_models():
            return False
        
        process = self.start_api_development()
        if not process:
            return False
        
        # Test API
        time.sleep(3)  # Give API time to fully start
        if not self.test_api():
            process.terminate()
            return False
        
        print("\n🎉 Development deployment successful!")
        print("Press Ctrl+C to stop the API")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping API...")
            process.terminate()
            process.wait()
            print("✅ API stopped")
        
        return True
    
    def deploy_production(self, workers=4, port=5000):
        """Deploy API in production mode."""
        print("🚀 PRODUCTION DEPLOYMENT")
        print("=" * 50)
        
        if not self.check_environment():
            return False
        
        if not self.prepare_models():
            return False
        
        process = self.start_api_production(workers, port)
        if not process:
            return False
        
        # Test API
        time.sleep(5)  # Give production API more time to start
        self.api_url = f"http://localhost:{port}"
        if not self.test_api():
            process.terminate()
            return False
        
        print("\n🎉 Production deployment successful!")
        print("Press Ctrl+C to stop the API")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping production API...")
            process.terminate()
            process.wait()
            print("✅ Production API stopped")
        
        return True


def main():
    """Main deployment function."""
    deployer = APIDeployer()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "dev" or mode == "development":
            deployer.deploy_development()
        elif mode == "prod" or mode == "production":
            workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
            port = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
            deployer.deploy_production(workers, port)
        elif mode == "test":
            if deployer.check_environment():
                print("✅ Environment check passed")
            else:
                print("❌ Environment check failed")
        else:
            print("Usage: python deploy_api.py [dev|prod|test] [workers] [port]")
    else:
        print("🚀 Sentiment Analysis API Deployment")
        print("=" * 40)
        print("Choose deployment mode:")
        print("1. Development (single process, debug mode)")
        print("2. Production (multi-worker, optimized)")
        print("3. Test environment only")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            deployer.deploy_development()
        elif choice == "2":
            workers = input("Number of workers (default: 4): ").strip()
            workers = int(workers) if workers else 4
            port = input("Port (default: 5000): ").strip()
            port = int(port) if port else 5000
            deployer.deploy_production(workers, port)
        elif choice == "3":
            if deployer.check_environment():
                print("✅ Environment ready for deployment")
            else:
                print("❌ Environment needs setup")
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
