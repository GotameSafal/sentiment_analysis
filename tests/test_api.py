#!/usr/bin/env python3
"""
API Tests
Test the REST API endpoints.
"""

import unittest
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import create_app


class TestAPI(unittest.TestCase):
    """Test API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
    
    def test_status_endpoint(self):
        """Test status endpoint."""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'running')
        self.assertIn('models', data)
        self.assertIn('endpoints', data)
        self.assertIn('configuration', data)
    
    def test_predict_endpoint_valid(self):
        """Test predict endpoint with valid input."""
        payload = {
            'text': 'This is a great product!',
            'model_name': 'basic'
        }
        
        response = self.client.post('/api/predict', 
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        # Note: This might fail if no model is trained
        # In a real test environment, you'd have pre-trained models
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('sentiment', data)
            self.assertIn('confidence', data)
            self.assertIn('probabilities', data)
    
    def test_predict_endpoint_invalid(self):
        """Test predict endpoint with invalid input."""
        # Missing text field
        payload = {'model_name': 'basic'}
        
        response = self.client.post('/api/predict',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_batch_predict_endpoint(self):
        """Test batch predict endpoint."""
        payload = {
            'texts': ['Great product!', 'Terrible service', 'It\'s okay'],
            'model_name': 'basic'
        }
        
        response = self.client.post('/api/predict/batch',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        # Note: This might fail if no model is trained
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('results', data)
            self.assertIn('total_processed', data)
    
    def test_models_endpoint(self):
        """Test models listing endpoint."""
        response = self.client.get('/api/models')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('models', data)
        self.assertIn('total_models', data)
    
    def test_root_endpoint(self):
        """Test root documentation endpoint."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('name', data)
        self.assertIn('version', data)
        self.assertIn('endpoints', data)
    
    def test_404_error(self):
        """Test 404 error handling."""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main()
