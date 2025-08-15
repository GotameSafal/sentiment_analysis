import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  Brain, 
  Activity,
  Clock,
  CheckCircle,
  AlertCircle,
  Server
} from 'lucide-react';
import sentimentAPI from '../services/api';

const Dashboard = ({ apiStatus }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await sentimentAPI.listModels();
        setModels(response.models || []);
      } catch (error) {
        console.error('Failed to fetch models:', error);
      } finally {
        setLoading(false);
      }
    };

    if (apiStatus?.status === 'healthy') {
      fetchModels();
    } else {
      setLoading(false);
    }
  }, [apiStatus]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'unhealthy':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Activity className="h-5 w-5 text-gray-500" />;
    }
  };

  const getUptimeDisplay = (uptime) => {
    if (!uptime) return 'Unknown';
    
    const hours = Math.floor(uptime / 3600);
    const minutes = Math.floor((uptime % 3600) / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <BarChart3 className="h-8 w-8 text-blue-600" />
          <h2 className="text-3xl font-bold gradient-text">Dashboard</h2>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Monitor your sentiment analysis platform's performance, API status, and available models.
        </p>
      </div>

      {/* API Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">API Status</p>
              <div className="flex items-center space-x-2 mt-1">
                {getStatusIcon(apiStatus?.status)}
                <span className="text-lg font-semibold capitalize">
                  {apiStatus?.status || 'Unknown'}
                </span>
              </div>
            </div>
            <Server className="h-8 w-8 text-blue-500" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Uptime</p>
              <p className="text-lg font-semibold mt-1">
                {getUptimeDisplay(apiStatus?.uptime)}
              </p>
            </div>
            <Clock className="h-8 w-8 text-green-500" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Available Models</p>
              <p className="text-lg font-semibold mt-1">
                {apiStatus?.available_models?.length || 0}
              </p>
            </div>
            <Brain className="h-8 w-8 text-purple-500" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Version</p>
              <p className="text-lg font-semibold mt-1">
                {apiStatus?.version || 'Unknown'}
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-orange-500" />
          </div>
        </div>
      </div>

      {/* System Information */}
      {apiStatus && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <h4 className="font-medium text-gray-700 mb-2">API Details</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Status:</span>
                  <span className={`font-medium ${
                    apiStatus.status === 'healthy' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {apiStatus.status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Version:</span>
                  <span className="font-medium">{apiStatus.version || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Uptime:</span>
                  <span className="font-medium">{getUptimeDisplay(apiStatus.uptime)}</span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-medium text-gray-700 mb-2">Features</h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Single Text Analysis</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Batch Processing</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>YouTube Analysis</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Model Training</span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-medium text-gray-700 mb-2">Performance</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Response Time:</span>
                  <span className="font-medium text-green-600">Fast</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Availability:</span>
                  <span className="font-medium text-green-600">99.9%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Load:</span>
                  <span className="font-medium text-yellow-600">Normal</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Available Models */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Available Models</h3>
        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-gray-600">Loading models...</p>
          </div>
        ) : models.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((model, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Brain className="h-5 w-5 text-purple-600" />
                  <h4 className="font-medium text-gray-900">{model.name || `Model ${index + 1}`}</h4>
                </div>
                <div className="space-y-1 text-sm text-gray-600">
                  <div>Type: {model.type || 'Unknown'}</div>
                  <div>Status: <span className="text-green-600">Active</span></div>
                  {model.accuracy && (
                    <div>Accuracy: {(model.accuracy * 100).toFixed(1)}%</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No models available</p>
            <p className="text-sm text-gray-500 mt-1">
              {apiStatus?.status !== 'healthy' 
                ? 'API connection required to load models'
                : 'Train your first model to get started'
              }
            </p>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <button className="p-4 text-left bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors">
            <div className="flex items-center space-x-2 mb-2">
              <Users className="h-5 w-5 text-blue-600" />
              <span className="font-medium">Analyze Text</span>
            </div>
            <p className="text-sm text-gray-600">Start analyzing sentiment of individual texts</p>
          </button>

          <button className="p-4 text-left bg-green-50 hover:bg-green-100 rounded-lg transition-colors">
            <div className="flex items-center space-x-2 mb-2">
              <BarChart3 className="h-5 w-5 text-green-600" />
              <span className="font-medium">Batch Process</span>
            </div>
            <p className="text-sm text-gray-600">Process multiple texts at once</p>
          </button>

          <button className="p-4 text-left bg-red-50 hover:bg-red-100 rounded-lg transition-colors">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="h-5 w-5 text-red-600" />
              <span className="font-medium">YouTube Analysis</span>
            </div>
            <p className="text-sm text-gray-600">Analyze video comments sentiment</p>
          </button>

          <button className="p-4 text-left bg-purple-50 hover:bg-purple-100 rounded-lg transition-colors">
            <div className="flex items-center space-x-2 mb-2">
              <Brain className="h-5 w-5 text-purple-600" />
              <span className="font-medium">Train Model</span>
            </div>
            <p className="text-sm text-gray-600">Create custom sentiment models</p>
          </button>
        </div>
      </div>

      {/* Connection Status */}
      {!apiStatus && (
        <div className="card bg-yellow-50 border-yellow-200">
          <div className="flex items-center space-x-2 text-yellow-800">
            <AlertCircle className="h-5 w-5" />
            <span className="font-medium">API Connection Required</span>
          </div>
          <p className="mt-2 text-yellow-700">
            Please ensure the sentiment analysis API is running to access all features.
          </p>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
