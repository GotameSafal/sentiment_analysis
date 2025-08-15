import React from 'react';
import { Brain, RefreshCw, Wifi, WifiOff } from 'lucide-react';

const Header = ({ apiStatus, onRefresh }) => {
  const getStatusColor = () => {
    if (!apiStatus) return 'text-gray-500';
    return apiStatus.status === 'healthy' ? 'text-green-500' : 'text-red-500';
  };

  const getStatusIcon = () => {
    if (!apiStatus) return <WifiOff className="h-4 w-4" />;
    return apiStatus.status === 'healthy' ? 
      <Wifi className="h-4 w-4" /> : 
      <WifiOff className="h-4 w-4" />;
  };

  const getStatusText = () => {
    if (!apiStatus) return 'Connecting...';
    return apiStatus.status === 'healthy' ? 'Connected' : 'Disconnected';
  };

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">
                Sentiment Analysis
              </h1>
              <p className="text-sm text-gray-500">
                AI-Powered Text Analysis Platform
              </p>
            </div>
          </div>

          {/* Status and Actions */}
          <div className="flex items-center space-x-4">
            {/* API Status */}
            <div className="flex items-center space-x-2">
              <div className={`flex items-center space-x-1 ${getStatusColor()}`}>
                {getStatusIcon()}
                <span className="text-sm font-medium">
                  {getStatusText()}
                </span>
              </div>
              
              {/* Refresh Button */}
              <button
                onClick={onRefresh}
                className="p-2 text-gray-400 hover:text-gray-600 transition-colors duration-200 rounded-lg hover:bg-gray-100"
                title="Refresh API Status"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>

            {/* API Info */}
            {apiStatus && (
              <div className="hidden md:block text-right">
                <div className="text-xs text-gray-500">
                  Models: {apiStatus.available_models?.length || 0}
                </div>
                <div className="text-xs text-gray-500">
                  Version: {apiStatus.version || 'Unknown'}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
