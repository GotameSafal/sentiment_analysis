import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import Navigation from './components/Navigation';
import SingleAnalysis from './components/SingleAnalysis';
import BatchAnalysis from './components/BatchAnalysis';
import YouTubeAnalysis from './components/YouTubeAnalysis';
import ModelTraining from './components/ModelTraining';
import Dashboard from './components/Dashboard';
import sentimentAPI from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [apiStatus, setApiStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  const checkApiStatus = async () => {
    try {
      const status = await sentimentAPI.getStatus();
      setApiStatus(status);
    } catch (error) {
      console.error('Failed to check API status:', error);
      setApiStatus(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkApiStatus();

    // Check API status every 30 seconds
    const interval = setInterval(checkApiStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'single':
        return <SingleAnalysis />;
      case 'batch':
        return <BatchAnalysis />;
      case 'youtube':
        return <YouTubeAnalysis />;
      case 'training':
        return <ModelTraining />;
      case 'dashboard':
        return <Dashboard apiStatus={apiStatus} />;
      default:
        return <SingleAnalysis />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-purple-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Connecting to API...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <Header apiStatus={apiStatus} onRefresh={checkApiStatus} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Navigation activeTab={activeTab} onTabChange={setActiveTab} />

        <main className="mt-8">
          {renderActiveComponent()}
        </main>
      </div>

      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#fff',
            color: '#374151',
            boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          },
        }}
      />
    </div>
  );
}

export default App;
