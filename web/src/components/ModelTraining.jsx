import React, { useState } from 'react';
import { Brain, Plus, Trash2, Loader2, Download, Upload } from 'lucide-react';
import toast from 'react-hot-toast';
import sentimentAPI from '../services/api';

const ModelTraining = () => {
  const [trainingData, setTrainingData] = useState([
    { text: '', label: 'positive' }
  ]);
  const [modelName, setModelName] = useState('basic');
  const [modelType, setModelType] = useState('logistic');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const addTrainingExample = () => {
    if (trainingData.length < 1000) {
      setTrainingData([...trainingData, { text: '', label: 'positive' }]);
    } else {
      toast.error('Maximum 1000 training examples allowed');
    }
  };

  const removeTrainingExample = (index) => {
    if (trainingData.length > 1) {
      const newData = trainingData.filter((_, i) => i !== index);
      setTrainingData(newData);
    }
  };

  const updateTrainingExample = (index, field, value) => {
    const newData = [...trainingData];
    newData[index][field] = value;
    setTrainingData(newData);
  };

  const loadSampleData = () => {
    const sampleData = [
      { text: "I love this product! It's amazing!", label: 'positive' },
      { text: "This is the best thing ever!", label: 'positive' },
      { text: "Absolutely fantastic experience!", label: 'positive' },
      { text: "Great quality and fast delivery!", label: 'positive' },
      { text: "This is okay, nothing special.", label: 'neutral' },
      { text: "It's average, does the job.", label: 'neutral' },
      { text: "Neither good nor bad.", label: 'neutral' },
      { text: "It's fine, I guess.", label: 'neutral' },
      { text: "I hate this! Terrible quality!", label: 'negative' },
      { text: "Worst purchase ever made!", label: 'negative' },
      { text: "Complete waste of money!", label: 'negative' },
      { text: "Very disappointed with this.", label: 'negative' }
    ];
    
    setTrainingData(sampleData);
    toast.success('Sample training data loaded!');
  };

  const handleTrain = async () => {
    const validData = trainingData.filter(item => item.text.trim());
    
    if (validData.length < 6) {
      toast.error('Please provide at least 6 training examples');
      return;
    }

    // Check if we have examples for each sentiment
    const sentiments = [...new Set(validData.map(item => item.label))];
    if (sentiments.length < 2) {
      toast.error('Please provide examples for at least 2 different sentiments');
      return;
    }

    setLoading(true);
    try {
      const texts = validData.map(item => item.text);
      const labels = validData.map(item => item.label);
      
      const response = await sentimentAPI.trainModel(texts, labels, modelName, modelType);
      setResults(response);
      toast.success('Model trained successfully!');
    } catch (error) {
      toast.error(`Training failed: ${error.message}`);
      console.error('Training error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDataDistribution = () => {
    const validData = trainingData.filter(item => item.text.trim());
    const distribution = { positive: 0, neutral: 0, negative: 0 };
    
    validData.forEach(item => {
      distribution[item.label] = (distribution[item.label] || 0) + 1;
    });
    
    return { ...distribution, total: validData.length };
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target.result;
        const lines = content.split('\n').filter(line => line.trim());
        const parsedData = [];

        lines.forEach((line, index) => {
          if (index === 0 && line.toLowerCase().includes('text')) return; // Skip header
          
          const parts = line.split(',');
          if (parts.length >= 2) {
            const text = parts[0].replace(/"/g, '').trim();
            const label = parts[1].replace(/"/g, '').trim().toLowerCase();
            
            if (text && ['positive', 'neutral', 'negative'].includes(label)) {
              parsedData.push({ text, label });
            }
          }
        });

        if (parsedData.length > 0) {
          setTrainingData(parsedData.slice(0, 1000)); // Limit to 1000
          toast.success(`Loaded ${parsedData.length} training examples`);
        } else {
          toast.error('No valid training data found in file');
        }
      } catch (error) {
        toast.error('Error reading file');
        console.error('File upload error:', error);
      }
    };

    reader.readAsText(file);
    event.target.value = '';
  };

  const downloadTemplate = () => {
    const csvContent = [
      ['text', 'label'],
      ['I love this product!', 'positive'],
      ['This is okay', 'neutral'],
      ['I hate this', 'negative']
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'training_data_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success('Template downloaded!');
  };

  const distribution = getDataDistribution();

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <Brain className="h-8 w-8 text-purple-600" />
          <h2 className="text-3xl font-bold gradient-text">Model Training</h2>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Train custom sentiment analysis models with your own data. Provide labeled examples to improve accuracy for your specific use case.
        </p>
      </div>

      {/* Configuration */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model Name
            </label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="Enter model name"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model Type
            </label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            >
              <option value="logistic">Logistic Regression</option>
              <option value="svm">Support Vector Machine</option>
              <option value="naive_bayes">Naive Bayes</option>
            </select>
          </div>
        </div>
      </div>

      {/* Data Statistics */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Distribution</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">{distribution.total}</div>
            <div className="text-sm text-gray-600">Total Examples</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{distribution.positive}</div>
            <div className="text-sm text-gray-600">Positive</div>
          </div>
          <div className="text-center p-4 bg-yellow-50 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600">{distribution.neutral}</div>
            <div className="text-sm text-gray-600">Neutral</div>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">{distribution.negative}</div>
            <div className="text-sm text-gray-600">Negative</div>
          </div>
        </div>
      </div>

      {/* Training Data */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Training Data</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={downloadTemplate}
              className="btn-secondary flex items-center space-x-2"
            >
              <Download className="h-4 w-4" />
              <span>Template</span>
            </button>
            
            <label className="btn-secondary cursor-pointer flex items-center space-x-2">
              <Upload className="h-4 w-4" />
              <span>Upload CSV</span>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
            
            <button
              onClick={loadSampleData}
              className="btn-secondary"
            >
              Load Sample
            </button>
            
            <button
              onClick={addTrainingExample}
              disabled={trainingData.length >= 1000}
              className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Plus className="h-4 w-4" />
              <span>Add Example</span>
            </button>
          </div>
        </div>

        <div className="space-y-3 max-h-96 overflow-y-auto custom-scrollbar">
          {trainingData.map((item, index) => (
            <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-500 w-8">
                {index + 1}.
              </span>
              <input
                type="text"
                value={item.text}
                onChange={(e) => updateTrainingExample(index, 'text', e.target.value)}
                placeholder="Enter training text..."
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              />
              <select
                value={item.label}
                onChange={(e) => updateTrainingExample(index, 'label', e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              >
                <option value="positive">Positive</option>
                <option value="neutral">Neutral</option>
                <option value="negative">Negative</option>
              </select>
              {trainingData.length > 1 && (
                <button
                  onClick={() => removeTrainingExample(index)}
                  className="p-2 text-gray-400 hover:text-red-600 transition-colors"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 pt-4 border-t border-gray-200">
          <button
            onClick={handleTrain}
            disabled={loading || distribution.total < 6}
            className="btn-primary flex items-center space-x-2 w-full justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Brain className="h-5 w-5" />
            )}
            <span>{loading ? 'Training Model...' : 'Train Model'}</span>
          </button>
        </div>
      </div>

      {/* Training Results */}
      {results && (
        <div className="card fade-in">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Results</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {results.accuracy ? `${(results.accuracy * 100).toFixed(1)}%` : 'N/A'}
                </div>
                <div className="text-sm text-gray-600">Accuracy</div>
              </div>
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {results.training_time ? `${results.training_time.toFixed(2)}s` : 'N/A'}
                </div>
                <div className="text-sm text-gray-600">Training Time</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {results.model_name || modelName}
                </div>
                <div className="text-sm text-gray-600">Model Name</div>
              </div>
            </div>
            
            {results.message && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-green-800">{results.message}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelTraining;
