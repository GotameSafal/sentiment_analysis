import React, { useState } from 'react';
import { Upload, Download, Loader2, FileText, BarChart3, Trash2 } from 'lucide-react';
import toast from 'react-hot-toast';
import sentimentAPI, { getSentimentColor, getSentimentIcon, formatConfidence } from '../services/api';

const BatchAnalysis = () => {
  const [texts, setTexts] = useState(['']);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelName, setModelName] = useState('basic');

  const addTextInput = () => {
    if (texts.length < 100) {
      setTexts([...texts, '']);
    } else {
      toast.error('Maximum 100 texts allowed');
    }
  };

  const removeTextInput = (index) => {
    if (texts.length > 1) {
      const newTexts = texts.filter((_, i) => i !== index);
      setTexts(newTexts);
    }
  };

  const updateText = (index, value) => {
    const newTexts = [...texts];
    newTexts[index] = value;
    setTexts(newTexts);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target.result;
        let parsedTexts = [];

        if (file.name.endsWith('.csv')) {
          // Simple CSV parsing (assumes one text per line)
          parsedTexts = content.split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .slice(0, 100); // Limit to 100 texts
        } else {
          // Plain text file - split by lines
          parsedTexts = content.split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .slice(0, 100);
        }

        if (parsedTexts.length > 0) {
          setTexts(parsedTexts);
          toast.success(`Loaded ${parsedTexts.length} texts from file`);
        } else {
          toast.error('No valid texts found in file');
        }
      } catch (error) {
        toast.error('Error reading file');
        console.error('File upload error:', error);
      }
    };

    reader.readAsText(file);
    event.target.value = ''; // Reset file input
  };

  const handleAnalyze = async () => {
    const validTexts = texts.filter(text => text.trim());
    
    if (validTexts.length === 0) {
      toast.error('Please enter at least one text to analyze');
      return;
    }

    setLoading(true);
    try {
      const response = await sentimentAPI.predictBatch(validTexts, modelName);
      setResults(response);
      toast.success(`Analyzed ${validTexts.length} texts successfully!`);
    } catch (error) {
      toast.error(`Analysis failed: ${error.message}`);
      console.error('Batch analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!results || !results.results) return;

    const csvContent = [
      ['Text', 'Sentiment', 'Confidence', 'Positive Score', 'Neutral Score', 'Negative Score'],
      ...results.results.map(result => [
        `"${result.text.replace(/"/g, '""')}"`, // Escape quotes in CSV
        result.sentiment,
        formatConfidence(result.confidence),
        formatConfidence(result.scores?.positive || 0),
        formatConfidence(result.scores?.neutral || 0),
        formatConfidence(result.scores?.negative || 0)
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sentiment_analysis_results_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success('Results downloaded successfully!');
  };

  const getSummaryStats = () => {
    if (!results || !results.results) return null;

    const sentiments = results.results.map(r => r.sentiment);
    const positive = sentiments.filter(s => s === 'positive').length;
    const neutral = sentiments.filter(s => s === 'neutral').length;
    const negative = sentiments.filter(s => s === 'negative').length;
    const total = sentiments.length;

    return { positive, neutral, negative, total };
  };

  const stats = getSummaryStats();

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <FileText className="h-8 w-8 text-blue-600" />
          <h2 className="text-3xl font-bold gradient-text">Batch Analysis</h2>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Analyze multiple texts at once. Upload a CSV file or enter texts manually to get sentiment analysis for all of them.
        </p>
      </div>

      {/* Input Section */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Input Texts</h3>
          <div className="flex items-center space-x-4">
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="basic">Basic Model</option>
              <option value="advanced">Advanced Model</option>
            </select>
            
            <label className="btn-secondary cursor-pointer flex items-center space-x-2">
              <Upload className="h-4 w-4" />
              <span>Upload File</span>
              <input
                type="file"
                accept=".csv,.txt"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
          </div>
        </div>

        <div className="space-y-3 max-h-96 overflow-y-auto custom-scrollbar">
          {texts.map((text, index) => (
            <div key={index} className="flex items-center space-x-3">
              <span className="text-sm font-medium text-gray-500 w-8">
                {index + 1}.
              </span>
              <input
                type="text"
                value={text}
                onChange={(e) => updateText(index, e.target.value)}
                placeholder="Enter text to analyze..."
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                maxLength={1000}
              />
              {texts.length > 1 && (
                <button
                  onClick={() => removeTextInput(index)}
                  className="p-2 text-gray-400 hover:text-red-600 transition-colors"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 pt-4 border-t border-gray-200 flex items-center justify-between">
          <button
            onClick={addTextInput}
            disabled={texts.length >= 100}
            className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Add Text ({texts.length}/100)
          </button>
          
          <button
            onClick={handleAnalyze}
            disabled={loading || texts.filter(t => t.trim()).length === 0}
            className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <BarChart3 className="h-5 w-5" />
            )}
            <span>{loading ? 'Analyzing...' : 'Analyze All'}</span>
          </button>
        </div>
      </div>

      {/* Results */}
      {results && (
        <div className="space-y-6 fade-in">
          {/* Summary Stats */}
          {stats && (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Summary Statistics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
                  <div className="text-sm text-gray-600">Total Analyzed</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{stats.positive}</div>
                  <div className="text-sm text-gray-600">Positive</div>
                </div>
                <div className="text-center p-4 bg-yellow-50 rounded-lg">
                  <div className="text-2xl font-bold text-yellow-600">{stats.neutral}</div>
                  <div className="text-sm text-gray-600">Neutral</div>
                </div>
                <div className="text-center p-4 bg-red-50 rounded-lg">
                  <div className="text-2xl font-bold text-red-600">{stats.negative}</div>
                  <div className="text-sm text-gray-600">Negative</div>
                </div>
              </div>
            </div>
          )}

          {/* Results Table */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Detailed Results</h3>
              <button
                onClick={downloadResults}
                className="btn-primary flex items-center space-x-2"
              >
                <Download className="h-4 w-4" />
                <span>Download CSV</span>
              </button>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-medium text-gray-900">#</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-900">Text</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-900">Sentiment</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-900">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {results.results?.map((result, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="py-3 px-4 text-gray-500">{index + 1}</td>
                      <td className="py-3 px-4 max-w-xs truncate" title={result.text}>
                        {result.text}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center space-x-2">
                          <span className="text-lg">{getSentimentIcon(result.sentiment)}</span>
                          <span className={`font-medium ${getSentimentColor(result.sentiment)}`}>
                            {result.sentiment}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4 font-medium">
                        {formatConfidence(result.confidence)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchAnalysis;
