import React, { useState } from 'react';
import { Send, Loader2, Sparkles, TrendingUp } from 'lucide-react';
import toast from 'react-hot-toast';
import sentimentAPI, { getSentimentColor, getSentimentIcon, formatConfidence, getConfidenceColor } from '../services/api';

const SingleAnalysis = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelName, setModelName] = useState('basic');

  const exampleTexts = [
    "I absolutely love this product! It's amazing and works perfectly.",
    "This is okay, nothing special but it does the job.",
    "I'm really disappointed with this purchase. It's terrible quality.",
    "The weather today is sunny and beautiful!",
    "I'm feeling quite neutral about this situation.",
    "This movie was absolutely horrible, worst I've ever seen!"
  ];

  const handleAnalyze = async () => {
    if (!text.trim()) {
      toast.error('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    try {
      const response = await sentimentAPI.predictSingle(text, modelName);
      setResult(response);
      toast.success('Analysis completed successfully!');
    } catch (error) {
      toast.error(`Analysis failed: ${error.message}`);
      console.error('Analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleText) => {
    setText(exampleText);
    setResult(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleAnalyze();
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <Sparkles className="h-8 w-8 text-blue-600" />
          <h2 className="text-3xl font-bold gradient-text">Single Text Analysis</h2>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Analyze the sentiment of any text instantly. Enter your text below and get real-time sentiment analysis with confidence scores.
        </p>
      </div>

      {/* Input Section */}
      <div className="card">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-lg font-semibold text-gray-900">
              Enter Text to Analyze
            </label>
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="basic">Basic Model</option>
              <option value="advanced">Advanced Model</option>
            </select>
          </div>
          
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Type or paste your text here... (Ctrl+Enter to analyze)"
            className="w-full h-32 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
            maxLength={5000}
          />
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500">
              {text.length}/5000 characters
            </span>
            <button
              onClick={handleAnalyze}
              disabled={loading || !text.trim()}
              className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
              <span>{loading ? 'Analyzing...' : 'Analyze Sentiment'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Example Texts */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Try These Examples</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {exampleTexts.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              className="text-left p-3 bg-gray-50 hover:bg-blue-50 rounded-lg transition-colors duration-200 text-sm"
            >
              "{example}"
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="card fade-in">
          <div className="flex items-center space-x-2 mb-6">
            <TrendingUp className="h-6 w-6 text-blue-600" />
            <h3 className="text-xl font-semibold text-gray-900">Analysis Results</h3>
          </div>
          
          <div className="space-y-6">
            {/* Main Result */}
            <div className="text-center p-6 bg-gray-50 rounded-lg">
              <div className="text-6xl mb-4">
                {getSentimentIcon(result.sentiment)}
              </div>
              <div className={`text-2xl font-bold mb-2 ${getSentimentColor(result.sentiment)}`}>
                {result.sentiment?.toUpperCase()}
              </div>
              <div className={`text-lg ${getConfidenceColor(result.confidence)}`}>
                Confidence: {formatConfidence(result.confidence)}
              </div>
            </div>

            {/* Detailed Scores */}
            {result.scores && (
              <div>
                <h4 className="text-lg font-semibold text-gray-900 mb-4">Detailed Scores</h4>
                <div className="space-y-3">
                  {Object.entries(result.scores).map(([sentiment, score]) => (
                    <div key={sentiment} className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="text-2xl">{getSentimentIcon(sentiment)}</span>
                        <span className="font-medium capitalize">{sentiment}</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${getSentimentColor(sentiment).replace('text-', 'bg-')}`}
                            style={{ width: `${score * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium w-12 text-right">
                          {formatConfidence(score)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Model Info */}
            <div className="text-sm text-gray-500 border-t pt-4">
              <p>Model: {result.model_name || modelName}</p>
              <p>Processing time: {result.processing_time ? `${result.processing_time.toFixed(3)}s` : 'N/A'}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SingleAnalysis;
