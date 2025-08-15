import React, { useState } from 'react';
import { Youtube, Play, Loader2, ExternalLink, Users, MessageCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import sentimentAPI, { getSentimentColor, getSentimentIcon, formatConfidence } from '../services/api';

const YouTubeAnalysis = () => {
  const [videoUrl, setVideoUrl] = useState('');
  const [maxComments, setMaxComments] = useState(100);
  const [modelName, setModelName] = useState('basic');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!videoUrl.trim()) {
      toast.error('Please enter a YouTube video URL');
      return;
    }

    setLoading(true);
    try {
      const response = await sentimentAPI.analyzeYouTubeVideo(
        videoUrl,
        maxComments,
        modelName
      );
      setResults(response);

      if (response.error) {
        toast.error(response.error);
      } else {
        toast.success(`Analyzed ${response.sentiment_summary?.analyzed_comments || 0} comments!`);
      }
    } catch (error) {
      toast.error(`Analysis failed: ${error.message}`);
      console.error('YouTube analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  const extractVideoId = (url) => {
    const regex = /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/;
    const match = url.match(regex);
    return match ? match[1] : null;
  };

  const videoId = extractVideoId(videoUrl);

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <Youtube className="h-8 w-8 text-red-600" />
          <h2 className="text-3xl font-bold gradient-text">YouTube Analysis</h2>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Analyze sentiment of YouTube video comments. Get insights into audience reactions and engagement patterns.
        </p>
      </div>

      {/* Input Section */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">Video Analysis Setup</h3>
        
        <div className="space-y-4">
          {/* Video URL */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              YouTube Video URL
            </label>
            <input
              type="url"
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* API Configuration Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <ExternalLink className="h-5 w-5 text-blue-600 mt-0.5" />
              <div className="text-sm text-blue-800">
                <p className="font-medium mb-1">YouTube API Configuration</p>
                <p>
                  YouTube API key is configured on the server. If you're the administrator,
                  set the <code className="bg-blue-100 px-1 rounded">YOUTUBE_API_KEY</code> environment variable.
                </p>
                <a
                  href="https://console.cloud.google.com/apis/credentials"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-700 underline mt-1 inline-block"
                >
                  Get API Key from Google Cloud Console →
                </a>
              </div>
            </div>
          </div>

          {/* Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Comments
              </label>
              <input
                type="number"
                value={maxComments}
                onChange={(e) => setMaxComments(Math.max(1, Math.min(500, parseInt(e.target.value) || 100)))}
                min="1"
                max="500"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model
              </label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="basic">Basic Model</option>
                <option value="advanced">Advanced Model</option>
              </select>
            </div>
          </div>

          {/* Video Preview */}
          {videoId && (
            <div className="mt-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Video Preview
              </label>
              <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                <iframe
                  src={`https://www.youtube.com/embed/${videoId}`}
                  title="YouTube video preview"
                  className="w-full h-full"
                  allowFullScreen
                />
              </div>
            </div>
          )}

          {/* Analyze Button */}
          <div className="pt-4 border-t border-gray-200">
            <button
              onClick={handleAnalyze}
              disabled={loading || !videoUrl.trim()}
              className="btn-primary flex items-center space-x-2 w-full justify-center disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Play className="h-5 w-5" />
              )}
              <span>{loading ? 'Analyzing Comments...' : 'Analyze Video Comments'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {results && !results.error && (
        <div className="space-y-6 fade-in">
          {/* Video Info */}
          {results.video_info && (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Video Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Youtube className="h-5 w-5 text-red-600" />
                  <div>
                    <div className="font-medium">{results.video_info.title}</div>
                    <div className="text-sm text-gray-500">Title</div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Users className="h-5 w-5 text-blue-600" />
                  <div>
                    <div className="font-medium">{results.video_info.view_count?.toLocaleString()}</div>
                    <div className="text-sm text-gray-500">Views</div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <MessageCircle className="h-5 w-5 text-green-600" />
                  <div>
                    <div className="font-medium">{results.video_info.comment_count?.toLocaleString()}</div>
                    <div className="text-sm text-gray-500">Total Comments</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Sentiment Summary */}
          {results.sentiment_summary && (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-gray-900">
                    {results.sentiment_summary.analyzed_comments}
                  </div>
                  <div className="text-sm text-gray-600">Analyzed</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {results.sentiment_summary.positive}
                  </div>
                  <div className="text-sm text-gray-600">Positive</div>
                </div>
                <div className="text-center p-4 bg-yellow-50 rounded-lg">
                  <div className="text-2xl font-bold text-yellow-600">
                    {results.sentiment_summary.neutral}
                  </div>
                  <div className="text-sm text-gray-600">Neutral</div>
                </div>
                <div className="text-center p-4 bg-red-50 rounded-lg">
                  <div className="text-2xl font-bold text-red-600">
                    {results.sentiment_summary.negative}
                  </div>
                  <div className="text-sm text-gray-600">Negative</div>
                </div>
              </div>

              {/* Overall Sentiment */}
              <div className="text-center p-6 bg-gray-50 rounded-lg">
                <div className="text-4xl mb-2">
                  {getSentimentIcon(results.sentiment_summary.overall_sentiment)}
                </div>
                <div className={`text-xl font-bold ${getSentimentColor(results.sentiment_summary.overall_sentiment)}`}>
                  Overall: {results.sentiment_summary.overall_sentiment?.toUpperCase()}
                </div>
              </div>
            </div>
          )}

          {/* Sample Comments */}
          {results.sample_comments && results.sample_comments.length > 0 && (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Sample Comments</h3>
              <div className="space-y-4 max-h-96 overflow-y-auto custom-scrollbar">
                {results.sample_comments.map((comment, index) => (
                  <div key={index} className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{getSentimentIcon(comment.sentiment)}</span>
                        <span className={`font-medium ${getSentimentColor(comment.sentiment)}`}>
                          {comment.sentiment}
                        </span>
                        <span className="text-sm text-gray-500">
                          ({formatConfidence(comment.confidence)})
                        </span>
                      </div>
                    </div>
                    <p className="text-gray-700">{comment.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {results && results.error && (
        <div className="card bg-red-50 border-red-200">
          <div className="flex items-center space-x-2 text-red-800">
            <ExternalLink className="h-5 w-5" />
            <span className="font-medium">Analysis Error</span>
          </div>
          <p className="mt-2 text-red-700">{results.error}</p>
        </div>
      )}
    </div>
  );
};

export default YouTubeAnalysis;
