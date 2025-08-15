#!/usr/bin/env python3
"""
YouTube Service
Handles YouTube comment extraction and analysis.
Separated from main sentiment analysis for modularity.
"""

import re
import time
from typing import Dict, List, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
from datetime import datetime

from config.settings import YouTubeConfig

logger = logging.getLogger(__name__)


class YouTubeCommentExtractor:
    """Extract comments from YouTube videos."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize YouTube API client."""
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            logger.info("YouTube API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API client: {e}")
            raise
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        logger.warning(f"Could not extract video ID from URL: {url}")
        return None
    
    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get basic video information."""
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                raise ValueError(f"Video not found: {video_id}")
            
            video = response['items'][0]
            snippet = video['snippet']
            statistics = video['statistics']
            
            return {
                'video_id': video_id,
                'title': snippet['title'],
                'channel_title': snippet['channelTitle'],
                'published_at': snippet['publishedAt'],
                'description': snippet.get('description', ''),
                'view_count': int(statistics.get('viewCount', 0)),
                'like_count': int(statistics.get('likeCount', 0)),
                'comment_count': int(statistics.get('commentCount', 0))
            }
            
        except HttpError as e:
            logger.error(f"YouTube API error getting video info: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
    
    def extract_comments(self, video_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Extract comments from a YouTube video."""
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_results:
                # Calculate how many comments to request in this batch
                remaining = max_results - len(comments)
                batch_size = min(remaining, 100)  # YouTube API max is 100 per request
                
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=batch_size,
                    order='relevance',
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    
                    comments.append({
                        'comment_id': item['id'],
                        'author': comment['authorDisplayName'],
                        'text': comment['textDisplay'],
                        'like_count': comment['likeCount'],
                        'published_at': comment['publishedAt'],
                        'updated_at': comment.get('updatedAt', comment['publishedAt']),
                        'reply_count': item['snippet']['totalReplyCount']
                    })
                
                # Check if there are more pages
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                
                # Rate limiting
                time.sleep(1.0 / YouTubeConfig.REQUESTS_PER_SECOND)
            
            logger.info(f"Extracted {len(comments)} comments from video {video_id}")
            return comments
            
        except HttpError as e:
            if e.resp.status == 403:
                logger.error("YouTube API quota exceeded or comments disabled")
                raise ValueError("YouTube API quota exceeded or comments disabled for this video")
            else:
                logger.error(f"YouTube API error: {e}")
                raise
        except Exception as e:
            logger.error(f"Error extracting comments: {e}")
            raise
    
    def extract_comments_from_url(self, url: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Extract comments from a YouTube URL."""
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        return self.extract_comments(video_id, max_results)


class YouTubeAnalysisService:
    """Service for analyzing YouTube video comments."""

    def __init__(self, sentiment_service):
        self.sentiment_service = sentiment_service
        self.api_key = YouTubeConfig.API_KEY
        self.extractor = None

        if self.api_key and YouTubeConfig.is_configured():
            self._initialize_extractor()
        else:
            logger.warning("YouTube API key not configured. YouTube analysis will not be available.")
    
    def _initialize_extractor(self):
        """Initialize YouTube comment extractor."""
        try:
            self.extractor = YouTubeCommentExtractor(self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize YouTube extractor: {e}")
            self.extractor = None
    
    def is_configured(self) -> bool:
        """Check if YouTube service is properly configured."""
        return YouTubeConfig.is_configured() and self.extractor is not None
    
    def analyze_video_comments(
        self, 
        video_url: str, 
        max_comments: int = 100,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze sentiment of YouTube video comments."""
        
        if not self.extractor:
            raise ValueError("YouTube API key not configured")
        
        # Validate max_comments
        if max_comments > YouTubeConfig.DEFAULT_MAX_COMMENTS:
            max_comments = YouTubeConfig.DEFAULT_MAX_COMMENTS
            logger.warning(f"Max comments limited to {YouTubeConfig.DEFAULT_MAX_COMMENTS}")
        
        try:
            # Extract video ID and get video info
            video_id = self.extractor.extract_video_id(video_url)
            if not video_id:
                raise ValueError(f"Invalid YouTube URL: {video_url}")
            
            video_info = self.extractor.get_video_info(video_id)
            
            # Extract comments
            logger.info(f"Extracting up to {max_comments} comments from video: {video_info['title']}")
            comments = self.extractor.extract_comments(video_id, max_comments)
            
            if not comments:
                return {
                    'video_info': video_info,
                    'error': 'No comments found or comments are disabled',
                    'total_comments': 0,
                    'analyzed_comments': [],
                    'sentiment_summary': {},
                    'timestamp': datetime.now().isoformat()
                }
            
            # Analyze sentiment of comments
            logger.info(f"Analyzing sentiment for {len(comments)} comments")
            comment_texts = [comment['text'] for comment in comments]
            sentiment_results = self.sentiment_service.predict_batch(comment_texts, model_name)
            
            # Combine comments with sentiment analysis
            analyzed_comments = []
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            confidence_scores = []
            
            for comment, sentiment_result in zip(comments, sentiment_results):
                if 'error' not in sentiment_result:
                    analyzed_comment = {
                        **comment,
                        'sentiment': sentiment_result['sentiment'],
                        'confidence': sentiment_result['confidence'],
                        'probabilities': sentiment_result['probabilities']
                    }
                    
                    sentiment_counts[sentiment_result['sentiment']] += 1
                    confidence_scores.append(sentiment_result['confidence'])
                else:
                    analyzed_comment = {
                        **comment,
                        'sentiment': 'unknown',
                        'confidence': 0.0,
                        'error': sentiment_result['error']
                    }
                
                analyzed_comments.append(analyzed_comment)
            
            # Calculate statistics
            total_analyzed = len([c for c in analyzed_comments if c['sentiment'] != 'unknown'])
            sentiment_percentages = {
                sentiment: (count / total_analyzed * 100) if total_analyzed > 0 else 0
                for sentiment, count in sentiment_counts.items()
            }
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Create summary
            sentiment_summary = {
                'total_comments': len(comments),
                'analyzed_comments': total_analyzed,
                'sentiment_counts': sentiment_counts,
                'sentiment_percentages': sentiment_percentages,
                'average_confidence': avg_confidence,
                'most_common_sentiment': max(sentiment_counts, key=sentiment_counts.get),
                'confidence_distribution': {
                    'high_confidence': len([s for s in confidence_scores if s >= 0.8]),
                    'medium_confidence': len([s for s in confidence_scores if 0.6 <= s < 0.8]),
                    'low_confidence': len([s for s in confidence_scores if s < 0.6])
                }
            }
            
            return {
                'video_info': video_info,
                'sentiment_summary': sentiment_summary,
                'analyzed_comments': analyzed_comments,
                'model_used': model_name or 'default',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing YouTube video: {e}")
            raise
    
    def get_video_sentiment_trends(
        self, 
        video_url: str, 
        max_comments: int = 200,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze sentiment trends over time for a video."""
        
        analysis_result = self.analyze_video_comments(video_url, max_comments, model_name)
        
        if 'error' in analysis_result:
            return analysis_result
        
        # Sort comments by publication date
        comments = analysis_result['analyzed_comments']
        comments.sort(key=lambda x: x['published_at'])
        
        # Group comments by time periods (e.g., by day)
        from collections import defaultdict
        daily_sentiments = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0})
        
        for comment in comments:
            if comment['sentiment'] != 'unknown':
                # Extract date (YYYY-MM-DD)
                date = comment['published_at'][:10]
                daily_sentiments[date][comment['sentiment']] += 1
                daily_sentiments[date]['total'] += 1
        
        # Calculate daily percentages
        trend_data = []
        for date, counts in sorted(daily_sentiments.items()):
            if counts['total'] > 0:
                trend_data.append({
                    'date': date,
                    'positive_percentage': (counts['positive'] / counts['total']) * 100,
                    'neutral_percentage': (counts['neutral'] / counts['total']) * 100,
                    'negative_percentage': (counts['negative'] / counts['total']) * 100,
                    'total_comments': counts['total']
                })
        
        return {
            'video_info': analysis_result['video_info'],
            'overall_summary': analysis_result['sentiment_summary'],
            'trend_data': trend_data,
            'model_used': model_name or 'default',
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_videos(
        self, 
        video_urls: List[str], 
        max_comments_per_video: int = 100,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare sentiment analysis across multiple videos."""
        
        if len(video_urls) > 5:
            raise ValueError("Maximum 5 videos can be compared at once")
        
        video_analyses = []
        
        for url in video_urls:
            try:
                analysis = self.analyze_video_comments(url, max_comments_per_video, model_name)
                video_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing video {url}: {e}")
                video_analyses.append({
                    'video_url': url,
                    'error': str(e)
                })
        
        # Create comparison summary
        successful_analyses = [a for a in video_analyses if 'error' not in a]
        
        if not successful_analyses:
            return {
                'error': 'No videos could be analyzed successfully',
                'video_analyses': video_analyses,
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate aggregate statistics
        total_comments = sum(a['sentiment_summary']['total_comments'] for a in successful_analyses)
        aggregate_sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for analysis in successful_analyses:
            for sentiment, count in analysis['sentiment_summary']['sentiment_counts'].items():
                aggregate_sentiment_counts[sentiment] += count
        
        aggregate_percentages = {
            sentiment: (count / sum(aggregate_sentiment_counts.values()) * 100) if sum(aggregate_sentiment_counts.values()) > 0 else 0
            for sentiment, count in aggregate_sentiment_counts.items()
        }
        
        return {
            'comparison_summary': {
                'total_videos_analyzed': len(successful_analyses),
                'total_comments_analyzed': total_comments,
                'aggregate_sentiment_counts': aggregate_sentiment_counts,
                'aggregate_sentiment_percentages': aggregate_percentages
            },
            'individual_analyses': video_analyses,
            'model_used': model_name or 'default',
            'timestamp': datetime.now().isoformat()
        }
