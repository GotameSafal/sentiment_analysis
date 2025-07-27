#!/usr/bin/env python3
"""
YouTube Comment Extractor

This script extracts all comments from a YouTube video given its URL.
Requires YouTube Data API v3 key.

Usage:
    python youtube_comment_extractor.py

Dependencies:
    pip install google-api-python-client
"""

import re
import json
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class YouTubeCommentExtractor:
    def __init__(self, api_key: str):
        """
        Initialize the YouTube Comment Extractor.
        
        Args:
            api_key (str): YouTube Data API v3 key
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            str: Video ID if found, None otherwise
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_video_info(self, video_id: str) -> Dict:
        """
        Get basic video information.
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            dict: Video information
        """
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
            
            if response['items']:
                video = response['items'][0]
                return {
                    'title': video['snippet']['title'],
                    'channel': video['snippet']['channelTitle'],
                    'published_at': video['snippet']['publishedAt'],
                    'view_count': video['statistics'].get('viewCount', 'N/A'),
                    'like_count': video['statistics'].get('likeCount', 'N/A'),
                    'comment_count': video['statistics'].get('commentCount', 'N/A')
                }
        except HttpError as e:
            print(f"Error fetching video info: {e}")
        
        return {}
    
    def get_comments(self, video_id: str, max_results: int = 100) -> List[Dict]:
        """
        Extract comments from a YouTube video.
        
        Args:
            video_id (str): YouTube video ID
            max_results (int): Maximum number of comments to retrieve
            
        Returns:
            list: List of comment dictionaries
        """
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_results:
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token,
                    order='relevance'  # Can be 'time' or 'relevance'
                )
                
                response = request.execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    
                    comment_data = {
                        'author': comment['authorDisplayName'],
                        'text': comment['textDisplay'],
                        'like_count': comment['likeCount'],
                        'published_at': comment['publishedAt'],
                        'updated_at': comment['updatedAt'],
                        'reply_count': item['snippet']['totalReplyCount']
                    }
                    
                    # Get replies if they exist
                    replies = []
                    if 'replies' in item:
                        for reply in item['replies']['comments']:
                            reply_snippet = reply['snippet']
                            replies.append({
                                'author': reply_snippet['authorDisplayName'],
                                'text': reply_snippet['textDisplay'],
                                'like_count': reply_snippet['likeCount'],
                                'published_at': reply_snippet['publishedAt']
                            })
                    
                    comment_data['replies'] = replies
                    comments.append(comment_data)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
        except HttpError as e:
            print(f"Error fetching comments: {e}")
            if "commentsDisabled" in str(e):
                print("Comments are disabled for this video.")
            elif "quotaExceeded" in str(e):
                print("API quota exceeded. Please try again later.")
        
        return comments
    
    def save_comments_to_file(self, comments: List[Dict], video_info: Dict, filename: str = None):
        """
        Save comments to a JSON file.
        
        Args:
            comments (list): List of comments
            video_info (dict): Video information
            filename (str): Output filename (optional)
        """
        if not filename:
            video_title = video_info.get('title', 'unknown_video')
            # Clean filename
            filename = re.sub(r'[^\w\s-]', '', video_title).strip()
            filename = re.sub(r'[-\s]+', '-', filename)
            filename = f"{filename}_comments.json"
        
        data = {
            'video_info': video_info,
            'total_comments': len(comments),
            'comments': comments
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Comments saved to: {filename}")
    
    def extract_comments_from_url(self, url: str, max_results: int = 100, save_to_file: bool = True) -> List[Dict]:
        """
        Main method to extract comments from YouTube URL.
        
        Args:
            url (str): YouTube video URL
            max_results (int): Maximum number of comments to retrieve
            save_to_file (bool): Whether to save comments to file
            
        Returns:
            list: List of comments
        """
        # Extract video ID
        video_id = self.extract_video_id(url)
        if not video_id:
            print("Invalid YouTube URL. Please provide a valid YouTube video URL.")
            return []
        
        print(f"Extracting comments for video ID: {video_id}")
        
        # Get video information
        video_info = self.get_video_info(video_id)
        if video_info:
            print(f"Video: {video_info['title']}")
            print(f"Channel: {video_info['channel']}")
            print(f"Comments: {video_info['comment_count']}")
        
        # Get comments
        comments = self.get_comments(video_id, max_results)
        print(f"Retrieved {len(comments)} comments")
        
        # Save to file if requested
        if save_to_file and comments:
            self.save_comments_to_file(comments, video_info)
        
        return comments


def main():
    """
    Main function to run the comment extractor interactively.
    """
    print("YouTube Comment Extractor")
    print("=" * 30)
    
    # Get API key
    api_key = input("Enter your YouTube Data API v3 key: ").strip()
    if not api_key:
        print("API key is required. Get one from: https://console.developers.google.com/")
        return
    
    # Initialize extractor
    extractor = YouTubeCommentExtractor(api_key)
    
    while True:
        print("\n" + "-" * 50)
        url = input("Enter YouTube video URL (or 'quit' to exit): ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            break
        
        if not url:
            continue
        
        try:
            max_results = int(input("Maximum number of comments to extract (default 100): ") or "100")
        except ValueError:
            max_results = 100
        
        print(f"\nExtracting comments from: {url}")
        comments = extractor.extract_comments_from_url(url, max_results)
        
        if comments:
            print(f"\nFirst few comments:")
            for i, comment in enumerate(comments[:3]):
                print(f"\n{i+1}. {comment['author']}: {comment['text'][:100]}...")
        
        continue_choice = input("\nExtract from another video? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
    
    print("Thank you for using YouTube Comment Extractor!")


if __name__ == "__main__":
    main()
