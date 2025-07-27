#!/usr/bin/env python3
"""
Example usage of YouTube Comment Extractor
"""

from youtube_comment_extractor import YouTubeCommentExtractor


def example_usage():
    # Replace with your actual API key
    API_KEY = "AIzaSyA0SExx6THq-Hikiattfaz3W5D1XAmyVqQ"

    # Initialize the extractor
    extractor = YouTubeCommentExtractor(API_KEY)

    # Example YouTube URL
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # Extract comments
    comments = extractor.extract_comments_from_url(
        url=youtube_url,
        max_results=50,  # Get up to 50 comments
        save_to_file=True,  # Save to JSON file
    )

    # Print some statistics
    if comments:
        print(f"\nExtracted {len(comments)} comments")
        print(f"Total replies: {sum(len(comment['replies']) for comment in comments)}")

        # Show top liked comments
        top_comments = sorted(comments, key=lambda x: x["like_count"], reverse=True)[:5]
        print("\nTop 5 most liked comments:")
        for i, comment in enumerate(top_comments, 1):
            print(f"{i}. {comment['author']} ({comment['like_count']} likes)")
            print(f"   {comment['text'][:100]}...")
    else:
        print("No comments extracted")


if __name__ == "__main__":
    example_usage()
