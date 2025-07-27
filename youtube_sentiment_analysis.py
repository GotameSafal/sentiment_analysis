#!/usr/bin/env python3
"""
YouTube Comment Sentiment Analysis

This script combines YouTube comment extraction with sentiment analysis to:
1. Extract comments from YouTube videos
2. Analyze sentiment of each comment
3. Generate comprehensive sentiment reports
4. Visualize sentiment distribution
5. Export results to various formats

Usage:
    python youtube_sentiment_analysis.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Any
import numpy as np
from youtube_comment_extractor import YouTubeCommentExtractor
from sentiment_analyzer import SentimentAnalyzer, SentimentDataset
import warnings

warnings.filterwarnings("ignore")


class YouTubeSentimentAnalyzer:
    """Complete YouTube comment sentiment analysis system."""

    def __init__(self, api_key: str):
        """Initialize with YouTube API key."""
        self.comment_extractor = YouTubeCommentExtractor(api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.results = None

    def setup_sentiment_model(
        self,
        model_path: str = None,
        retrain: bool = False,
        use_youtube_dataset: bool = True,
    ):
        """Setup or train the sentiment analysis model."""
        if model_path and os.path.exists(model_path) and not retrain:
            print("Loading existing sentiment model...")
            self.sentiment_analyzer.load_model(model_path)
        else:
            if use_youtube_dataset and os.path.exists("YoutubeCommentsDataSet.csv"):
                print("Training new sentiment model with YouTube dataset...")
                # Load the YouTube dataset
                df = pd.read_csv("YoutubeCommentsDataSet.csv")
                df = df.rename(columns={"Comment": "text", "Sentiment": "sentiment"})
                df["sentiment"] = df["sentiment"].str.lower()

                # Sample a subset for faster training if dataset is very large
                if len(df) > 10000:
                    print(
                        f"Large dataset detected ({len(df)} samples). Using balanced sample of 10,000 for training..."
                    )
                    # Create balanced sample
                    sample_size_per_class = 3333
                    balanced_dfs = []
                    for sentiment in df["sentiment"].unique():
                        sentiment_df = df[df["sentiment"] == sentiment]
                        if len(sentiment_df) >= sample_size_per_class:
                            sampled = sentiment_df.sample(
                                n=sample_size_per_class, random_state=42
                            )
                        else:
                            sampled = sentiment_df
                        balanced_dfs.append(sampled)
                    df = pd.concat(balanced_dfs, ignore_index=True).sample(
                        frac=1, random_state=42
                    )

                print(f"Training with {len(df)} samples...")
                print(
                    "Sentiment distribution:", df["sentiment"].value_counts().to_dict()
                )
                self.sentiment_analyzer.train_model(df, model_type="logistic")
            else:
                print("Training new sentiment model with sample data...")
                dataset = SentimentDataset()
                df = dataset.create_sample_dataset(size=2000)
                self.sentiment_analyzer.train_model(df, model_type="logistic")

            # Save the model
            if model_path:
                self.sentiment_analyzer.save_model(model_path)

    def analyze_video_comments(
        self, video_url: str, max_comments: int = 100
    ) -> Dict[str, Any]:
        """Analyze sentiment of YouTube video comments."""
        print(f"Extracting comments from: {video_url}")

        # Extract comments
        comments = self.comment_extractor.extract_comments_from_url(
            video_url, max_comments, save_to_file=False
        )

        if not comments:
            print("No comments found or extraction failed.")
            return {}

        print(f"Analyzing sentiment for {len(comments)} comments...")

        # Analyze sentiment for each comment
        analyzed_comments = []
        for comment in comments:
            sentiment_result = self.sentiment_analyzer.predict_sentiment(
                comment["text"]
            )

            analyzed_comment = {
                "author": comment["author"],
                "text": comment["text"],
                "like_count": comment["like_count"],
                "reply_count": comment["reply_count"],
                "published_at": comment["published_at"],
                "sentiment": sentiment_result["sentiment"],
                "confidence": sentiment_result["confidence"],
                "sentiment_probabilities": sentiment_result["probabilities"],
            }
            analyzed_comments.append(analyzed_comment)

        # Get video info
        video_id = self.comment_extractor.extract_video_id(video_url)
        video_info = self.comment_extractor.get_video_info(video_id) if video_id else {}

        # Calculate sentiment statistics
        sentiments = [c["sentiment"] for c in analyzed_comments]
        sentiment_counts = pd.Series(sentiments).value_counts()

        # Calculate average confidence by sentiment
        sentiment_confidence = {}
        for sentiment in ["positive", "negative", "neutral"]:
            sentiment_comments = [
                c for c in analyzed_comments if c["sentiment"] == sentiment
            ]
            if sentiment_comments:
                avg_confidence = np.mean([c["confidence"] for c in sentiment_comments])
                sentiment_confidence[sentiment] = avg_confidence
            else:
                sentiment_confidence[sentiment] = 0.0

        # Find most liked comments by sentiment
        top_comments_by_sentiment = {}
        for sentiment in ["positive", "negative", "neutral"]:
            sentiment_comments = [
                c for c in analyzed_comments if c["sentiment"] == sentiment
            ]
            if sentiment_comments:
                top_comment = max(sentiment_comments, key=lambda x: x["like_count"])
                top_comments_by_sentiment[sentiment] = top_comment

        self.results = {
            "video_info": video_info,
            "video_url": video_url,
            "analysis_date": datetime.now().isoformat(),
            "total_comments": len(analyzed_comments),
            "comments": analyzed_comments,
            "sentiment_distribution": sentiment_counts.to_dict(),
            "sentiment_percentages": (
                sentiment_counts / len(analyzed_comments) * 100
            ).to_dict(),
            "average_confidence": sentiment_confidence,
            "top_comments_by_sentiment": top_comments_by_sentiment,
            "overall_sentiment": sentiment_counts.index[0]
            if len(sentiment_counts) > 0
            else "neutral",
        }

        return self.results

    def generate_report(self) -> str:
        """Generate a comprehensive text report."""
        if not self.results:
            return "No analysis results available. Run analyze_video_comments() first."

        report = []
        report.append("=" * 60)
        report.append("YOUTUBE COMMENT SENTIMENT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Video information
        video_info = self.results["video_info"]
        if video_info:
            report.append("VIDEO INFORMATION:")
            report.append(f"Title: {video_info.get('title', 'N/A')}")
            report.append(f"Channel: {video_info.get('channel', 'N/A')}")
            report.append(f"Views: {video_info.get('view_count', 'N/A')}")
            report.append(f"Likes: {video_info.get('like_count', 'N/A')}")
            report.append("")

        # Analysis summary
        report.append("ANALYSIS SUMMARY:")
        report.append(f"Total Comments Analyzed: {self.results['total_comments']}")
        report.append(f"Analysis Date: {self.results['analysis_date']}")
        report.append(f"Overall Sentiment: {self.results['overall_sentiment'].upper()}")
        report.append("")

        # Sentiment distribution
        report.append("SENTIMENT DISTRIBUTION:")
        for sentiment, count in self.results["sentiment_distribution"].items():
            percentage = self.results["sentiment_percentages"][sentiment]
            confidence = self.results["average_confidence"][sentiment]
            report.append(
                f"{sentiment.capitalize()}: {count} comments ({percentage:.1f}%) - Avg Confidence: {confidence:.3f}"
            )
        report.append("")

        # Top comments by sentiment
        report.append("TOP LIKED COMMENTS BY SENTIMENT:")
        for sentiment, comment in self.results["top_comments_by_sentiment"].items():
            report.append(f"\n{sentiment.upper()}:")
            report.append(f"Author: {comment['author']}")
            report.append(f"Likes: {comment['like_count']}")
            report.append(
                f"Text: {comment['text'][:200]}{'...' if len(comment['text']) > 200 else ''}"
            )
            report.append(f"Confidence: {comment['confidence']:.3f}")

        return "\n".join(report)

    def create_visualizations(self, save_path: str = "sentiment_analysis_plots.png"):
        """Create sentiment analysis visualizations."""
        if not self.results:
            print("No analysis results available.")
            return

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "YouTube Comment Sentiment Analysis", fontsize=16, fontweight="bold"
        )

        # 1. Sentiment Distribution Pie Chart
        sentiment_counts = self.results["sentiment_distribution"]
        colors = ["#ff6b6b", "#ffd93d", "#6bcf7f"]  # red, yellow, green
        axes[0, 0].pie(
            sentiment_counts.values(),
            labels=sentiment_counts.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[0, 0].set_title("Sentiment Distribution")

        # 2. Sentiment Distribution Bar Chart
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        bars = axes[0, 1].bar(sentiments, counts, color=colors)
        axes[0, 1].set_title("Sentiment Counts")
        axes[0, 1].set_ylabel("Number of Comments")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        # 3. Confidence Distribution by Sentiment
        df_comments = pd.DataFrame(self.results["comments"])
        sns.boxplot(data=df_comments, x="sentiment", y="confidence", ax=axes[1, 0])
        axes[1, 0].set_title("Confidence Distribution by Sentiment")
        axes[1, 0].set_ylabel("Confidence Score")

        # 4. Like Count vs Sentiment
        sns.boxplot(data=df_comments, x="sentiment", y="like_count", ax=axes[1, 1])
        axes[1, 1].set_title("Like Count Distribution by Sentiment")
        axes[1, 1].set_ylabel("Like Count")
        axes[1, 1].set_yscale("log")  # Log scale for better visualization

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Visualizations saved to: {save_path}")

    def export_results(self, format_type: str = "json", filename: str = None):
        """Export results to various formats."""
        if not self.results:
            print("No analysis results available.")
            return

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_sentiment_analysis_{timestamp}"

        if format_type.lower() == "json":
            filepath = f"{filename}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Results exported to: {filepath}")

        elif format_type.lower() == "csv":
            filepath = f"{filename}.csv"
            df = pd.DataFrame(self.results["comments"])
            # Flatten sentiment probabilities
            for sentiment in ["positive", "negative", "neutral"]:
                df[f"{sentiment}_probability"] = df["sentiment_probabilities"].apply(
                    lambda x: x[sentiment]
                )
            df = df.drop("sentiment_probabilities", axis=1)
            df.to_csv(filepath, index=False, encoding="utf-8")
            print(f"Results exported to: {filepath}")

        elif format_type.lower() == "txt":
            filepath = f"{filename}.txt"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.generate_report())
            print(f"Report exported to: {filepath}")

        else:
            print(f"Unsupported format: {format_type}")


def main():
    """Main function for interactive sentiment analysis."""
    print("YouTube Comment Sentiment Analysis")
    print("=" * 40)

    # Get API key
    api_key = input("Enter your YouTube Data API key: ").strip()
    if not api_key:
        print("API key is required!")
        return

    # Initialize analyzer
    analyzer = YouTubeSentimentAnalyzer(api_key)

    # Setup sentiment model
    print("\nSetting up sentiment analysis model...")
    analyzer.setup_sentiment_model("sentiment_model.pkl")

    while True:
        print("\n" + "-" * 50)
        video_url = input("Enter YouTube video URL (or 'quit' to exit): ").strip()

        if video_url.lower() in ["quit", "exit", "q"]:
            break

        if not video_url:
            continue

        try:
            max_comments = int(
                input("Maximum comments to analyze (default 100): ") or "100"
            )
        except ValueError:
            max_comments = 100

        # Analyze comments
        print(f"\nAnalyzing sentiment for up to {max_comments} comments...")
        results = analyzer.analyze_video_comments(video_url, max_comments)

        if results:
            # Display report
            print("\n" + analyzer.generate_report())

            # Ask for visualizations
            show_viz = input("\nGenerate visualizations? (y/n): ").strip().lower()
            if show_viz in ["y", "yes"]:
                analyzer.create_visualizations()

            # Ask for export
            export_choice = (
                input("\nExport results? (json/csv/txt/n): ").strip().lower()
            )
            if export_choice in ["json", "csv", "txt"]:
                analyzer.export_results(export_choice)

        continue_choice = input("\nAnalyze another video? (y/n): ").strip().lower()
        if continue_choice not in ["y", "yes"]:
            break

    print("Thank you for using YouTube Sentiment Analysis!")


if __name__ == "__main__":
    main()
