#!/usr/bin/env python3
"""
Continuous Learning Sentiment Analysis

This enhanced version can:
1. Use existing trained model for analysis
2. Collect new YouTube comments for potential training
3. Allow manual labeling of new comments
4. Retrain model with expanded dataset
5. Implement active learning for uncertain predictions

Usage:
    python continuous_learning_sentiment.py
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Any, Tuple
from youtube_comment_extractor import YouTubeCommentExtractor
from sentiment_analyzer import SentimentAnalyzer, SentimentDataset
import warnings
warnings.filterwarnings('ignore')


class ContinuousLearningSentimentAnalyzer:
    """Enhanced sentiment analyzer with continuous learning capabilities."""
    
    def __init__(self, api_key: str):
        self.comment_extractor = YouTubeCommentExtractor(api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.new_training_data = []
        self.uncertain_predictions = []
        self.confidence_threshold = 0.7  # Below this = uncertain
        
    def setup_model(self, model_path: str = "sentiment_model.pkl"):
        """Load existing model or train new one."""
        if os.path.exists(model_path):
            print(f"Loading existing model: {model_path}")
            self.sentiment_analyzer.load_model(model_path)
            return True
        else:
            print("No existing model found. Please train a model first.")
            return False
    
    def analyze_with_learning_opportunity(self, video_url: str, max_comments: int = 100) -> Dict[str, Any]:
        """Analyze video comments and identify learning opportunities."""
        print(f"Analyzing video: {video_url}")
        
        # Extract comments
        comments = self.comment_extractor.extract_comments_from_url(
            video_url, max_comments, save_to_file=False
        )
        
        if not comments:
            return {}
        
        # Analyze sentiment and track confidence
        analyzed_comments = []
        uncertain_comments = []
        high_confidence_comments = []
        
        for comment in comments:
            result = self.sentiment_analyzer.predict_sentiment(comment['text'])
            
            analyzed_comment = {
                'author': comment['author'],
                'text': comment['text'],
                'like_count': comment['like_count'],
                'reply_count': comment['reply_count'],
                'published_at': comment['published_at'],
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'video_url': video_url,
                'analysis_date': datetime.now().isoformat()
            }
            
            analyzed_comments.append(analyzed_comment)
            
            # Categorize by confidence
            if result['confidence'] < self.confidence_threshold:
                uncertain_comments.append(analyzed_comment)
            else:
                high_confidence_comments.append(analyzed_comment)
        
        # Store uncertain predictions for potential manual labeling
        self.uncertain_predictions.extend(uncertain_comments)
        
        # Calculate statistics
        sentiments = [c['sentiment'] for c in analyzed_comments]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        # Confidence statistics
        confidences = [c['confidence'] for c in analyzed_comments]
        avg_confidence = np.mean(confidences)
        low_confidence_count = len(uncertain_comments)
        
        results = {
            'video_url': video_url,
            'total_comments': len(analyzed_comments),
            'comments': analyzed_comments,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': (sentiment_counts / len(analyzed_comments) * 100).to_dict(),
            'average_confidence': avg_confidence,
            'uncertain_predictions': low_confidence_count,
            'uncertain_percentage': (low_confidence_count / len(analyzed_comments)) * 100,
            'high_confidence_comments': len(high_confidence_comments),
            'learning_opportunities': low_confidence_count
        }
        
        # Print analysis summary
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"Total Comments: {len(analyzed_comments)}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Uncertain Predictions: {low_confidence_count} ({(low_confidence_count/len(analyzed_comments)*100):.1f}%)")
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(analyzed_comments)) * 100
            print(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        return results
    
    def review_uncertain_predictions(self, max_review: int = 10) -> List[Dict]:
        """Review and manually label uncertain predictions."""
        if not self.uncertain_predictions:
            print("No uncertain predictions to review.")
            return []
        
        print(f"\n🔍 REVIEWING UNCERTAIN PREDICTIONS")
        print(f"Found {len(self.uncertain_predictions)} uncertain predictions")
        print("Let's review the most uncertain ones for potential retraining...")
        
        # Sort by confidence (lowest first)
        sorted_uncertain = sorted(self.uncertain_predictions, key=lambda x: x['confidence'])
        to_review = sorted_uncertain[:max_review]
        
        labeled_data = []
        
        print("\nInstructions:")
        print("- Enter 'p' for positive")
        print("- Enter 'n' for negative") 
        print("- Enter 'u' for neutral")
        print("- Enter 's' to skip")
        print("- Enter 'q' to quit review")
        
        for i, comment in enumerate(to_review):
            print(f"\n--- Comment {i+1}/{len(to_review)} ---")
            print(f"Text: {comment['text']}")
            print(f"Model predicted: {comment['sentiment']} (confidence: {comment['confidence']:.3f})")
            print(f"Author: {comment['author']}, Likes: {comment['like_count']}")
            
            while True:
                user_label = input("Your label (p/n/u/s/q): ").strip().lower()
                
                if user_label == 'q':
                    print("Review stopped.")
                    return labeled_data
                elif user_label == 's':
                    break
                elif user_label in ['p', 'n', 'u']:
                    label_map = {'p': 'positive', 'n': 'negative', 'u': 'neutral'}
                    correct_label = label_map[user_label]
                    
                    # Add to training data
                    training_sample = {
                        'text': comment['text'],
                        'sentiment': correct_label,
                        'confidence': comment['confidence'],
                        'model_prediction': comment['sentiment'],
                        'human_labeled': True,
                        'labeled_date': datetime.now().isoformat(),
                        'source': 'uncertain_review'
                    }
                    
                    labeled_data.append(training_sample)
                    self.new_training_data.append(training_sample)
                    
                    # Check if model was wrong
                    if correct_label != comment['sentiment']:
                        print(f"✓ Correction noted: {comment['sentiment']} → {correct_label}")
                    else:
                        print(f"✓ Model was correct: {correct_label}")
                    break
                else:
                    print("Invalid input. Use p/n/u/s/q")
        
        print(f"\n✅ Review complete! Labeled {len(labeled_data)} comments for retraining.")
        return labeled_data
    
    def add_high_confidence_to_training(self, min_confidence: float = 0.9, max_samples: int = 100):
        """Add high-confidence predictions to training data (pseudo-labeling)."""
        if not hasattr(self, 'last_analysis_results'):
            print("No recent analysis results available.")
            return
        
        high_conf_comments = [
            c for c in self.last_analysis_results.get('comments', [])
            if c['confidence'] >= min_confidence
        ]
        
        if not high_conf_comments:
            print(f"No comments with confidence >= {min_confidence}")
            return
        
        # Sample to avoid overwhelming the training data
        if len(high_conf_comments) > max_samples:
            high_conf_comments = np.random.choice(high_conf_comments, max_samples, replace=False)
        
        print(f"Adding {len(high_conf_comments)} high-confidence predictions to training data...")
        
        for comment in high_conf_comments:
            training_sample = {
                'text': comment['text'],
                'sentiment': comment['sentiment'],
                'confidence': comment['confidence'],
                'human_labeled': False,
                'labeled_date': datetime.now().isoformat(),
                'source': 'high_confidence_pseudo'
            }
            self.new_training_data.append(training_sample)
        
        print(f"✅ Added {len(high_conf_comments)} samples for retraining")
    
    def retrain_with_new_data(self, model_save_path: str = None):
        """Retrain the model with original + new training data."""
        if not self.new_training_data:
            print("No new training data available.")
            return
        
        print(f"\n🔄 RETRAINING MODEL")
        print(f"New training samples: {len(self.new_training_data)}")
        
        # Load original training data
        original_data = []
        if os.path.exists("YoutubeCommentsDataSet.csv"):
            df_original = pd.read_csv("YoutubeCommentsDataSet.csv")
            df_original = df_original.rename(columns={'Comment': 'text', 'Sentiment': 'sentiment'})
            df_original['sentiment'] = df_original['sentiment'].str.lower()
            original_data = df_original.to_dict('records')
            print(f"Original training samples: {len(original_data)}")
        
        # Combine original + new data
        all_training_data = original_data + self.new_training_data
        combined_df = pd.DataFrame(all_training_data)
        
        print(f"Combined training samples: {len(combined_df)}")
        print("New sentiment distribution:")
        print(combined_df['sentiment'].value_counts())
        
        # Retrain the model
        print("Retraining model...")
        results = self.sentiment_analyzer.train_model(combined_df, model_type='logistic')
        
        # Save updated model
        if model_save_path:
            self.sentiment_analyzer.save_model(model_save_path)
            print(f"✅ Updated model saved: {model_save_path}")
        
        # Save new training data for future use
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_data_file = f"new_training_data_{timestamp}.csv"
        pd.DataFrame(self.new_training_data).to_csv(new_data_file, index=False)
        print(f"📄 New training data saved: {new_data_file}")
        
        # Clear new training data
        self.new_training_data = []
        
        return results
    
    def save_analysis_with_learning_data(self, results: Dict, filename: str = None):
        """Save analysis results including learning opportunities."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_with_learning_{timestamp}.json"
        
        # Add learning metadata
        results['learning_metadata'] = {
            'uncertain_predictions_available': len(self.uncertain_predictions),
            'new_training_samples': len(self.new_training_data),
            'confidence_threshold': self.confidence_threshold,
            'can_retrain': len(self.new_training_data) > 0
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 Analysis with learning data saved: {filename}")


def main():
    """Main continuous learning workflow."""
    print("🔄 Continuous Learning Sentiment Analysis")
    print("=" * 50)
    
    # Get API key
    api_key = input("Enter your YouTube Data API key: ").strip()
    if not api_key:
        print("API key required!")
        return
    
    # Initialize analyzer
    analyzer = ContinuousLearningSentimentAnalyzer(api_key)
    
    # Load existing model
    if not analyzer.setup_model():
        print("Please train a model first using train_with_youtube_dataset.py")
        return
    
    while True:
        print("\n" + "="*50)
        print("OPTIONS:")
        print("1. Analyze YouTube video")
        print("2. Review uncertain predictions")
        print("3. Add high-confidence predictions to training")
        print("4. Retrain model with new data")
        print("5. View learning statistics")
        print("6. Quit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            video_url = input("Enter YouTube video URL: ").strip()
            if video_url:
                max_comments = int(input("Max comments to analyze (default 100): ") or "100")
                results = analyzer.analyze_with_learning_opportunity(video_url, max_comments)
                analyzer.last_analysis_results = results
                
                if results:
                    save_results = input("Save analysis results? (y/n): ").strip().lower()
                    if save_results in ['y', 'yes']:
                        analyzer.save_analysis_with_learning_data(results)
        
        elif choice == '2':
            max_review = int(input("Max comments to review (default 10): ") or "10")
            labeled = analyzer.review_uncertain_predictions(max_review)
            if labeled:
                print(f"✅ {len(labeled)} comments labeled for retraining")
        
        elif choice == '3':
            if hasattr(analyzer, 'last_analysis_results'):
                min_conf = float(input("Minimum confidence (default 0.9): ") or "0.9")
                max_samples = int(input("Max samples to add (default 100): ") or "100")
                analyzer.add_high_confidence_to_training(min_conf, max_samples)
            else:
                print("No recent analysis available. Analyze a video first.")
        
        elif choice == '4':
            if analyzer.new_training_data:
                print(f"Ready to retrain with {len(analyzer.new_training_data)} new samples")
                confirm = input("Proceed with retraining? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    analyzer.retrain_with_new_data("updated_sentiment_model.pkl")
            else:
                print("No new training data available.")
        
        elif choice == '5':
            print(f"\n📊 LEARNING STATISTICS:")
            print(f"Uncertain predictions collected: {len(analyzer.uncertain_predictions)}")
            print(f"New training samples ready: {len(analyzer.new_training_data)}")
            print(f"Confidence threshold: {analyzer.confidence_threshold}")
            
            if analyzer.new_training_data:
                df_new = pd.DataFrame(analyzer.new_training_data)
                print(f"New data sentiment distribution:")
                print(df_new['sentiment'].value_counts())
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
