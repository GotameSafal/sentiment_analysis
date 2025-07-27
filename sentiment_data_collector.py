#!/usr/bin/env python3
"""
Sentiment Data Collector

This script helps collect and prepare sentiment analysis training data from various sources:
1. Manual labeling interface for YouTube comments
2. Import from existing datasets
3. Data augmentation techniques
4. Export labeled data for model training

Usage:
    python sentiment_data_collector.py
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import os
from datetime import datetime
import random
import re


class SentimentDataCollector:
    """Collect and manage sentiment analysis training data."""
    
    def __init__(self):
        self.labeled_data = []
        self.current_batch = []
        
    def load_youtube_comments(self, json_file: str) -> List[Dict[str, Any]]:
        """Load comments from YouTube comment extractor output."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            comments = []
            for comment in data.get('comments', []):
                comments.append({
                    'text': comment['text'],
                    'author': comment['author'],
                    'like_count': comment['like_count'],
                    'reply_count': comment['reply_count'],
                    'published_at': comment.get('published_at', ''),
                    'source': 'youtube'
                })
            
            print(f"Loaded {len(comments)} comments from {json_file}")
            return comments
        
        except Exception as e:
            print(f"Error loading comments: {e}")
            return []
    
    def manual_labeling_interface(self, comments: List[Dict[str, Any]], 
                                 start_index: int = 0) -> List[Dict[str, Any]]:
        """Interactive interface for manually labeling comments."""
        print("\nManual Sentiment Labeling Interface")
        print("=" * 40)
        print("Instructions:")
        print("- Enter 'p' for positive sentiment")
        print("- Enter 'n' for negative sentiment")
        print("- Enter 'u' for neutral sentiment")
        print("- Enter 's' to skip this comment")
        print("- Enter 'q' to quit and save progress")
        print("- Enter 'b' to go back to previous comment")
        print()
        
        labeled_comments = []
        i = start_index
        
        while i < len(comments):
            comment = comments[i]
            
            print(f"\nComment {i+1}/{len(comments)}")
            print(f"Author: {comment['author']}")
            print(f"Likes: {comment['like_count']}")
            print(f"Text: {comment['text']}")
            print("-" * 50)
            
            while True:
                label = input("Sentiment (p/n/u/s/b/q): ").strip().lower()
                
                if label == 'q':
                    print(f"Saving progress... Labeled {len(labeled_comments)} comments.")
                    return labeled_comments
                
                elif label == 'b' and i > 0:
                    i -= 1
                    if labeled_comments:
                        labeled_comments.pop()  # Remove last label
                    break
                
                elif label == 's':
                    i += 1
                    break
                
                elif label in ['p', 'n', 'u']:
                    sentiment_map = {'p': 'positive', 'n': 'negative', 'u': 'neutral'}
                    
                    labeled_comment = comment.copy()
                    labeled_comment['sentiment'] = sentiment_map[label]
                    labeled_comment['labeled_by'] = 'manual'
                    labeled_comment['labeled_at'] = datetime.now().isoformat()
                    
                    labeled_comments.append(labeled_comment)
                    i += 1
                    break
                
                else:
                    print("Invalid input. Please enter p/n/u/s/b/q")
        
        print(f"Labeling completed! Labeled {len(labeled_comments)} comments.")
        return labeled_comments
    
    def create_balanced_dataset(self, labeled_data: List[Dict[str, Any]], 
                               target_size_per_class: int = 500) -> pd.DataFrame:
        """Create a balanced dataset from labeled data."""
        df = pd.DataFrame(labeled_data)
        
        if 'sentiment' not in df.columns:
            print("No sentiment labels found in data.")
            return pd.DataFrame()
        
        balanced_data = []
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = df[df['sentiment'] == sentiment]
            
            if len(sentiment_data) >= target_size_per_class:
                # Sample if we have more than needed
                sampled_data = sentiment_data.sample(n=target_size_per_class, random_state=42)
            else:
                # Use all available data
                sampled_data = sentiment_data
                print(f"Warning: Only {len(sentiment_data)} {sentiment} samples available, "
                      f"target was {target_size_per_class}")
            
            balanced_data.append(sampled_data)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Created balanced dataset with {len(balanced_df)} samples:")
        print(balanced_df['sentiment'].value_counts())
        
        return balanced_df
    
    def augment_data(self, df: pd.DataFrame, augmentation_factor: float = 0.5) -> pd.DataFrame:
        """Apply data augmentation techniques to increase dataset size."""
        augmented_data = []
        
        for _, row in df.iterrows():
            text = row['text']
            sentiment = row['sentiment']
            
            # Original text
            augmented_data.append(row.to_dict())
            
            # Apply augmentation with probability
            if random.random() < augmentation_factor:
                # Synonym replacement (simple version)
                augmented_text = self._simple_synonym_replacement(text)
                if augmented_text != text:
                    aug_row = row.to_dict()
                    aug_row['text'] = augmented_text
                    aug_row['augmented'] = True
                    augmented_data.append(aug_row)
            
            # Random insertion of common words
            if random.random() < augmentation_factor * 0.3:
                augmented_text = self._random_insertion(text)
                if augmented_text != text:
                    aug_row = row.to_dict()
                    aug_row['text'] = augmented_text
                    aug_row['augmented'] = True
                    augmented_data.append(aug_row)
        
        augmented_df = pd.DataFrame(augmented_data)
        print(f"Data augmentation: {len(df)} -> {len(augmented_df)} samples")
        
        return augmented_df
    
    def _simple_synonym_replacement(self, text: str) -> str:
        """Simple synonym replacement for data augmentation."""
        # Basic synonym dictionary
        synonyms = {
            'good': ['great', 'excellent', 'amazing', 'wonderful'],
            'bad': ['terrible', 'awful', 'horrible', 'poor'],
            'like': ['love', 'enjoy', 'appreciate'],
            'hate': ['dislike', 'despise', 'detest'],
            'nice': ['pleasant', 'lovely', 'beautiful'],
            'ugly': ['hideous', 'unattractive', 'unsightly']
        }
        
        words = text.lower().split()
        new_words = []
        
        for word in words:
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in synonyms and random.random() < 0.3:
                # Replace with random synonym
                synonym = random.choice(synonyms[clean_word])
                new_words.append(word.replace(clean_word, synonym))
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def _random_insertion(self, text: str) -> str:
        """Random insertion of common words."""
        common_words = ['really', 'very', 'quite', 'so', 'just', 'actually', 'definitely']
        
        words = text.split()
        if len(words) < 3:
            return text
        
        # Insert a random word at a random position
        insert_pos = random.randint(1, len(words) - 1)
        insert_word = random.choice(common_words)
        
        words.insert(insert_pos, insert_word)
        return ' '.join(words)
    
    def import_existing_dataset(self, file_path: str, text_column: str, 
                               sentiment_column: str) -> pd.DataFrame:
        """Import existing labeled dataset."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                print("Unsupported file format. Use CSV or JSON.")
                return pd.DataFrame()
            
            # Rename columns to standard format
            df = df.rename(columns={text_column: 'text', sentiment_column: 'sentiment'})
            
            # Filter required columns
            df = df[['text', 'sentiment']].copy()
            
            # Standardize sentiment labels
            df['sentiment'] = df['sentiment'].str.lower()
            df['source'] = 'imported'
            df['labeled_at'] = datetime.now().isoformat()
            
            print(f"Imported {len(df)} samples from {file_path}")
            print("Sentiment distribution:")
            print(df['sentiment'].value_counts())
            
            return df
        
        except Exception as e:
            print(f"Error importing dataset: {e}")
            return pd.DataFrame()
    
    def export_training_data(self, df: pd.DataFrame, filename: str = None):
        """Export labeled data for model training."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_training_data_{timestamp}"
        
        # Export as CSV
        csv_file = f"{filename}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Training data exported to: {csv_file}")
        
        # Export as JSON
        json_file = f"{filename}.json"
        df.to_json(json_file, orient='records', indent=2, force_ascii=False)
        print(f"Training data exported to: {json_file}")
        
        # Create summary
        summary = {
            'total_samples': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'export_date': datetime.now().isoformat(),
            'columns': list(df.columns)
        }
        
        summary_file = f"{filename}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary exported to: {summary_file}")
    
    def get_labeling_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the labeled data."""
        stats = {
            'total_samples': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_text_length': df['text'].str.len().mean(),
            'text_length_stats': df['text'].str.len().describe().to_dict()
        }
        
        if 'like_count' in df.columns:
            stats['like_count_by_sentiment'] = df.groupby('sentiment')['like_count'].mean().to_dict()
        
        return stats


def main():
    """Main function for data collection interface."""
    print("Sentiment Analysis Data Collector")
    print("=" * 40)
    
    collector = SentimentDataCollector()
    
    while True:
        print("\nOptions:")
        print("1. Load and label YouTube comments")
        print("2. Import existing dataset")
        print("3. Create balanced dataset")
        print("4. Export training data")
        print("5. View statistics")
        print("6. Quit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            json_file = input("Enter YouTube comments JSON file path: ").strip()
            if os.path.exists(json_file):
                comments = collector.load_youtube_comments(json_file)
                if comments:
                    labeled = collector.manual_labeling_interface(comments)
                    collector.labeled_data.extend(labeled)
                    print(f"Added {len(labeled)} labeled comments to dataset.")
            else:
                print("File not found.")
        
        elif choice == '2':
            file_path = input("Enter dataset file path: ").strip()
            text_col = input("Enter text column name: ").strip()
            sentiment_col = input("Enter sentiment column name: ").strip()
            
            if os.path.exists(file_path):
                imported_df = collector.import_existing_dataset(file_path, text_col, sentiment_col)
                if not imported_df.empty:
                    collector.labeled_data.extend(imported_df.to_dict('records'))
                    print(f"Added {len(imported_df)} samples to dataset.")
            else:
                print("File not found.")
        
        elif choice == '3':
            if collector.labeled_data:
                target_size = int(input("Target samples per class (default 500): ") or "500")
                balanced_df = collector.create_balanced_dataset(collector.labeled_data, target_size)
                
                if not balanced_df.empty:
                    augment = input("Apply data augmentation? (y/n): ").strip().lower()
                    if augment in ['y', 'yes']:
                        balanced_df = collector.augment_data(balanced_df)
                    
                    collector.current_batch = balanced_df.to_dict('records')
                    print("Balanced dataset created and ready for export.")
            else:
                print("No labeled data available.")
        
        elif choice == '4':
            if collector.current_batch:
                filename = input("Enter filename (optional): ").strip() or None
                df = pd.DataFrame(collector.current_batch)
                collector.export_training_data(df, filename)
            else:
                print("No data ready for export. Create balanced dataset first.")
        
        elif choice == '5':
            if collector.labeled_data:
                df = pd.DataFrame(collector.labeled_data)
                stats = collector.get_labeling_statistics(df)
                print("\nDataset Statistics:")
                print(f"Total samples: {stats['total_samples']}")
                print(f"Sentiment distribution: {stats['sentiment_distribution']}")
                print(f"Average text length: {stats['average_text_length']:.1f} characters")
            else:
                print("No data available.")
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-6.")


if __name__ == "__main__":
    main()
