#!/usr/bin/env python3
"""
Data Quality Enhancement for Higher Accuracy
Strategies to improve training data quality:
1. Remove noisy/ambiguous samples
2. Data augmentation
3. Active learning for difficult cases
4. External dataset integration
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random


class DataQualityEnhancer:
    """Enhance data quality for better model performance."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def detect_noisy_samples(self, df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """Detect and remove potentially mislabeled samples."""
        print("🔍 Detecting noisy samples...")
        
        clean_samples = []
        
        for sentiment in df['sentiment'].unique():
            sentiment_data = df[df['sentiment'] == sentiment].copy()
            
            if len(sentiment_data) < 10:
                clean_samples.append(sentiment_data)
                continue
            
            # Vectorize texts
            texts = sentiment_data['text'].astype(str).tolist()
            vectors = self.vectorizer.fit_transform(texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(vectors)
            
            # Find samples with low similarity to others in same class
            avg_similarities = []
            for i in range(len(sentiment_data)):
                # Average similarity to other samples in same class
                similarities = similarity_matrix[i]
                avg_sim = np.mean([sim for j, sim in enumerate(similarities) if j != i])
                avg_similarities.append(avg_sim)
            
            # Keep samples above threshold
            keep_indices = [i for i, sim in enumerate(avg_similarities) if sim >= threshold]
            clean_sentiment_data = sentiment_data.iloc[keep_indices]
            
            removed_count = len(sentiment_data) - len(clean_sentiment_data)
            print(f"  {sentiment}: Removed {removed_count}/{len(sentiment_data)} noisy samples")
            
            clean_samples.append(clean_sentiment_data)
        
        clean_df = pd.concat(clean_samples, ignore_index=True)
        print(f"✅ Clean dataset size: {len(clean_df)} (removed {len(df) - len(clean_df)} samples)")
        
        return clean_df
    
    def augment_data(self, df: pd.DataFrame, augment_factor: float = 0.3) -> pd.DataFrame:
        """Augment data using various techniques."""
        print("🔄 Augmenting data...")
        
        augmented_samples = []
        
        for _, row in df.iterrows():
            text = str(row['text'])
            sentiment = row['sentiment']
            
            # Original sample
            augmented_samples.append({'text': text, 'sentiment': sentiment})
            
            # Random augmentation based on factor
            if random.random() < augment_factor:
                # Technique 1: Synonym replacement (simplified)
                augmented_text = self._simple_synonym_replacement(text)
                if augmented_text != text:
                    augmented_samples.append({'text': augmented_text, 'sentiment': sentiment})
                
                # Technique 2: Random insertion of punctuation
                if random.random() < 0.5:
                    augmented_text = self._add_emphasis(text)
                    augmented_samples.append({'text': augmented_text, 'sentiment': sentiment})
        
        augmented_df = pd.DataFrame(augmented_samples)
        print(f"✅ Augmented dataset size: {len(augmented_df)} (added {len(augmented_df) - len(df)} samples)")
        
        return augmented_df
    
    def _simple_synonym_replacement(self, text: str) -> str:
        """Simple synonym replacement for data augmentation."""
        # Basic synonym dictionary
        synonyms = {
            'good': ['great', 'excellent', 'awesome', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'disgusting'],
            'like': ['love', 'enjoy', 'appreciate'],
            'hate': ['dislike', 'despise', 'loathe'],
            'amazing': ['incredible', 'outstanding', 'remarkable'],
            'stupid': ['dumb', 'foolish', 'ridiculous']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in synonyms and random.random() < 0.3:
                replacement = random.choice(synonyms[word_lower])
                # Preserve original case
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = word.replace(word_lower, replacement)
        
        return ' '.join(words)
    
    def _add_emphasis(self, text: str) -> str:
        """Add emphasis through punctuation."""
        if random.random() < 0.5:
            # Add exclamation marks
            if not text.endswith(('!', '?', '.')):
                text += '!'
            else:
                text = text[:-1] + '!!'
        
        return text
    
    def balance_dataset_advanced(self, df: pd.DataFrame, strategy: str = 'oversample') -> pd.DataFrame:
        """Advanced dataset balancing."""
        print(f"⚖️  Balancing dataset using {strategy}...")
        
        sentiment_counts = df['sentiment'].value_counts()
        print(f"Original distribution: {dict(sentiment_counts)}")
        
        if strategy == 'oversample':
            # Oversample minority classes
            max_count = sentiment_counts.max()
            balanced_samples = []
            
            for sentiment in df['sentiment'].unique():
                sentiment_data = df[df['sentiment'] == sentiment]
                current_count = len(sentiment_data)
                
                if current_count < max_count:
                    # Oversample by repeating with slight variations
                    additional_needed = max_count - current_count
                    additional_samples = []
                    
                    for _ in range(additional_needed):
                        # Randomly select a sample and slightly modify it
                        sample = sentiment_data.sample(1).iloc[0]
                        modified_text = self._simple_synonym_replacement(sample['text'])
                        additional_samples.append({
                            'text': modified_text,
                            'sentiment': sentiment
                        })
                    
                    # Combine original and additional samples
                    combined_data = pd.concat([
                        sentiment_data,
                        pd.DataFrame(additional_samples)
                    ], ignore_index=True)
                    
                    balanced_samples.append(combined_data)
                else:
                    balanced_samples.append(sentiment_data)
            
            balanced_df = pd.concat(balanced_samples, ignore_index=True)
            
        elif strategy == 'undersample':
            # Undersample majority classes
            min_count = sentiment_counts.min()
            balanced_samples = []
            
            for sentiment in df['sentiment'].unique():
                sentiment_data = df[df['sentiment'] == sentiment].sample(n=min_count, random_state=42)
                balanced_samples.append(sentiment_data)
            
            balanced_df = pd.concat(balanced_samples, ignore_index=True)
        
        else:  # 'hybrid'
            # Hybrid approach: moderate oversampling + undersampling
            target_count = int(sentiment_counts.mean())
            balanced_samples = []
            
            for sentiment in df['sentiment'].unique():
                sentiment_data = df[df['sentiment'] == sentiment]
                current_count = len(sentiment_data)
                
                if current_count > target_count:
                    # Undersample
                    sentiment_data = sentiment_data.sample(n=target_count, random_state=42)
                elif current_count < target_count:
                    # Oversample
                    additional_needed = target_count - current_count
                    additional_samples = []
                    
                    for _ in range(additional_needed):
                        sample = sentiment_data.sample(1).iloc[0]
                        modified_text = self._simple_synonym_replacement(sample['text'])
                        additional_samples.append({
                            'text': modified_text,
                            'sentiment': sentiment
                        })
                    
                    sentiment_data = pd.concat([
                        sentiment_data,
                        pd.DataFrame(additional_samples)
                    ], ignore_index=True)
                
                balanced_samples.append(sentiment_data)
            
            balanced_df = pd.concat(balanced_samples, ignore_index=True)
        
        new_counts = balanced_df['sentiment'].value_counts()
        print(f"Balanced distribution: {dict(new_counts)}")
        
        return balanced_df
    
    def create_high_quality_dataset(self, csv_file: str = "YoutubeCommentsDataSet.csv") -> pd.DataFrame:
        """Create a high-quality dataset for training."""
        print("🚀 Creating High-Quality Dataset")
        print("=" * 50)
        
        # Load original data
        df = pd.read_csv(csv_file)
        df = df.dropna()
        df.columns = ['text', 'sentiment']
        df['sentiment'] = df['sentiment'].str.lower()
        
        print(f"Original dataset size: {len(df)}")
        
        # Step 1: Remove noisy samples
        clean_df = self.detect_noisy_samples(df, threshold=0.2)
        
        # Step 2: Balance dataset
        balanced_df = self.balance_dataset_advanced(clean_df, strategy='hybrid')
        
        # Step 3: Augment data
        final_df = self.augment_data(balanced_df, augment_factor=0.2)
        
        # Shuffle the final dataset
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✅ Final high-quality dataset size: {len(final_df)}")
        print(f"Final distribution: {dict(final_df['sentiment'].value_counts())}")
        
        return final_df


def main():
    """Create high-quality dataset for improved accuracy."""
    enhancer = DataQualityEnhancer()
    
    # Create high-quality dataset
    high_quality_df = enhancer.create_high_quality_dataset()
    
    # Save the enhanced dataset
    output_file = "high_quality_sentiment_dataset.csv"
    high_quality_df.to_csv(output_file, index=False)
    print(f"\n💾 High-quality dataset saved as: {output_file}")
    
    # Show sample data
    print("\n📝 Sample data from each class:")
    for sentiment in high_quality_df['sentiment'].unique():
        sample = high_quality_df[high_quality_df['sentiment'] == sentiment].sample(1)
        print(f"\n{sentiment.upper()}: {sample['text'].iloc[0][:100]}...")


if __name__ == "__main__":
    main()
