#!/usr/bin/env python3
"""
Data Loading Utilities
Utilities for loading and preparing training data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data for sentiment analysis."""
    
    @staticmethod
    def load_csv_dataset(filepath: str, text_column: str = 'text', label_column: str = 'sentiment') -> pd.DataFrame:
        """Load dataset from CSV file."""
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in dataset")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in dataset")
            
            # Clean data
            df = df.dropna(subset=[text_column, label_column])
            df[text_column] = df[text_column].astype(str)
            df[label_column] = df[label_column].astype(str).str.lower()
            
            logger.info(f"Loaded dataset with {len(df)} samples from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset from {filepath}: {e}")
            raise
    
    @staticmethod
    def create_sample_dataset(size: int = 1000) -> pd.DataFrame:
        """Create a sample dataset for testing."""
        positive_samples = [
            "This is absolutely amazing!",
            "I love this so much!",
            "Fantastic work, really impressed!",
            "Outstanding quality and service!",
            "Couldn't be happier with this!",
            "Excellent product, highly recommend!",
            "This exceeded my expectations!",
            "Brilliant, just brilliant!",
            "Perfect in every way!",
            "Amazing experience, will definitely return!"
        ]
        
        negative_samples = [
            "This is terrible, completely disappointed.",
            "Worst experience ever, avoid at all costs.",
            "Absolutely horrible, waste of time and money.",
            "Disgusting quality, very poor service.",
            "I hate this, completely useless.",
            "Awful product, doesn't work at all.",
            "Terrible customer service, very rude.",
            "Complete garbage, total waste.",
            "Horrible experience, never again.",
            "Pathetic quality, extremely disappointed."
        ]
        
        neutral_samples = [
            "It's okay, nothing special.",
            "Average quality, does the job.",
            "Not bad, but not great either.",
            "It's fine, meets basic expectations.",
            "Decent enough, could be better.",
            "Acceptable quality for the price.",
            "It works as expected, nothing more.",
            "Standard product, nothing remarkable.",
            "Fair quality, reasonable price.",
            "It's alright, serves its purpose."
        ]
        
        # Generate samples
        samples = []
        samples_per_class = size // 3
        
        for _ in range(samples_per_class):
            samples.append({'text': random.choice(positive_samples), 'sentiment': 'positive'})
            samples.append({'text': random.choice(negative_samples), 'sentiment': 'negative'})
            samples.append({'text': random.choice(neutral_samples), 'sentiment': 'neutral'})
        
        # Add remaining samples to reach exact size
        remaining = size - len(samples)
        all_samples = positive_samples + negative_samples + neutral_samples
        all_labels = ['positive'] * len(positive_samples) + ['negative'] * len(negative_samples) + ['neutral'] * len(neutral_samples)
        
        for _ in range(remaining):
            idx = random.randint(0, len(all_samples) - 1)
            samples.append({'text': all_samples[idx], 'sentiment': all_labels[idx]})
        
        # Shuffle samples
        random.shuffle(samples)
        
        df = pd.DataFrame(samples)
        logger.info(f"Created sample dataset with {len(df)} samples")
        return df
    
    @staticmethod
    def balance_dataset(df: pd.DataFrame, text_column: str = 'text', label_column: str = 'sentiment', 
                       method: str = 'undersample') -> pd.DataFrame:
        """Balance dataset by class."""
        label_counts = df[label_column].value_counts()
        logger.info(f"Original distribution: {dict(label_counts)}")
        
        if method == 'undersample':
            # Undersample to smallest class
            min_count = label_counts.min()
            balanced_dfs = []
            
            for label in label_counts.index:
                label_df = df[df[label_column] == label].sample(n=min_count, random_state=42)
                balanced_dfs.append(label_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif method == 'oversample':
            # Oversample to largest class
            max_count = label_counts.max()
            balanced_dfs = []
            
            for label in label_counts.index:
                label_df = df[df[label_column] == label]
                if len(label_df) < max_count:
                    # Oversample with replacement
                    additional_samples = max_count - len(label_df)
                    oversampled = label_df.sample(n=additional_samples, replace=True, random_state=42)
                    label_df = pd.concat([label_df, oversampled], ignore_index=True)
                balanced_dfs.append(label_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        new_counts = balanced_df[label_column].value_counts()
        logger.info(f"Balanced distribution: {dict(new_counts)}")
        
        return balanced_df
    
    @staticmethod
    def split_dataset(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, 
                     text_column: str = 'text', label_column: str = 'sentiment') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, 
            stratify=df[label_column]
        )
        
        # Second split: train vs val
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size_adjusted, random_state=42,
                stratify=train_val_df[label_column]
            )
        else:
            train_df = train_val_df
            val_df = pd.DataFrame()
        
        logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def get_dataset_statistics(df: pd.DataFrame, text_column: str = 'text', label_column: str = 'sentiment') -> Dict:
        """Get comprehensive dataset statistics."""
        stats = {
            'total_samples': len(df),
            'label_distribution': dict(df[label_column].value_counts()),
            'label_percentages': dict(df[label_column].value_counts(normalize=True) * 100),
            'text_statistics': {
                'avg_length': df[text_column].str.len().mean(),
                'min_length': df[text_column].str.len().min(),
                'max_length': df[text_column].str.len().max(),
                'avg_words': df[text_column].str.split().str.len().mean(),
                'min_words': df[text_column].str.split().str.len().min(),
                'max_words': df[text_column].str.split().str.len().max()
            },
            'missing_values': {
                'text': df[text_column].isnull().sum(),
                'labels': df[label_column].isnull().sum()
            }
        }
        
        return stats


class DataAugmenter:
    """Data augmentation utilities."""
    
    @staticmethod
    def synonym_replacement(text: str, n: int = 1) -> str:
        """Replace n words with synonyms."""
        # Simple synonym dictionary (in practice, use WordNet or similar)
        synonyms = {
            'good': ['great', 'excellent', 'fantastic', 'wonderful'],
            'bad': ['terrible', 'awful', 'horrible', 'disgusting'],
            'like': ['love', 'enjoy', 'appreciate', 'adore'],
            'hate': ['dislike', 'despise', 'loathe', 'detest'],
            'amazing': ['incredible', 'outstanding', 'remarkable', 'extraordinary'],
            'terrible': ['awful', 'horrible', 'dreadful', 'appalling']
        }
        
        words = text.split()
        new_words = words.copy()
        
        # Randomly replace n words
        replaceable_indices = [i for i, word in enumerate(words) if word.lower() in synonyms]
        
        if replaceable_indices:
            indices_to_replace = random.sample(replaceable_indices, min(n, len(replaceable_indices)))
            
            for idx in indices_to_replace:
                original_word = words[idx].lower()
                if original_word in synonyms:
                    replacement = random.choice(synonyms[original_word])
                    # Preserve original case
                    if words[idx][0].isupper():
                        replacement = replacement.capitalize()
                    new_words[idx] = replacement
        
        return ' '.join(new_words)
    
    @staticmethod
    def random_insertion(text: str, n: int = 1) -> str:
        """Randomly insert n words."""
        words = text.split()
        
        # Common words to insert
        insertable_words = ['really', 'very', 'quite', 'extremely', 'totally', 'absolutely']
        
        for _ in range(n):
            new_word = random.choice(insertable_words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, new_word)
        
        return ' '.join(words)
    
    @staticmethod
    def random_swap(text: str, n: int = 1) -> str:
        """Randomly swap n pairs of words."""
        words = text.split()
        
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    @staticmethod
    def random_deletion(text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        # If all words were deleted, return original
        if not new_words:
            return text
        
        return ' '.join(new_words)
    
    @staticmethod
    def augment_dataset(df: pd.DataFrame, augmentation_factor: float = 0.5, 
                       text_column: str = 'text', label_column: str = 'sentiment') -> pd.DataFrame:
        """Augment dataset using various techniques."""
        augmented_samples = []
        
        for _, row in df.iterrows():
            text = row[text_column]
            label = row[label_column]
            
            # Original sample
            augmented_samples.append({text_column: text, label_column: label})
            
            # Apply augmentation with given probability
            if random.random() < augmentation_factor:
                # Choose random augmentation technique
                technique = random.choice([
                    DataAugmenter.synonym_replacement,
                    DataAugmenter.random_insertion,
                    DataAugmenter.random_swap,
                    DataAugmenter.random_deletion
                ])
                
                try:
                    augmented_text = technique(text)
                    if augmented_text != text:  # Only add if actually changed
                        augmented_samples.append({text_column: augmented_text, label_column: label})
                except Exception as e:
                    logger.warning(f"Augmentation failed for text '{text[:50]}...': {e}")
        
        augmented_df = pd.DataFrame(augmented_samples)
        logger.info(f"Augmented dataset from {len(df)} to {len(augmented_df)} samples")
        
        return augmented_df
