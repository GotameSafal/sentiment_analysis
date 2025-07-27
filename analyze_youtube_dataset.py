#!/usr/bin/env python3
"""
Quick Analysis of YouTube Comments Dataset

This script provides a quick overview of your YouTube comments dataset
to understand its structure and characteristics before training.

Usage:
    python analyze_youtube_dataset.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re


def analyze_dataset(csv_file: str = "YoutubeCommentsDataSet.csv"):
    """Analyze the YouTube comments dataset."""
    print("📊 YouTube Comments Dataset Analysis")
    print("=" * 50)
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_file)
    
    # Basic information
    print(f"\n📈 BASIC STATISTICS:")
    print(f"Total comments: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values:")
        print(missing[missing > 0])
    else:
        print(f"\n✅ No missing values found")
    
    # Sentiment distribution
    print(f"\n🎭 SENTIMENT DISTRIBUTION:")
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_percentages = (sentiment_counts / len(df) * 100).round(2)
    
    for sentiment in sentiment_counts.index:
        count = sentiment_counts[sentiment]
        percentage = sentiment_percentages[sentiment]
        print(f"  {sentiment.capitalize():<10}: {count:>6,} ({percentage:>5.1f}%)")
    
    # Text length analysis
    df['text_length'] = df['Comment'].str.len()
    df['word_count'] = df['Comment'].str.split().str.len()
    
    print(f"\n📝 TEXT LENGTH ANALYSIS:")
    print(f"Character length stats:")
    print(f"  Mean: {df['text_length'].mean():.1f}")
    print(f"  Median: {df['text_length'].median():.1f}")
    print(f"  Min: {df['text_length'].min()}")
    print(f"  Max: {df['text_length'].max()}")
    
    print(f"\nWord count stats:")
    print(f"  Mean: {df['word_count'].mean():.1f}")
    print(f"  Median: {df['word_count'].median():.1f}")
    print(f"  Min: {df['word_count'].min()}")
    print(f"  Max: {df['word_count'].max()}")
    
    # Sample comments from each sentiment
    print(f"\n💬 SAMPLE COMMENTS:")
    for sentiment in df['Sentiment'].unique():
        print(f"\n{sentiment.upper()} Examples:")
        samples = df[df['Sentiment'] == sentiment]['Comment'].head(2)
        for i, comment in enumerate(samples, 1):
            truncated = comment[:100] + "..." if len(comment) > 100 else comment
            print(f"  {i}. {truncated}")
    
    # Most common words by sentiment
    print(f"\n🔤 MOST COMMON WORDS BY SENTIMENT:")
    for sentiment in df['Sentiment'].unique():
        comments = df[df['Sentiment'] == sentiment]['Comment'].str.lower()
        all_words = ' '.join(comments).split()
        # Remove common stop words and punctuation
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'a', 'an', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        filtered_words = [word for word in all_words if word.isalpha() and len(word) > 2 and word not in stop_words]
        common_words = Counter(filtered_words).most_common(10)
        
        print(f"\n{sentiment.capitalize()}:")
        for word, count in common_words:
            print(f"  {word}: {count}")
    
    # Dataset quality assessment
    print(f"\n🔍 DATASET QUALITY ASSESSMENT:")
    
    # Check for very short comments
    very_short = df[df['text_length'] < 10]
    print(f"Very short comments (<10 chars): {len(very_short)} ({len(very_short)/len(df)*100:.1f}%)")
    
    # Check for very long comments
    very_long = df[df['text_length'] > 500]
    print(f"Very long comments (>500 chars): {len(very_long)} ({len(very_long)/len(df)*100:.1f}%)")
    
    # Check for potential duplicates
    duplicates = df.duplicated(subset=['Comment']).sum()
    print(f"Potential duplicate comments: {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    # Balance assessment
    min_sentiment = sentiment_counts.min()
    max_sentiment = sentiment_counts.max()
    balance_ratio = min_sentiment / max_sentiment
    print(f"Dataset balance ratio: {balance_ratio:.2f} (1.0 = perfectly balanced)")
    
    if balance_ratio < 0.5:
        print("⚠️  Dataset is significantly imbalanced - consider balancing during training")
    elif balance_ratio < 0.8:
        print("⚠️  Dataset is moderately imbalanced - balancing might help")
    else:
        print("✅ Dataset is reasonably balanced")
    
    # Recommendations
    print(f"\n💡 TRAINING RECOMMENDATIONS:")
    print(f"✓ Dataset size is good for training ({len(df):,} samples)")
    
    if len(df) > 10000:
        print("✓ Large dataset - can use complex models (SVM, ensemble methods)")
    elif len(df) > 5000:
        print("✓ Medium dataset - good for most ML models")
    else:
        print("⚠️  Small dataset - consider simpler models or data augmentation")
    
    if balance_ratio < 0.7:
        print("• Consider using balanced training data")
        print("• Use stratified cross-validation")
        print("• Consider class weights in models")
    
    if df['text_length'].mean() > 200:
        print("• Comments are relatively long - TF-IDF with n-grams should work well")
    else:
        print("• Comments are relatively short - consider character-level features")
    
    print(f"\n🚀 Ready to train! Run: python train_with_youtube_dataset.py")
    
    return df


def create_quick_visualization(df):
    """Create a quick visualization of the dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('YouTube Comments Dataset Overview', fontsize=14, fontweight='bold')
    
    # Sentiment distribution
    sentiment_counts = df['Sentiment'].value_counts()
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Sentiment Distribution')
    
    # Text length histogram
    axes[0, 1].hist(df['text_length'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Comment Length Distribution')
    axes[0, 1].set_xlabel('Characters')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim(0, 1000)
    
    # Word count by sentiment
    for sentiment in df['Sentiment'].unique():
        data = df[df['Sentiment'] == sentiment]['word_count']
        axes[1, 0].hist(data, alpha=0.6, label=sentiment, bins=30)
    axes[1, 0].set_title('Word Count by Sentiment')
    axes[1, 0].set_xlabel('Word Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 100)
    
    # Box plot of text lengths
    sentiment_data = [df[df['Sentiment'] == sentiment]['text_length'] 
                     for sentiment in df['Sentiment'].unique()]
    axes[1, 1].boxplot(sentiment_data, labels=df['Sentiment'].unique())
    axes[1, 1].set_title('Text Length by Sentiment')
    axes[1, 1].set_ylabel('Characters')
    
    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualization saved as 'dataset_overview.png'")


def main():
    """Main analysis function."""
    try:
        df = analyze_dataset()
        
        # Ask if user wants visualization
        create_viz = input("\nCreate visualization? (y/n): ").strip().lower()
        if create_viz in ['y', 'yes']:
            create_quick_visualization(df)
        
        print(f"\n✅ Analysis complete!")
        
    except FileNotFoundError:
        print("❌ Error: YoutubeCommentsDataSet.csv not found!")
        print("   Make sure the file is in the current directory.")
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")


if __name__ == "__main__":
    main()
