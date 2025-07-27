#!/usr/bin/env python3
"""
Train Sentiment Model with YouTube Comments Dataset

This script trains sentiment analysis models using the existing YouTube comments CSV dataset.
It provides comprehensive analysis, model comparison, and saves the best performing model.

Usage:
    python train_with_youtube_dataset.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from sentiment_analyzer import TextPreprocessor, SentimentAnalyzer


class YouTubeDatasetTrainer:
    """Train sentiment models using the YouTube comments dataset."""
    
    def __init__(self, csv_file: str = "YoutubeCommentsDataSet.csv"):
        self.csv_file = csv_file
        self.df = None
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_explore_data(self):
        """Load and explore the YouTube comments dataset."""
        print("Loading YouTube Comments Dataset...")
        print("=" * 50)
        
        # Load the dataset
        self.df = pd.read_csv(self.csv_file)
        
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(self.df)}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Basic statistics
        print(f"\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print(f"\nMissing values:")
        print(missing_values)
        
        # Remove any missing values
        initial_size = len(self.df)
        self.df = self.df.dropna()
        final_size = len(self.df)
        if initial_size != final_size:
            print(f"Removed {initial_size - final_size} rows with missing values")
        
        # Sentiment distribution
        print(f"\nSentiment Distribution:")
        sentiment_counts = self.df['Sentiment'].value_counts()
        print(sentiment_counts)
        print(f"\nSentiment Percentages:")
        sentiment_percentages = (sentiment_counts / len(self.df) * 100).round(2)
        print(sentiment_percentages)
        
        # Text length statistics
        self.df['text_length'] = self.df['Comment'].str.len()
        print(f"\nText Length Statistics:")
        print(self.df['text_length'].describe())
        
        # Sample comments from each sentiment
        print(f"\nSample Comments:")
        for sentiment in self.df['Sentiment'].unique():
            print(f"\n{sentiment.upper()} Example:")
            sample = self.df[self.df['Sentiment'] == sentiment]['Comment'].iloc[0]
            print(f"'{sample[:150]}{'...' if len(sample) > 150 else ''}'")
        
        return self.df
    
    def visualize_data_distribution(self):
        """Create visualizations of the dataset."""
        print("\nCreating data visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YouTube Comments Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment distribution pie chart
        sentiment_counts = self.df['Sentiment'].value_counts()
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # red, yellow, green
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                       autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Sentiment Distribution')
        
        # 2. Sentiment distribution bar chart
        sentiment_counts.plot(kind='bar', ax=axes[0, 1], color=colors)
        axes[0, 1].set_title('Sentiment Counts')
        axes[0, 1].set_ylabel('Number of Comments')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Text length distribution by sentiment
        for sentiment in self.df['Sentiment'].unique():
            data = self.df[self.df['Sentiment'] == sentiment]['text_length']
            axes[1, 0].hist(data, alpha=0.7, label=sentiment, bins=50)
        axes[1, 0].set_title('Text Length Distribution by Sentiment')
        axes[1, 0].set_xlabel('Text Length (characters)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].set_xlim(0, 1000)  # Limit x-axis for better visualization
        
        # 4. Box plot of text lengths by sentiment
        sentiment_data = [self.df[self.df['Sentiment'] == sentiment]['text_length'] 
                         for sentiment in self.df['Sentiment'].unique()]
        axes[1, 1].boxplot(sentiment_data, labels=self.df['Sentiment'].unique())
        axes[1, 1].set_title('Text Length Distribution (Box Plot)')
        axes[1, 1].set_ylabel('Text Length (characters)')
        
        plt.tight_layout()
        plt.savefig('youtube_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Dataset analysis plots saved as 'youtube_dataset_analysis.png'")
    
    def prepare_data(self, test_size: float = 0.2, balance_data: bool = False):
        """Prepare the data for training."""
        print(f"\nPreparing data for training...")
        
        # Balance the dataset if requested
        if balance_data:
            print("Balancing dataset...")
            min_count = self.df['Sentiment'].value_counts().min()
            balanced_dfs = []
            for sentiment in self.df['Sentiment'].unique():
                sentiment_df = self.df[self.df['Sentiment'] == sentiment].sample(n=min_count, random_state=42)
                balanced_dfs.append(sentiment_df)
            self.df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
            print(f"Balanced dataset size: {len(self.df)}")
            print("New distribution:")
            print(self.df['Sentiment'].value_counts())
        
        # Prepare features and labels
        X = self.df['Comment'].tolist()
        y = self.df['Sentiment'].tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Training distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Testing distribution: {pd.Series(y_test).value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def create_models(self):
        """Create different model configurations optimized for the dataset."""
        models = {
            'naive_bayes_tfidf': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), 
                                        stop_words='english', min_df=2, max_df=0.95)),
                ('classifier', MultinomialNB(alpha=0.1))
            ]),
            
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), 
                                        stop_words='english', min_df=2, max_df=0.95)),
                ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
            ]),
            
            'svm_linear': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2), 
                                        stop_words='english', min_df=2, max_df=0.95)),
                ('classifier', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
            ]),
            
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                        stop_words='english', min_df=2, max_df=0.95)),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=20, 
                                                    random_state=42, n_jobs=-1))
            ]),
            
            'naive_bayes_count': Pipeline([
                ('count', CountVectorizer(max_features=10000, ngram_range=(1, 2), 
                                        stop_words='english', min_df=2, max_df=0.95)),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
        }
        
        return models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models."""
        print("\nTraining and evaluating models...")
        print("=" * 50)
        
        models = self.create_models()
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = datetime.now()
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            
            # Training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✓ {name}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Training Time: {training_time:.2f}s")
        
        self.models = models
        self.results = results
        
        # Find best model
        self.best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Best Accuracy: {results[self.best_model_name]['accuracy']:.4f}")
        
        return results
    
    def create_ensemble_model(self, X_train, y_train):
        """Create an ensemble model from the best performing individual models."""
        print("\nCreating ensemble model...")
        
        # Select top 3 models based on accuracy
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top_3_models = sorted_models[:3]
        
        print("Top 3 models for ensemble:")
        for name, result in top_3_models:
            print(f"  {name}: {result['accuracy']:.4f}")
        
        # Create ensemble
        estimators = [(name, result['model']) for name, result in top_3_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        print("Training ensemble model...")
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def save_best_model(self, filename: str = None):
        """Save the best model using SentimentAnalyzer format."""
        if not self.best_model:
            print("No trained model to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_sentiment_model_{timestamp}.pkl"
        
        # Create SentimentAnalyzer instance
        analyzer = SentimentAnalyzer()
        
        # Extract the trained components
        if hasattr(self.best_model.named_steps, 'tfidf'):
            analyzer.vectorizer = self.best_model.named_steps['tfidf']
        elif hasattr(self.best_model.named_steps, 'count'):
            analyzer.vectorizer = self.best_model.named_steps['count']
        
        analyzer.model = self.best_model.named_steps['classifier']
        analyzer.is_trained = True
        
        # Save the model
        analyzer.save_model(filename)
        
        print(f"✓ Best model ({self.best_model_name}) saved as: {filename}")
        print(f"  Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        print(f"  F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}")
        
        return filename
    
    def generate_detailed_report(self):
        """Generate a comprehensive training report."""
        if not self.results:
            return "No results available."
        
        report = []
        report.append("=" * 70)
        report.append("YOUTUBE COMMENTS SENTIMENT MODEL TRAINING REPORT")
        report.append("=" * 70)
        report.append(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset: {self.csv_file}")
        report.append(f"Total Samples: {len(self.df)}")
        report.append("")
        
        # Dataset statistics
        report.append("DATASET STATISTICS:")
        sentiment_dist = self.df['Sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(self.df)) * 100
            report.append(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Model performance comparison
        report.append("MODEL PERFORMANCE COMPARISON:")
        report.append("-" * 70)
        report.append(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'CV Mean':<10} {'Time(s)':<8}")
        report.append("-" * 70)
        
        for name, result in sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            report.append(f"{name:<25} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
                         f"{result['cv_mean']:<10.4f} {result['training_time']:<8.2f}")
        
        report.append("")
        report.append(f"🏆 BEST MODEL: {self.best_model_name}")
        report.append(f"   Final Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        report.append(f"   F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}")
        report.append("")
        
        # Detailed classification report for best model
        report.append("DETAILED CLASSIFICATION REPORT (Best Model):")
        report.append("-" * 50)
        report.append(self.results[self.best_model_name]['classification_report'])
        
        return "\n".join(report)


def main():
    """Main training workflow."""
    print("🚀 YouTube Comments Sentiment Analysis Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = YouTubeDatasetTrainer()
    
    # Load and explore data
    df = trainer.load_and_explore_data()
    
    # Create visualizations
    create_viz = input("\nCreate data visualizations? (y/n): ").strip().lower()
    if create_viz in ['y', 'yes']:
        trainer.visualize_data_distribution()
    
    # Ask about data balancing
    balance = input("\nBalance the dataset (equal samples per sentiment)? (y/n): ").strip().lower()
    balance_data = balance in ['y', 'yes']
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(balance_data=balance_data)
    
    # Train models
    print("\n🤖 Starting model training...")
    results = trainer.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Create ensemble model
    create_ensemble = input("\nCreate ensemble model? (y/n): ").strip().lower()
    if create_ensemble in ['y', 'yes']:
        ensemble = trainer.create_ensemble_model(X_train, y_train)
        ensemble_accuracy = ensemble.score(X_test, y_test)
        print(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
    
    # Generate and display report
    print("\n📊 TRAINING RESULTS:")
    report = trainer.generate_detailed_report()
    print(report)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"training_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Report saved as: {report_file}")
    
    # Save best model
    save_model = input("\nSave the best model? (y/n): ").strip().lower()
    if save_model in ['y', 'yes']:
        model_file = trainer.save_best_model()
        print(f"\n✅ Training completed! Model ready for use.")
        print(f"   Load with: analyzer.load_model('{model_file}')")
    
    print("\n🎉 Training workflow completed successfully!")


if __name__ == "__main__":
    main()
