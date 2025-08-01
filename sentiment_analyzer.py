#!/usr/bin/env python3
"""
Sentiment Analysis Model from Scratch

This module implements a complete sentiment analysis system including:
- Data preprocessing
- Feature extraction (TF-IDF, N-grams)
- Multiple ML models (Naive Bayes, SVM, Logistic Regression)
- Model training and evaluation
- Prediction on new text

Author: AI Assistant
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')


class TextPreprocessor:
    """Text preprocessing utilities for sentiment analysis."""
    
    def __init__(self):
        """Initialize the preprocessor with NLTK components."""
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def _download_nltk_data(self):
        """Download required NLTK data."""
        nltk_downloads = [
            'punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'omw-1.4'
        ]
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except:
                pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^a-zA-Z0-9\s:;=\-\)\(\[\]]+', '', text)
        
        return text.strip()
    
    def tokenize_and_process(self, text: str, remove_stopwords: bool = True, 
                           use_stemming: bool = False, use_lemmatization: bool = True) -> List[str]:
        """Tokenize and process text."""
        # Clean text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming or lemmatization
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def preprocess_text(self, text: str, join_tokens: bool = True) -> str:
        """Complete text preprocessing pipeline."""
        tokens = self.tokenize_and_process(text)
        
        if join_tokens:
            return ' '.join(tokens)
        return tokens


class SentimentDataset:
    """Handle sentiment analysis datasets."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def create_sample_dataset(self, size: int = 1000) -> pd.DataFrame:
        """Create a sample dataset for demonstration."""
        # Sample positive comments
        positive_samples = [
            "This video is absolutely amazing! Love it!",
            "Great content, keep up the good work!",
            "Fantastic explanation, very helpful",
            "I really enjoyed watching this",
            "Excellent video quality and content",
            "This is so inspiring and motivational",
            "Perfect tutorial, exactly what I needed",
            "Amazing work, subscribed immediately",
            "This made my day, thank you so much",
            "Brilliant content, very well explained"
        ] * (size // 30)
        
        # Sample negative comments
        negative_samples = [
            "This video is terrible and boring",
            "Waste of time, didn't learn anything",
            "Poor quality content, very disappointing",
            "I hate this, completely useless",
            "Worst video I've ever seen",
            "This is so confusing and poorly made",
            "Terrible explanation, makes no sense",
            "Boring content, couldn't finish watching",
            "This is awful, don't recommend",
            "Very bad quality, not worth watching"
        ] * (size // 30)
        
        # Sample neutral comments
        neutral_samples = [
            "This is okay, nothing special",
            "Average content, could be better",
            "It's fine, not great but not bad",
            "Decent video, some good points",
            "This is alright, seen better though",
            "Okay explanation, could use improvement",
            "It's fine for what it is",
            "Average quality, nothing outstanding",
            "This is okay, not my favorite",
            "Decent content, room for improvement"
        ] * (size // 30)
        
        # Create DataFrame
        texts = positive_samples + negative_samples + neutral_samples
        labels = (['positive'] * len(positive_samples) + 
                 ['negative'] * len(negative_samples) + 
                 ['neutral'] * len(neutral_samples))
        
        # Shuffle the data
        data = list(zip(texts, labels))
        np.random.shuffle(data)
        texts, labels = zip(*data)
        
        df = pd.DataFrame({
            'text': texts[:size],
            'sentiment': labels[:size]
        })
        
        return df
    
    def load_youtube_comments(self, json_file: str) -> pd.DataFrame:
        """Load YouTube comments from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            comments = []
            for comment in data.get('comments', []):
                comments.append({
                    'text': comment['text'],
                    'author': comment['author'],
                    'like_count': comment['like_count'],
                    'reply_count': comment['reply_count']
                })
            
            return pd.DataFrame(comments)
        
        except Exception as e:
            print(f"Error loading YouTube comments: {e}")
            return pd.DataFrame()


class SentimentAnalyzer:
    """Complete sentiment analysis system."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.label_encoder = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.is_trained = False
        
    def prepare_features(self, texts: List[str], fit_vectorizer: bool = False) -> np.ndarray:
        """Prepare features using TF-IDF vectorization."""
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        
        if fit_vectorizer or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            features = self.vectorizer.fit_transform(processed_texts)
        else:
            features = self.vectorizer.transform(processed_texts)
        
        return features
    
    def train_model(self, df: pd.DataFrame, model_type: str = 'logistic') -> Dict[str, Any]:
        """Train the sentiment analysis model."""
        print("Preparing training data...")
        
        # Prepare features and labels
        X = self.prepare_features(df['text'].tolist(), fit_vectorizer=True)
        y = np.array([self.label_encoder[label] for label in df['sentiment']])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Choose model
        models = {
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', probability=True),
            'logistic': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        
        if model_type not in models:
            model_type = 'logistic'
        
        self.model = models[model_type]
        
        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        self.is_trained = True
        
        results = {
            'model_type': model_type,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=['negative', 'neutral', 'positive']),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"Training completed!")
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train_model() first.")

        # Prepare features
        features = self.prepare_features([text])

        # Get prediction and probabilities
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        # Get sentiment label - fix numpy string compatibility
        if hasattr(prediction, 'item'):
            prediction_key = prediction.item()
        else:
            prediction_key = prediction

        # Handle string predictions directly
        if isinstance(prediction_key, str):
            sentiment = prediction_key
        else:
            sentiment = self.label_decoder[prediction_key]
        
        # Create probability dictionary
        prob_dict = {
            'negative': probabilities[0],
            'neutral': probabilities[1],
            'positive': probabilities[2]
        }
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': max(probabilities),
            'probabilities': prob_dict
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts."""
        return [self.predict_sentiment(text) for text in texts]
    
    def save_model(self, filepath: str):
        """Save the trained model and vectorizer."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and vectorizer."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.label_decoder = model_data['label_decoder']
            self.is_trained = True
            
            print(f"Model loaded from {filepath}")
        
        except Exception as e:
            print(f"Error loading model: {e}")


def main():
    """Main function to demonstrate the sentiment analyzer."""
    print("Sentiment Analysis System - Training Demo")
    print("=" * 50)
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    dataset = SentimentDataset()
    
    # Create sample dataset
    print("Creating sample dataset...")
    df = dataset.create_sample_dataset(size=1000)
    print(f"Dataset created with {len(df)} samples")
    print(f"Label distribution:\n{df['sentiment'].value_counts()}")
    
    # Train model
    print("\nTraining sentiment analysis model...")
    results = analyzer.train_model(df, model_type='logistic')
    
    # Test predictions
    print("\nTesting predictions:")
    test_texts = [
        "This video is absolutely amazing!",
        "This is terrible and boring",
        "It's okay, nothing special",
        "I love this content so much!",
        "Worst video ever, complete waste of time"
    ]
    
    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"Text: '{text}'")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print()
    
    # Save model
    analyzer.save_model('sentiment_model.pkl')
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
