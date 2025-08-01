#!/usr/bin/env python3
"""
Enhanced Sentiment Analyzer with Advanced Features
Designed to achieve 80%+ accuracy through:
1. Advanced feature engineering
2. Ensemble methods
3. Deep learning integration
4. Better preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import re
import pickle
from typing import List, Dict, Any
from sentiment_analyzer import TextPreprocessor
import warnings
warnings.filterwarnings('ignore')


class AdvancedTextFeatures(BaseEstimator, TransformerMixin):
    """Extract advanced text features for better sentiment analysis."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract advanced features from text."""
        features = []
        
        for text in X:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            # Basic text statistics
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Punctuation features
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_count = sum(1 for c in text if c.isupper())
            caps_ratio = caps_count / max(char_count, 1)
            
            # Sentiment indicators
            positive_words = ['good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'like', 'best', 'perfect', 'wonderful']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'stupid', 'useless', 'annoying']
            
            positive_count = sum(1 for word in positive_words if word in text.lower())
            negative_count = sum(1 for word in negative_words if word in text.lower())
            
            # Emoji and special characters
            emoji_count = len(re.findall(r'[😀-🙏]', text))
            
            # Average word length
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            feature_vector = [
                char_count, word_count, sentence_count,
                exclamation_count, question_count, caps_ratio,
                positive_count, negative_count, emoji_count,
                avg_word_length
            ]
            
            features.append(feature_vector)
        
        return np.array(features)


class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer with advanced features and ensemble methods."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = AdvancedTextFeatures()
        self.models = {}
        self.ensemble_model = None
        self.label_encoder = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.is_trained = False
        
    def create_feature_pipeline(self):
        """Create advanced feature extraction pipeline."""
        # TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True
        )
        
        # Character-level TF-IDF
        char_tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=5000,
            min_df=2
        )
        
        # Combine all features
        feature_pipeline = FeatureUnion([
            ('tfidf', tfidf),
            ('char_tfidf', char_tfidf),
            ('advanced_features', self.feature_extractor)
        ])
        
        return feature_pipeline
    
    def create_ensemble_models(self):
        """Create ensemble of different models."""
        models = {
            'logistic': LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='linear',
                probability=True,
                C=1.0,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=0.1)
        }
        
        return models
    
    def train_enhanced_model(self, df: pd.DataFrame, test_size: float = 0.2):
        """Train enhanced model with advanced features."""
        print("🚀 Training Enhanced Sentiment Model...")
        print("=" * 50)
        
        # Prepare data
        texts = [self.preprocessor.preprocess_text(text) for text in df['text']]
        labels = [self.label_encoder[label] for label in df['sentiment']]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create feature pipeline
        feature_pipeline = self.create_feature_pipeline()
        
        # Fit feature pipeline
        print("Extracting advanced features...")
        X_train_features = feature_pipeline.fit_transform(X_train)
        X_test_features = feature_pipeline.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_features.shape}")
        
        # Train individual models
        models = self.create_ensemble_models()
        results = {}
        
        print("\n🤖 Training Individual Models:")
        print("-" * 40)
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_features, y_train)
            
            # Evaluate
            train_score = model.score(X_train_features, y_train)
            test_score = model.score(X_test_features, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_features, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  ✓ {name}: Test={test_score:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # Create ensemble model
        print("\n🎯 Creating Ensemble Model...")
        ensemble_estimators = [
            ('logistic', results['logistic']['model']),
            ('svm', results['svm']['model']),
            ('rf', results['random_forest']['model']),
            ('gb', results['gradient_boost']['model'])
        ]
        
        ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'  # Use probability voting
        )
        
        ensemble_model.fit(X_train_features, y_train)
        ensemble_score = ensemble_model.score(X_test_features, y_test)
        
        print(f"🏆 Ensemble Accuracy: {ensemble_score:.4f}")
        
        # Store the best components
        self.feature_pipeline = feature_pipeline
        self.ensemble_model = ensemble_model
        self.models = results
        self.is_trained = True
        
        # Detailed evaluation
        y_pred = ensemble_model.predict(X_test_features)
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))
        
        return {
            'ensemble_accuracy': ensemble_score,
            'individual_results': results,
            'feature_count': X_train_features.shape[1]
        }
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment using enhanced model."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Extract features
        features = self.feature_pipeline.transform([processed_text])
        
        # Get prediction and probabilities
        prediction = self.ensemble_model.predict(features)[0]
        probabilities = self.ensemble_model.predict_proba(features)[0]
        
        # Get sentiment label
        sentiment = self.label_decoder[prediction]
        
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
    
    def save_enhanced_model(self, filepath: str):
        """Save the enhanced model."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'feature_pipeline': self.feature_pipeline,
            'ensemble_model': self.ensemble_model,
            'models': self.models,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Enhanced model saved to {filepath}")
    
    def load_enhanced_model(self, filepath: str):
        """Load the enhanced model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.feature_pipeline = model_data['feature_pipeline']
            self.ensemble_model = model_data['ensemble_model']
            self.models = model_data['models']
            self.label_encoder = model_data['label_encoder']
            self.label_decoder = model_data['label_decoder']
            self.is_trained = True
            
            print(f"Enhanced model loaded from {filepath}")
        
        except Exception as e:
            print(f"Error loading enhanced model: {e}")


def main():
    """Demonstrate enhanced sentiment analysis."""
    print("🚀 Enhanced Sentiment Analysis Training")
    print("=" * 50)
    
    # Load data
    print("Loading YouTube dataset...")
    df = pd.read_csv("YoutubeCommentsDataSet.csv")
    
    # Clean data
    df = df.dropna()
    df.columns = ['text', 'sentiment']
    df['sentiment'] = df['sentiment'].str.lower()
    
    # Balance dataset
    min_samples = df['sentiment'].value_counts().min()
    balanced_df = df.groupby('sentiment').sample(n=min_samples, random_state=42)
    
    print(f"Balanced dataset size: {len(balanced_df)}")
    print(f"Distribution: {balanced_df['sentiment'].value_counts()}")
    
    # Train enhanced model
    analyzer = EnhancedSentimentAnalyzer()
    results = analyzer.train_enhanced_model(balanced_df)
    
    # Save model
    analyzer.save_enhanced_model('enhanced_sentiment_model.pkl')
    
    # Test predictions
    print("\n🎯 Testing Enhanced Model:")
    test_texts = [
        "This video is absolutely amazing! I love it so much!",
        "This is the worst content I've ever seen, terrible!",
        "It's okay, nothing special really"
    ]
    
    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"Text: \"{text}\"")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print()


if __name__ == "__main__":
    main()
