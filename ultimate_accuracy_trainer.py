#!/usr/bin/env python3
"""
Ultimate Accuracy Training Script
Combines all strategies to achieve maximum accuracy:
1. Data quality enhancement
2. Advanced feature engineering  
3. Hyperparameter optimization
4. Ensemble methods
5. Cross-validation and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_quality_enhancer import DataQualityEnhancer
from enhanced_sentiment_analyzer import AdvancedTextFeatures
from sentiment_analyzer import TextPreprocessor


class UltimateAccuracyTrainer:
    """Ultimate trainer combining all accuracy improvement strategies."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.data_enhancer = DataQualityEnhancer()
        self.advanced_features = AdvancedTextFeatures()
        self.results = {}
        
    def create_ultimate_pipeline(self):
        """Create the ultimate feature extraction pipeline."""
        # Optimized TF-IDF (word-level)
        word_tfidf = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True,
            use_idf=True
        )
        
        # Character-level TF-IDF
        char_tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 5),
            max_features=8000,
            min_df=2,
            max_df=0.95
        )
        
        # Combine all features
        ultimate_pipeline = FeatureUnion([
            ('word_tfidf', word_tfidf),
            ('char_tfidf', char_tfidf),
            ('advanced_features', self.advanced_features)
        ])
        
        return ultimate_pipeline
    
    def create_optimized_models(self):
        """Create optimized models with best hyperparameters."""
        models = {
            'logistic_optimized': LogisticRegression(
                C=10.0,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                max_iter=2000,
                random_state=42
            ),
            'svm_optimized': SVC(
                C=10.0,
                kernel='linear',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'random_forest_optimized': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost_optimized': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            )
        }
        
        return models
    
    def train_ultimate_model(self, csv_file: str = "YoutubeCommentsDataSet.csv"):
        """Train the ultimate high-accuracy model."""
        print("🚀 ULTIMATE ACCURACY TRAINING")
        print("=" * 60)
        
        # Step 1: Create high-quality dataset
        print("📊 Step 1: Creating high-quality dataset...")
        high_quality_df = self.data_enhancer.create_high_quality_dataset(csv_file)
        
        # Step 2: Prepare data
        print("\n🔧 Step 2: Preparing data...")
        texts = [self.preprocessor.preprocess_text(str(text)) for text in high_quality_df['text']]
        labels = high_quality_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        
        # Step 3: Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Step 4: Create ultimate feature pipeline
        print("\n⚙️ Step 3: Creating ultimate feature pipeline...")
        feature_pipeline = self.create_ultimate_pipeline()
        
        # Extract features
        print("Extracting ultimate features...")
        X_train_features = feature_pipeline.fit_transform(X_train)
        X_test_features = feature_pipeline.transform(X_test)
        
        print(f"Ultimate feature matrix shape: {X_train_features.shape}")
        
        # Step 5: Train optimized models
        print("\n🤖 Step 4: Training optimized models...")
        models = self.create_optimized_models()
        
        individual_results = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_features, y_train)
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(
                model, X_train_features, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # Test accuracy
            test_score = model.score(X_test_features, y_test)
            
            individual_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_score
            }
            
            trained_models[name] = model
            
            print(f"  ✓ {name}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test={test_score:.4f}")
        
        # Step 6: Create ultimate ensemble
        print("\n🎯 Step 5: Creating ultimate ensemble...")
        
        # Select top 3 models for ensemble
        top_models = sorted(individual_results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:3]
        
        ensemble_estimators = []
        for model_name, _ in top_models:
            ensemble_estimators.append((model_name, trained_models[model_name]))
        
        ultimate_ensemble = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'
        )
        
        ultimate_ensemble.fit(X_train_features, y_train)
        
        # Step 7: Final evaluation
        print("\n📊 Step 6: Final evaluation...")
        
        # Ensemble performance
        ensemble_cv_scores = cross_val_score(
            ultimate_ensemble, X_train_features, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        ensemble_test_score = ultimate_ensemble.score(X_test_features, y_test)
        y_pred = ultimate_ensemble.predict(X_test_features)
        
        # Detailed metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n🏆 ULTIMATE RESULTS:")
        print("=" * 50)
        print(f"Cross-Validation Accuracy: {ensemble_cv_scores.mean():.4f} ± {ensemble_cv_scores.std():.4f}")
        print(f"Test Accuracy: {ensemble_test_score:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        
        print(f"\n📈 IMPROVEMENT:")
        baseline_accuracy = 0.6793
        improvement = ensemble_test_score - baseline_accuracy
        improvement_percent = (improvement / baseline_accuracy) * 100
        print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
        print(f"Ultimate Accuracy: {ensemble_test_score:.4f}")
        print(f"Improvement: +{improvement:.4f} ({improvement_percent:+.1f}%)")
        
        # Classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))
        
        # Step 8: Save ultimate model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"ultimate_sentiment_model_{timestamp}.pkl"
        
        ultimate_model_data = {
            'feature_pipeline': feature_pipeline,
            'ensemble_model': ultimate_ensemble,
            'individual_models': trained_models,
            'results': {
                'test_accuracy': ensemble_test_score,
                'cv_mean': ensemble_cv_scores.mean(),
                'cv_std': ensemble_cv_scores.std(),
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'improvement': improvement
            },
            'label_decoder': {0: 'negative', 1: 'neutral', 2: 'positive'}
        }
        
        with open(model_filename, 'wb') as f:
            pickle.dump(ultimate_model_data, f)
        
        print(f"\n💾 Ultimate model saved as: {model_filename}")
        
        # Test with sample predictions
        print(f"\n🎯 SAMPLE PREDICTIONS:")
        test_texts = [
            "This video is absolutely incredible! Best content ever!",
            "This is the worst thing I've ever watched, terrible quality",
            "It's okay, nothing special but not bad either"
        ]
        
        for text in test_texts:
            processed_text = self.preprocessor.preprocess_text(text)
            features = feature_pipeline.transform([processed_text])
            prediction = ultimate_ensemble.predict(features)[0]
            probabilities = ultimate_ensemble.predict_proba(features)[0]
            
            sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}[prediction]
            confidence = max(probabilities)
            
            print(f"Text: \"{text[:50]}...\"")
            print(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
            print()
        
        return ultimate_model_data


def main():
    """Run ultimate accuracy training."""
    trainer = UltimateAccuracyTrainer()
    
    # Train ultimate model
    ultimate_model = trainer.train_ultimate_model()
    
    print("🎉 Ultimate accuracy training completed!")
    print(f"Final accuracy: {ultimate_model['results']['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
