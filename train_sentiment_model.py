#!/usr/bin/env python3
"""
Sentiment Model Training Script

This script provides a complete workflow for training sentiment analysis models:
1. Data preparation and preprocessing
2. Feature engineering
3. Model training with multiple algorithms
4. Model evaluation and comparison
5. Hyperparameter tuning
6. Model saving and deployment

Usage:
    python train_sentiment_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_analyzer import SentimentAnalyzer, SentimentDataset, TextPreprocessor
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Advanced model training and evaluation system."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def prepare_data(self, data_source: str = 'sample', data_path: str = None, 
                    test_size: float = 0.2) -> tuple:
        """Prepare training and testing data."""
        print("Preparing data...")
        
        if data_source == 'sample':
            # Create sample dataset
            dataset = SentimentDataset()
            df = dataset.create_sample_dataset(size=3000)
            print(f"Created sample dataset with {len(df)} samples")
        
        elif data_source == 'file' and data_path:
            # Load from file
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            print(f"Loaded dataset with {len(df)} samples from {data_path}")
        
        else:
            raise ValueError("Invalid data source or missing data path")
        
        # Check required columns
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'sentiment' columns")
        
        # Display data distribution
        print("Sentiment distribution:")
        print(df['sentiment'].value_counts())
        
        # Prepare features and labels
        X = df['text'].tolist()
        y = df['sentiment'].tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_models(self) -> dict:
        """Create different model configurations."""
        models = {
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                        stop_words='english')),
                ('classifier', MultinomialNB())
            ]),
            
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                        stop_words='english')),
                ('classifier', LogisticRegression(max_iter=1000))
            ]),
            
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), 
                                        stop_words='english')),
                ('classifier', SVC(kernel='linear', probability=True))
            ]),
            
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), 
                                        stop_words='english')),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            
            'count_nb': Pipeline([
                ('count', CountVectorizer(max_features=5000, ngram_range=(1, 2), 
                                        stop_words='english')),
                ('classifier', MultinomialNB())
            ])
        }
        
        return models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test) -> dict:
        """Train and evaluate multiple models."""
        print("\nTraining and evaluating models...")
        
        models = self.create_models()
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = models
        self.results = results
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name: str = 'logistic_regression'):
        """Perform hyperparameter tuning for the best model."""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'logistic_regression':
            param_grid = {
                'tfidf__max_features': [3000, 5000, 8000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'classifier__C': [0.1, 1, 10]
            }
            base_model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('classifier', LogisticRegression(max_iter=1000))
            ])
        
        elif model_name == 'svm':
            param_grid = {
                'tfidf__max_features': [3000, 5000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__C': [0.1, 1, 10]
            }
            base_model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('classifier', SVC(kernel='linear', probability=True))
            ])
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def create_ensemble_model(self, X_train, y_train):
        """Create an ensemble model combining multiple algorithms."""
        print("\nCreating ensemble model...")
        
        # Individual models for ensemble
        nb_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        lr_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        svm_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', SVC(kernel='linear', probability=True))
        ])
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('nb', nb_model),
                ('lr', lr_model),
                ('svm', svm_model)
            ],
            voting='soft'  # Use probability-based voting
        )
        
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def visualize_results(self, y_test, save_path: str = "model_evaluation.png"):
        """Create visualizations for model evaluation."""
        if not self.results:
            print("No results to visualize.")
            return
        
        # Set up plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Model Accuracy Comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confusion Matrix for Best Model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        cm = self.results[best_model_name]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. Cross-validation scores
        cv_data = []
        for name in model_names:
            cv_data.extend([self.results[name]['cv_mean']] * 5)  # Simulate CV scores
        
        model_labels = []
        for name in model_names:
            model_labels.extend([name] * 5)
        
        cv_df = pd.DataFrame({'Model': model_labels, 'CV_Score': cv_data})
        sns.boxplot(data=cv_df, x='Model', y='CV_Score', ax=axes[1, 0])
        axes[1, 0].set_title('Cross-Validation Score Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Feature importance (for models that support it)
        if 'random_forest' in self.results:
            rf_model = self.results['random_forest']['model']
            feature_names = rf_model.named_steps['tfidf'].get_feature_names_out()
            importances = rf_model.named_steps['classifier'].feature_importances_
            
            # Get top 20 features
            top_indices = np.argsort(importances)[-20:]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            axes[1, 1].barh(range(len(top_features)), top_importances)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features)
            axes[1, 1].set_title('Top 20 Feature Importances (Random Forest)')
            axes[1, 1].set_xlabel('Importance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Evaluation plots saved to: {save_path}")
    
    def save_best_model(self, filepath: str = "best_sentiment_model.pkl"):
        """Save the best performing model."""
        if not self.results:
            print("No models trained yet.")
            return
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        
        # Create SentimentAnalyzer instance and save
        analyzer = SentimentAnalyzer()
        analyzer.model = best_model.named_steps['classifier']
        analyzer.vectorizer = best_model.named_steps.get('tfidf') or best_model.named_steps.get('count')
        analyzer.is_trained = True
        
        analyzer.save_model(filepath)
        
        print(f"Best model ({best_model_name}) saved to: {filepath}")
        print(f"Best model accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
        return best_model_name


def main():
    """Main training workflow."""
    print("Sentiment Analysis Model Training")
    print("=" * 40)
    
    trainer = ModelTrainer()
    
    # Prepare data
    print("1. Data Preparation")
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        data_source='sample',  # Change to 'file' and provide path for custom data
        test_size=0.2
    )
    
    # Train and evaluate models
    print("\n2. Model Training and Evaluation")
    results = trainer.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Display results summary
    print("\n3. Results Summary")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name:20} | Accuracy: {result['accuracy']:.4f} | CV: {result['cv_mean']:.4f}")
    
    # Hyperparameter tuning for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\n4. Hyperparameter Tuning for {best_model_name}")
    
    if best_model_name in ['logistic_regression', 'svm']:
        tuned_model = trainer.hyperparameter_tuning(X_train, y_train, best_model_name)
        if tuned_model:
            # Evaluate tuned model
            tuned_accuracy = tuned_model.score(X_test, y_test)
            print(f"Tuned model accuracy: {tuned_accuracy:.4f}")
    
    # Create ensemble model
    print("\n5. Ensemble Model")
    ensemble_model = trainer.create_ensemble_model(X_train, y_train)
    ensemble_accuracy = ensemble_model.score(X_test, y_test)
    print(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
    
    # Visualize results
    print("\n6. Generating Visualizations")
    trainer.visualize_results(y_test)
    
    # Save best model
    print("\n7. Saving Best Model")
    best_name = trainer.save_best_model()
    
    print(f"\nTraining completed! Best model: {best_name}")


if __name__ == "__main__":
    main()
