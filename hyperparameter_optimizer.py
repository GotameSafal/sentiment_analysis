#!/usr/bin/env python3
"""
Hyperparameter Optimization for Maximum Accuracy
Advanced techniques to find optimal model parameters:
1. Grid Search with Cross-Validation
2. Random Search
3. Bayesian Optimization
4. Feature selection optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """Optimize hyperparameters for maximum accuracy."""
    
    def __init__(self):
        self.best_params = {}
        self.best_scores = {}
        
    def optimize_tfidf_parameters(self, X_train, y_train):
        """Optimize TF-IDF vectorizer parameters."""
        print("🔧 Optimizing TF-IDF parameters...")
        
        # TF-IDF parameter grid
        tfidf_params = {
            'tfidf__max_features': [5000, 10000, 15000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
            'tfidf__min_df': [1, 2, 3, 5],
            'tfidf__max_df': [0.8, 0.9, 0.95, 0.99],
            'tfidf__sublinear_tf': [True, False],
            'tfidf__use_idf': [True, False]
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        # Random search for efficiency
        random_search = RandomizedSearchCV(
            pipeline,
            tfidf_params,
            n_iter=50,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best TF-IDF score: {random_search.best_score_:.4f}")
        print(f"Best TF-IDF params: {random_search.best_params_}")
        
        self.best_params['tfidf'] = random_search.best_params_
        self.best_scores['tfidf'] = random_search.best_score_
        
        return random_search.best_estimator_
    
    def optimize_logistic_regression(self, X_train, y_train, tfidf_params=None):
        """Optimize Logistic Regression parameters."""
        print("🔧 Optimizing Logistic Regression...")
        
        # Use best TF-IDF params if available
        if tfidf_params:
            tfidf = TfidfVectorizer(**{k.replace('tfidf__', ''): v for k, v in tfidf_params.items()})
        else:
            tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        
        # Logistic Regression parameter grid
        lr_params = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__class_weight': [None, 'balanced'],
            'classifier__max_iter': [1000, 2000, 3000]
        }
        
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', LogisticRegression())
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            lr_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best Logistic Regression score: {grid_search.best_score_:.4f}")
        print(f"Best Logistic Regression params: {grid_search.best_params_}")
        
        self.best_params['logistic'] = grid_search.best_params_
        self.best_scores['logistic'] = grid_search.best_score_
        
        return grid_search.best_estimator_
    
    def optimize_svm(self, X_train, y_train, tfidf_params=None):
        """Optimize SVM parameters."""
        print("🔧 Optimizing SVM...")
        
        if tfidf_params:
            tfidf = TfidfVectorizer(**{k.replace('tfidf__', ''): v for k, v in tfidf_params.items()})
        else:
            tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        
        # SVM parameter grid
        svm_params = {
            'classifier__C': [0.1, 1.0, 10.0, 100.0],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'classifier__class_weight': [None, 'balanced']
        }
        
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', SVC(probability=True))
        ])
        
        # Use RandomizedSearchCV for SVM (faster)
        random_search = RandomizedSearchCV(
            pipeline,
            svm_params,
            n_iter=30,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best SVM score: {random_search.best_score_:.4f}")
        print(f"Best SVM params: {random_search.best_params_}")
        
        self.best_params['svm'] = random_search.best_params_
        self.best_scores['svm'] = random_search.best_score_
        
        return random_search.best_estimator_
    
    def optimize_random_forest(self, X_train, y_train, tfidf_params=None):
        """Optimize Random Forest parameters."""
        print("🔧 Optimizing Random Forest...")
        
        if tfidf_params:
            tfidf = TfidfVectorizer(**{k.replace('tfidf__', ''): v for k, v in tfidf_params.items()})
        else:
            tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        
        # Random Forest parameter grid
        rf_params = {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [10, 20, 30, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': [None, 'balanced'],
            'classifier__bootstrap': [True, False]
        }
        
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        random_search = RandomizedSearchCV(
            pipeline,
            rf_params,
            n_iter=50,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best Random Forest score: {random_search.best_score_:.4f}")
        print(f"Best Random Forest params: {random_search.best_params_}")
        
        self.best_params['random_forest'] = random_search.best_params_
        self.best_scores['random_forest'] = random_search.best_score_
        
        return random_search.best_estimator_
    
    def optimize_feature_selection(self, X_train, y_train, best_model):
        """Optimize feature selection."""
        print("🔧 Optimizing feature selection...")
        
        # Feature selection parameter grid
        feature_params = {
            'feature_selection__k': [1000, 2000, 5000, 10000, 15000, 'all']
        }
        
        # Create pipeline with feature selection
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')),
            ('feature_selection', SelectKBest(chi2)),
            ('classifier', best_model)
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            feature_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best feature selection score: {grid_search.best_score_:.4f}")
        print(f"Best feature selection params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def comprehensive_optimization(self, df: pd.DataFrame):
        """Run comprehensive hyperparameter optimization."""
        print("🚀 Comprehensive Hyperparameter Optimization")
        print("=" * 60)
        
        # Prepare data
        from sentiment_analyzer import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        texts = [preprocessor.preprocess_text(text) for text in df['text']]
        labels = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Step 1: Optimize TF-IDF
        best_tfidf_model = self.optimize_tfidf_parameters(X_train, y_train)
        tfidf_params = self.best_params['tfidf']
        
        # Step 2: Optimize individual models
        best_lr = self.optimize_logistic_regression(X_train, y_train, tfidf_params)
        best_svm = self.optimize_svm(X_train, y_train, tfidf_params)
        best_rf = self.optimize_random_forest(X_train, y_train, tfidf_params)
        
        # Step 3: Find the best overall model
        best_model_name = max(self.best_scores, key=self.best_scores.get)
        best_score = self.best_scores[best_model_name]
        
        print(f"\n🏆 OPTIMIZATION RESULTS:")
        print("=" * 40)
        for model_name, score in self.best_scores.items():
            print(f"{model_name:15}: {score:.4f}")
        
        print(f"\n🥇 Best Model: {best_model_name} (Score: {best_score:.4f})")
        
        # Test on holdout set
        if best_model_name == 'logistic':
            final_model = best_lr
        elif best_model_name == 'svm':
            final_model = best_svm
        elif best_model_name == 'random_forest':
            final_model = best_rf
        else:
            final_model = best_tfidf_model
        
        # Final evaluation
        y_pred = final_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Final Test Accuracy: {final_accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))
        
        return final_model, self.best_params, final_accuracy


def main():
    """Run hyperparameter optimization."""
    # Load data
    df = pd.read_csv("YoutubeCommentsDataSet.csv")
    df = df.dropna()
    df.columns = ['text', 'sentiment']
    df['sentiment'] = df['sentiment'].str.lower()
    
    # Balance dataset
    min_samples = df['sentiment'].value_counts().min()
    balanced_df = df.groupby('sentiment').sample(n=min_samples, random_state=42)
    
    print(f"Dataset size: {len(balanced_df)}")
    print(f"Distribution: {balanced_df['sentiment'].value_counts()}")
    
    # Run optimization
    optimizer = HyperparameterOptimizer()
    best_model, best_params, final_accuracy = optimizer.comprehensive_optimization(balanced_df)
    
    # Save results
    import pickle
    with open('optimized_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('best_hyperparameters.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    print(f"\n💾 Optimized model saved with accuracy: {final_accuracy:.4f}")


if __name__ == "__main__":
    main()
