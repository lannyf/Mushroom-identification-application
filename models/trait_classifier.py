"""
Trait-Based Mushroom Classification Models

Implements decision tree and random forest classifiers for identifying mushrooms
based on structured morphological traits (cap color, gill structure, stem shape, etc).

These models provide an identification method that works without images and can
be easily explained to users (rule-based reasoning).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import logging

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

logger = logging.getLogger(__name__)


class TraitClassifier:
    """
    Trains and manages trait-based classification models.
    
    Supports two algorithms:
    1. Decision Tree - Interpretable, rule-based decisions
    2. Random Forest - High accuracy, ensemble method
    
    Both models learn patterns from mushroom trait data and predict species.
    """
    
    ALGORITHMS = ['decision_tree', 'random_forest']
    
    def __init__(self, 
                 algorithm: str = 'random_forest',
                 n_species: int = 20,
                 random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            algorithm: One of ['decision_tree', 'random_forest']
            n_species: Number of species classes
            random_state: Random seed for reproducibility
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {self.ALGORITHMS}")
        
        self.algorithm = algorithm
        self.n_species = n_species
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        
        # Training history
        self.train_history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'feature_importances': {}
        }
        
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the underlying sklearn model."""
        if self.algorithm == 'decision_tree':
            self.model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                criterion='gini',
                random_state=self.random_state
            )
        
        elif self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                criterion='gini',
                n_jobs=-1,
                random_state=self.random_state
            )
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Train the classifier on trait data.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation labels
            feature_names: List of feature names for interpretation
            class_names: List of class/species names
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.algorithm} on {X_train.shape[0]} samples...")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Store metadata
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.class_names = class_names or [f"class_{i}" for i in range(self.n_species)]
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        metrics = {'train_accuracy': train_accuracy}
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            metrics['val_accuracy'] = val_accuracy
            logger.info(f"Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}")
        else:
            logger.info(f"Train accuracy: {train_accuracy:.4f}")
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.train_history['feature_importances'] = dict(
                zip(self.feature_names, importances)
            )
        
        self.train_history['train_accuracy'].append(train_accuracy)
        if 'val_accuracy' in metrics:
            self.train_history['val_accuracy'].append(metrics['val_accuracy'])
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict species labels for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Predicted labels of shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Probability matrix of shape (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_with_confidence(self, X: np.ndarray, top_k: int = 5
                               ) -> List[List[Tuple[str, float]]]:
        """
        Predict species with confidence scores.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            top_k: Number of top predictions to return per sample
        
        Returns:
            List of lists containing (species_name, confidence) tuples
        """
        proba = self.predict_proba(X)
        results = []
        
        for sample_proba in proba:
            # Get top-k indices and scores
            top_indices = np.argsort(sample_proba)[::-1][:top_k]
            
            predictions = [
                (self.class_names[idx], float(sample_proba[idx]))
                for idx in top_indices
            ]
            results.append(predictions)
        
        return results
    
    def get_feature_importance(self, top_n: int = 15) -> List[Tuple[str, float]]:
        """
        Get most important features for classification.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            List of (feature_name, importance) tuples
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Feature importances not available for this algorithm")
        
        importances = self.model.feature_importances_
        feature_importance_pairs = list(zip(self.feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance_pairs[:top_n]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray
                ) -> Dict[str, Any]:
        """
        Comprehensive evaluation on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Try to compute ROC AUC (for binary/multiclass)
        try:
            if len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = 0.0
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        per_class_report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': per_class_report
        }
    
    def save(self, model_path: str, scaler_path: str, metadata_path: str) -> None:
        """
        Save model, scaler, and metadata.
        
        Args:
            model_path: Path to save sklearn model
            scaler_path: Path to save feature scaler
            metadata_path: Path to save metadata (feature names, class names, etc)
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        metadata = {
            'algorithm': self.algorithm,
            'n_species': self.n_species,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'is_trained': self.is_trained,
            'train_history': self.train_history
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved: {model_path}, {scaler_path}, {metadata_path}")
    
    def load(self, model_path: str, scaler_path: str, metadata_path: str) -> None:
        """
        Load model, scaler, and metadata.
        
        Args:
            model_path: Path to saved sklearn model
            scaler_path: Path to saved feature scaler
            metadata_path: Path to saved metadata
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.algorithm = metadata['algorithm']
        self.n_species = metadata['n_species']
        self.feature_names = metadata['feature_names']
        self.class_names = metadata['class_names']
        self.is_trained = metadata['is_trained']
        self.train_history = metadata['train_history']
        
        logger.info(f"Model loaded: {model_path}")


class Predictor:
    """
    Inference interface for trait-based classification.
    
    Provides a clean API for making predictions on new observations.
    """
    
    def __init__(self, classifier: TraitClassifier):
        """
        Initialize predictor.
        
        Args:
            classifier: Trained TraitClassifier instance
        """
        self.classifier = classifier
    
    def predict_from_traits(self, traits: Dict[str, str], top_k: int = 5
                           ) -> List[Tuple[str, float]]:
        """
        Predict species from trait dictionary.
        
        Args:
            traits: {trait_id: value} dictionary
            top_k: Number of top predictions
        
        Returns:
            List of (species_name, confidence) tuples
        """
        # This requires a TraitEncoder - should be passed separately
        raise NotImplementedError("Use TraitObservation to encode traits first")
    
    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Provide explanation for a prediction.
        
        Args:
            X: Feature matrix
            sample_idx: Which sample to explain
        
        Returns:
            Dictionary with explanation details
        """
        predictions = self.classifier.predict_with_confidence(X[sample_idx:sample_idx+1])
        feature_importance = self.classifier.get_feature_importance(top_n=5)
        
        return {
            'predictions': predictions[0],
            'top_features': feature_importance
        }
