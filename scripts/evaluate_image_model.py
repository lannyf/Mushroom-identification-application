"""
Evaluation script for image recognition model.

Evaluates model performance on test set and generates metrics.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config.image_model_config import ARTIFACTS_DIR
from models.image_recognition import ImageRecognitionModel, Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: Path,
                  test_images: List[np.ndarray],
                  test_labels: List[int],
                  species_ids: List[str],
                  batch_size: int = 32) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to saved model
        test_images: List of test images
        test_labels: List of test labels
        species_ids: List of species IDs
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("="*80)
    logger.info("Model Evaluation")
    logger.info("="*80)
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model = ImageRecognitionModel()
    model.load(model_path)
    
    # Create predictor
    predictor = Predictor(model, species_ids)
    
    # Make predictions
    logger.info("Making predictions on test set...")
    predictions = []
    confidences = []
    
    for image in test_images:
        preds = predictor.predict(image, top_k=1)
        if preds:
            pred_class = preds[0]['class_index']
            confidence = preds[0]['confidence']
            predictions.append(pred_class)
            confidences.append(confidence)
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    test_labels = np.array(test_labels)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = {
        'accuracy': float(accuracy_score(test_labels, predictions)),
        'precision_macro': float(precision_score(test_labels, predictions, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(test_labels, predictions, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)),
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences)),
    }
    
    # Per-class metrics
    logger.info("Computing per-class metrics...")
    precision_per_class = precision_score(test_labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(test_labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(test_labels, predictions, average=None, zero_division=0)
    
    per_class_metrics = {}
    for i, species_id in enumerate(species_ids):
        if i < len(precision_per_class):
            per_class_metrics[species_id] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
            }
    
    # Confusion matrix
    logger.info("Computing confusion matrix...")
    cm = confusion_matrix(test_labels, predictions)
    
    # Classification report
    logger.info("Classification Report:")
    report = classification_report(
        test_labels, predictions,
        target_names=species_ids[:len(np.unique(test_labels))],
        zero_division=0
    )
    logger.info(report)
    
    # Results
    results = {
        'test_samples': len(test_images),
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }
    
    # Save results
    logger.info("Saving evaluation results...")
    results_path = ARTIFACTS_DIR / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Print summary
    logger.info("="*80)
    logger.info("Evaluation Summary")
    logger.info("="*80)
    logger.info(f"Accuracy:        {metrics['accuracy']:.4f}")
    logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    logger.info(f"F1-Score (macro):  {metrics['f1_macro']:.4f}")
    logger.info(f"Mean Confidence:   {metrics['mean_confidence']:.4f}")
    logger.info("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate image recognition model')
    parser.add_argument('--model-path', type=Path,
                       default=ARTIFACTS_DIR / 'image_recognition_final.pt',
                       help='Path to saved model')
    parser.add_argument('--output-dir', type=Path, default=ARTIFACTS_DIR,
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # For now, use dummy data
    logger.warning("Using dummy test data - replace with real data when available")
    
    num_test = 20
    num_classes = 20
    test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                   for _ in range(num_test)]
    test_labels = np.random.randint(0, num_classes, num_test).tolist()
    species_ids = [f"SP{i:03d}" for i in range(num_classes)]
    
    # Evaluate
    results = evaluate_model(
        args.model_path,
        test_images,
        test_labels,
        species_ids
    )
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
