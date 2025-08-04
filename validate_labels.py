#!/usr/bin/env python3
"""
Label Validation Script
Compares the output of our RAG Judge system with ground truth labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_results(results_path):
    """Load the results from our RAG Judge evaluation"""
    df = pd.read_csv(results_path)
    return df

def load_ground_truth(labels_path):
    """Load the ground truth labels"""
    df = pd.read_csv(labels_path)
    return df

def map_our_categories_to_labels(row):
    """
    Map our system's output to the expected label categories
    """
    # Our system categories:
    # - core_passed: True/False
    # - safety_score: -1 (dangerous), 0 (neutral), 1 (safe)
    # - primary_category: various categories
    
    # If safety score is -1, it's DANGEROUS
    if row['safety_score'] == -1:
        return 'DANGEROUS'
    
    # If core criteria passed and safety is ok, it's CORRECT
    if row['core_passed'] and row['safety_score'] >= 0:
        return 'CORRECT'
    
    # Otherwise it's INCORRECT
    return 'INCORRECT'

def calculate_metrics(predicted, actual):
    """Calculate accuracy and other metrics"""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(actual, predicted)
    report = classification_report(actual, predicted, zero_division=0)
    cm = confusion_matrix(actual, predicted, labels=['CORRECT', 'INCORRECT', 'DANGEROUS'])
    
    return accuracy, report, cm

def main():
    parser = argparse.ArgumentParser(description='Validate RAG Judge results against ground truth')
    parser.add_argument('--results', required=True, help='Path to results CSV file')
    parser.add_argument('--labels', required=True, help='Path to ground truth labels CSV file')
    
    args = parser.parse_args()
    
    print("🔍 RAG Judge Label Validation")
    print("=" * 50)
    
    # Load data
    print(f"📂 Loading results: {args.results}")
    results_df = load_results(args.results)
    
    print(f"📂 Loading ground truth: {args.labels}")
    gt_df = load_ground_truth(args.labels)
    
    # Map our categories to expected labels
    print("🔄 Mapping system output to label categories...")
    results_df['predicted_label'] = results_df.apply(map_our_categories_to_labels, axis=1)
    
    # Align the data (by row index)
    print("📊 Aligning data...")
    merged_df = pd.merge(
        results_df[['row_number', 'predicted_label']], 
        gt_df,
        left_on='row_number',
        right_on='RowID',
        how='inner'
    )
    
    print(f"✅ Successfully aligned {len(merged_df)} rows")
    
    # Calculate metrics
    predicted = merged_df['predicted_label'].values
    actual = merged_df['Label'].values
    
    accuracy, report, cm = calculate_metrics(predicted, actual)
    
    # Display results
    print("\n📈 VALIDATION RESULTS")
    print("=" * 50)
    print(f"🎯 Overall Accuracy: {accuracy:.2%}")
    print()
    
    print("📊 Detailed Classification Report:")
    print(report)
    print()
    
    print("🔢 Confusion Matrix:")
    print("                Predicted")
    print("Actual      CORRECT  INCORRECT  DANGEROUS")
    labels = ['CORRECT', 'INCORRECT', 'DANGEROUS']
    for i, label in enumerate(labels):
        row_str = f"{label:8}"
        for j in range(len(labels)):
            row_str += f"{cm[i][j]:9d}"
        print(row_str)
    print()
    
    # Detailed analysis
    print("🔍 DETAILED ANALYSIS")
    print("=" * 50)
    
    comparison_df = merged_df[['row_number', 'predicted_label', 'Label']].copy()
    comparison_df['match'] = comparison_df['predicted_label'] == comparison_df['Label']
    
    print("Row-by-row comparison:")
    print(comparison_df.to_string(index=False))
    print()
    
    # Error analysis
    errors = comparison_df[~comparison_df['match']]
    if len(errors) > 0:
        print(f"❌ MISCLASSIFICATIONS ({len(errors)} total):")
        for _, row in errors.iterrows():
            print(f"   Row {row['row_number']}: Predicted {row['predicted_label']}, Actual {row['Label']}")
    else:
        print("✅ No misclassifications!")
    
    print()
    
    # Category-specific metrics
    print("📊 CATEGORY-SPECIFIC PERFORMANCE")
    print("=" * 50)
    
    for label in ['CORRECT', 'INCORRECT', 'DANGEROUS']:
        label_mask = actual == label
        if label_mask.sum() > 0:
            label_accuracy = (predicted[label_mask] == actual[label_mask]).mean()
            print(f"{label:10}: {label_accuracy:.2%} ({label_mask.sum()} samples)")
    
    print()
    print("✅ Validation complete!")

if __name__ == "__main__":
    main()
