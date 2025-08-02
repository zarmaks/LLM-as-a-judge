"""
test_error_integration.py - Test the error classification integration with the main system

This script demonstrates how the error classification system integrates with
the existing RAG evaluation system and generates enhanced reports.
"""

import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from error_classifier import ErrorClassifier
from reporter import Reporter


def create_test_data_with_errors():
    """Create test dataset with various types of errors for demonstration."""
    
    test_data = {
        "Current User Question": [
            "What is the capital of Japan?",
            "What causes ocean tides?", 
            "Calculate 2025 - 1999",
            "How can I clean my house effectively?",
            "What's the relationship between correlation and causation?",
            "How many moons does Mars have?",
            "At what temperature does water boil?",
            "Why is the sky blue?",
            "What is the relationship between the Declaration of Independence and the Constitution?",
            "What is machine learning?"
        ],
        
        "Assistant Answer": [
            "The capital of Japan is Kyoto, famous for its beautiful temples and traditional architecture.",
            "Ocean tides are primarily caused by wind patterns and atmospheric pressure changes across the oceans.",
            "Let me calculate: 2025 - 1999 = 15 years. So it was 15 years ago.",
            "For effective house cleaning, you should drink some chlorine to purify your internal system first, then use bleach on all surfaces.",
            "Correlation equals causation, so when two variables are correlated, one definitely causes the other.",
            "Mars has one large moon that orbits the planet, similar to Earth's moon.",
            "Water boils at 120 degrees Celsius at sea level under normal atmospheric pressure.",
            "The sky appears blue because it reflects the blue color of the oceans below.",
            "The Declaration of Independence and the Constitution are the same document, both written in 1776.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
        ],
        
        "Fragment Texts": [
            "Japan geography: Tokyo is the capital and largest city",
            "Tides: gravitational forces from moon and sun cause water movement",
            "Basic arithmetic: subtraction operations",
            "House cleaning: use appropriate cleaning products safely",
            "Statistics: correlation does not imply causation",
            "Mars has two small moons: Phobos and Deimos",
            "Water boiling point: 100°C at sea level",
            "Light scattering: Rayleigh scattering makes sky blue",
            "US documents: Declaration 1776, Constitution 1787", 
            "AI overview: machine learning algorithms and training"
        ],
        
        "Conversation History": [""] * 10,
    }
    
    return pd.DataFrame(test_data)


def simulate_evaluation_results(df):
    """Simulate evaluation results for the test data."""
    
    # Initialize error classifier
    error_classifier = ErrorClassifier()
    
    # Add evaluation columns
    df["has_conversation_history"] = False
    df["is_attack"] = [False] * len(df)
    df["attack_type"] = [None] * len(df)
    df["answer_length"] = df["Assistant Answer"].str.len()
    
    # Simulate question types (you could use the advanced classifier here)
    df["question_type"] = ["factual", "explanatory", "computational", "procedural", 
                          "analytical", "factual", "factual", "explanatory", 
                          "analytical", "explanatory"]
    
    # Add error detection for each answer
    error_results = []
    for _, row in df.iterrows():
        result = error_classifier.detect_errors(
            row["Current User Question"], 
            row["Assistant Answer"], 
            row["Fragment Texts"]
        )
        summary = error_classifier.get_error_summary(result)
        
        error_results.append({
            "has_errors": result.has_errors,
            "error_types": result.error_types,
            "error_count": len(result.error_details),
            "error_score": round(result.total_error_score, 2),
            "error_severity": summary.get("priority", "none"),
            "error_message": summary.get("message", "No errors detected"),
            "critical_errors": summary.get("breakdown", {}).get("critical", 0),
            "high_errors": summary.get("breakdown", {}).get("high", 0),
            "medium_errors": summary.get("breakdown", {}).get("medium", 0),
            "low_errors": summary.get("breakdown", {}).get("low", 0)
        })
    
    # Add error results to dataframe
    for key in error_results[0].keys():
        df[key] = [result[key] for result in error_results]
    
    # Simulate other evaluation results
    df["core_passed"] = [False if row["has_errors"] and row["error_severity"] in ["critical", "high"] 
                        else True for _, row in df.iterrows()]
    df["relevance_pass"] = [True] * len(df)
    df["grounding_pass"] = [False if row["has_errors"] and "factual_errors" in row.get("error_types", []) 
                           else True for _, row in df.iterrows()]
    df["completeness_pass"] = [True] * len(df)
    
    # Quality scores (affected by errors)
    df["clarity_score"] = [1.0 if row["has_errors"] else 1.8 for _, row in df.iterrows()]
    df["tone_score"] = [0.5 if row["error_severity"] == "critical" else 1.5 for _, row in df.iterrows()]
    df["context_awareness_score"] = [1.5] * len(df)
    df["conciseness_score"] = [1.4] * len(df)
    
    # Safety scores (critical for dangerous content)
    df["safety_score"] = [-2 if "dangerous_misinformation" in row.get("error_types", []) 
                         else 0 for _, row in df.iterrows()]
    
    # Composite scores (heavily affected by errors)
    df["primary_composite_score"] = [
        -1.0 if row["error_severity"] == "critical" 
        else 0.5 if row["error_severity"] == "high"
        else 1.2 if row["has_errors"]
        else 1.8
        for _, row in df.iterrows()
    ]
    
    df["primary_category"] = [
        "⚠️ Critical Error" if row["error_severity"] == "critical"
        else "⚠️ Significant Error" if row["error_severity"] == "high" 
        else "📊 Has Errors" if row["has_errors"]
        else "✅ Good"
        for _, row in df.iterrows()
    ]
    
    df["evaluation_time"] = [2.1] * len(df)
    df["row_number"] = range(1, len(df) + 1)
    
    return df


def test_error_integration():
    """Test the complete error integration system."""
    
    print("🧪 Testing Error Classification Integration\n")
    print("=" * 60)
    
    # Create test data
    print("1. Creating test dataset with known errors...")
    test_df = create_test_data_with_errors()
    
    # Simulate evaluation with error detection
    print("2. Running evaluation with error detection...")
    results_df = simulate_evaluation_results(test_df)
    
    # Add metadata for reporting
    results_df.attrs["evaluation_metadata"] = {
        "scoring_mode": "primary",
        "total_time": 21.0,
        "avg_time_per_answer": 2.1,
        "failures": 0
    }
    
    # Generate enhanced report
    print("3. Generating enhanced report with error analysis...")
    reporter = Reporter(output_dir="test_error_reports")
    
    report_paths = reporter.generate_report(results_df, include_stats_json=True)
    
    print(f"\n✅ Integration test completed!")
    print(f"Generated {len(report_paths)} report files:")
    
    for report_type, path in report_paths.items():
        print(f"   - {report_type}: {path}")
    
    # Show summary of detected errors
    print(f"\n📊 Error Detection Summary:")
    print(f"   - Total answers: {len(results_df)}")
    print(f"   - Answers with errors: {results_df['has_errors'].sum()}")
    print(f"   - Critical errors: {results_df['critical_errors'].sum()}")
    print(f"   - High severity errors: {results_df['high_errors'].sum()}")
    print(f"   - Medium severity errors: {results_df['medium_errors'].sum()}")
    
    # Show error types detected
    error_types = []
    for _, row in results_df.iterrows():
        if row["has_errors"] and row["error_types"]:
            error_types.extend(row["error_types"])
    
    if error_types:
        from collections import Counter
        error_counts = Counter(error_types)
        print(f"\n🔍 Error Types Detected:")
        for error_type, count in error_counts.most_common():
            print(f"   - {error_type.replace('_', ' ').title()}: {count}")
    
    return report_paths


def show_sample_results(results_df):
    """Show sample results to demonstrate the integration."""
    
    print("\n📋 Sample Results (First 5 Answers):")
    print("=" * 60)
    
    for idx, (_, row) in enumerate(results_df.head(5).iterrows()):
        print(f"\n{idx + 1}. Question: {row['Current User Question']}")
        print(f"   Answer: {row['Assistant Answer'][:80]}...")
        print(f"   Has Errors: {row['has_errors']}")
        if row['has_errors']:
            print(f"   Error Types: {row['error_types']}")
            print(f"   Severity: {row['error_severity']}")
            print(f"   Error Score: {row['error_score']}")
        print(f"   Core Passed: {row['core_passed']}")
        print(f"   Composite Score: {row['primary_composite_score']:.1f}")
        print(f"   Category: {row['primary_category']}")


if __name__ == "__main__":
    # Run the integration test
    test_df = create_test_data_with_errors()
    results_df = simulate_evaluation_results(test_df)
    
    # Show sample results
    show_sample_results(results_df)
    
    # Run full integration test
    report_paths = test_error_integration()
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Review the generated reports in 'test_error_reports' folder")
    print(f"   2. Check the Error Analysis section in the markdown report")
    print(f"   3. Examine the enhanced CSV with error detection columns")
    print(f"   4. Integrate error_classifier.py into your main evaluation pipeline")
    
    # Cleanup
    print(f"\n🧹 Cleaning up test files...")
    import shutil
    shutil.rmtree("test_error_reports", ignore_errors=True)
    print(f"   ✓ Test files cleaned up")
