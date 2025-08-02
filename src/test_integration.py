"""
Quick test to verify error classification is working in reports
"""
import pandas as pd
from judge import RAGJudge
from reporter import RAGReporter

def test_full_integration():
    """Test full integration: Judge -> ErrorClassifier -> Reporter"""
    
    # Create simple test data
    test_data = pd.DataFrame([
        {
            "Current User Question": "What is the capital of Japan?",
            "Assistant Answer": "The capital of Japan is Kyoto, a beautiful historic city.",
            "Fragment Texts": "Japan's capital city Tokyo is a major metropolitan area.",
            "Conversation History": ""
        },
        {
            "Current User Question": "What is 2 + 2?",
            "Assistant Answer": "2 + 2 equals 4.",
            "Fragment Texts": "Basic arithmetic: two plus two equals four.",
            "Conversation History": ""
        }
    ])
    
    # Save test CSV
    test_csv = "test_integration.csv"
    test_data.to_csv(test_csv, index=False)
    
    try:
        print("🧪 Testing full integration: Judge -> ErrorClassifier -> Reporter")
        
        # Initialize judge
        judge = RAGJudge(scoring_mode="primary")
        
        # Evaluate dataset (should detect error in first answer)
        print("\n1. Running evaluation...")
        results = judge.evaluate_dataset(test_csv)
        
        print(f"✅ Evaluated {len(results)} answers")
        
        # Check error detection results
        print("\n2. Error detection results:")
        for idx, row in results.iterrows():
            question = row['Current User Question'][:50]
            has_errors = row.get('has_errors', False)
            error_types = row.get('error_types', [])
            error_score = row.get('error_score', 0)
            print(f"   Q: {question}...")
            print(f"   Errors: {has_errors}, Types: {error_types}, Score: {error_score}")
        
        # Generate report
        print("\n3. Generating report...")
        reporter = RAGReporter()
        report_path = reporter.generate_report(results, "test_integration_report")
        
        print(f"✅ Report generated: {report_path}")
        print("\n4. Checking if error analysis is included in report...")
        
        # Check if report contains error analysis
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
            
        if "Error Analysis" in report_content:
            print("✅ Error Analysis section found in report")
        else:
            print("❌ Error Analysis section NOT found in report")
            
        if "Error Type Analysis" in report_content:
            print("✅ Error Type Analysis found in report")
        else:
            print("❌ Error Type Analysis NOT found in report")
            
        if "factual_errors" in report_content or "Factual Errors" in report_content:
            print("✅ Specific error types found in report")
        else:
            print("❌ Specific error types NOT found in report")
        
    finally:
        # Cleanup
        import os
        if os.path.exists(test_csv):
            os.remove(test_csv)

if __name__ == "__main__":
    test_full_integration()
