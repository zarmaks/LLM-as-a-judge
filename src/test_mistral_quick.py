"""
Quick test of Mistral error classification
"""

from error_classifier_mistral import ErrorClassifier

def test_mistral_simple():
    """Test just the Mistral functionality"""
    classifier = ErrorClassifier()
    
    # Test with obvious error that should trigger Mistral (not just patterns)
    question = "What is the capital of France?"
    answer = "The capital of France is London."  # Clear factual error
    
    print("🧪 Testing Mistral-based error detection...")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    result = classifier.detect_errors(question, answer)
    summary = classifier.get_error_summary(result)
    
    print(f"\nResult:")
    print(f"- Has errors: {result.has_errors}")
    print(f"- Error types: {result.error_types}")
    print(f"- Error score: {result.total_error_score:.2f}")
    print(f"- Status: {summary.get('message', 'Unknown')}")
    
    if result.error_details:
        for error in result.error_details:
            print(f"- Method: {error.get('method', 'unknown')}")
            print(f"- Description: {error.get('description', 'N/A')}")

if __name__ == "__main__":
    test_mistral_simple()
