#!/usr/bin/env python3
"""Test script for BART-enhanced error classifier."""

import sys
sys.path.append('.')

try:
    from src.error_classifier import ErrorClassifier
    print('✅ Import successful')
    
    classifier = ErrorClassifier()
    print('✅ Error Classifier initialized successfully')
    print(f'BART Available: {classifier.nlp_classifier is not None}')
    
    # Test cases
    test_cases = [
        {
            "question": "What is 2+2?",
            "answer": "2+2 equals 5. This is basic math.",
            "fragments": "According to mathematics, 2+2=4",
            "expected": "mathematical_errors"
        },
        {
            "question": "What is the capital of Japan?", 
            "answer": "The capital of Japan is Kyoto.",
            "fragments": "Tokyo is the capital and largest city of Japan.",
            "expected": "factual_errors"
        },
        {
            "question": "How do plants make food?",
            "answer": "Plants use photosynthesis to convert sunlight into energy using chlorophyll.",
            "fragments": "Photosynthesis is the process by which plants use sunlight to produce glucose.",
            "expected": "no_errors"
        }
    ]
    
    print("\n🔍 Testing Error Classification:")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Question: {case['question']}")
        print(f"   Answer: {case['answer']}")
        print(f"   Expected: {case['expected']}")
        
        try:
            result = classifier.detect_errors(
                case["question"], 
                case["answer"], 
                case["fragments"]
            )
            
            print(f"   ✅ Detected: {result.error_types if result.has_errors else 'no_errors'}")
            print(f"   Confidence: {max(result.confidence_scores.values()) if result.confidence_scores else 0:.2f}")
            
            if result.error_details:
                for detail in result.error_details[:1]:  # Show first error
                    print(f"   Details: {detail.get('description', 'N/A')}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Classification method: {'BART NLP' if classifier.nlp_classifier else 'Pattern-based'}")
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
