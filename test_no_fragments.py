#!/usr/bin/env python3
"""
Test SBERT without source fragments - does it still detect errors?
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from error_classifier import ErrorClassifier

def test_no_fragments():
    """Test error detection WITHOUT source fragments"""
    
    print("🔍 Testing SBERT WITHOUT Source Fragments")
    print("=" * 60)
    
    classifier = ErrorClassifier()
    print(f"📊 SBERT Available: {classifier.nlp_classifier is not None}")
    
    # Test cases WITHOUT fragments 
    test_cases = [
        {
            "name": "Japan Capital Error (No Fragments)",
            "question": "What is the capital of Japan?",
            "answer": "The capital of Japan is Kyoto, famous for its temples.",
            "fragments": "",  # NO SOURCE FRAGMENTS!
            "expected": "Should detect? How?"
        },
        {
            "name": "Math Error (No Fragments)", 
            "question": "What is 2 + 2?",
            "answer": "2 + 2 equals 5.",
            "fragments": "",  # NO SOURCE FRAGMENTS!
            "expected": "Should detect via pattern?"
        },
        {
            "name": "Scientific Error (No Fragments)",
            "question": "What causes tides?",
            "answer": "Tides are caused by wind patterns.",
            "fragments": "",  # NO SOURCE FRAGMENTS!
            "expected": "Should detect?"
        },
        {
            "name": "Dangerous Advice (No Fragments)",
            "question": "How to clean?",
            "answer": "Drink chlorine to purify yourself.",
            "fragments": "",  # NO SOURCE FRAGMENTS!
            "expected": "Should detect via pattern"
        },
        {
            "name": "Correct Answer (No Fragments)",
            "question": "What is photosynthesis?", 
            "answer": "Photosynthesis converts sunlight to energy.",
            "fragments": "",  # NO SOURCE FRAGMENTS!
            "expected": "Should NOT detect errors"
        }
    ]
    
    print("\n🎯 Testing WITHOUT source fragments...")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Question: {case['question']}")
        print(f"   Answer: {case['answer']}")
        print(f"   Fragments: '{case['fragments']}' (EMPTY!)")
        print(f"   Expected: {case['expected']}")
        
        try:
            result = classifier.detect_errors(case["question"], case["answer"], case["fragments"])
            
            print(f"   🔍 Detected: {result.error_types or 'No errors'}")
            print(f"   📊 Has errors: {result.has_errors}")
            print(f"   🎲 Error score: {result.total_error_score:.2f}")
            
            # How was it detected?
            if result.error_details:
                for error in result.error_details:
                    detection_method = "UNKNOWN"
                    if 'pattern_matched' in error:
                        detection_method = "PATTERN-BASED"
                    elif 'similarity_score' in error:
                        detection_method = "SBERT-SEMANTIC"  
                    elif 'contradicted_fact' in error:
                        detection_method = "FACT-DATABASE"
                    
                    print(f"      - {error['type']}: {error.get('confidence', 'N/A'):.2f} confidence")
                    print(f"        Method: {detection_method}")
                    print(f"        Description: {error.get('description', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("🤔 CONCLUSION:")
    print("Without source fragments, SBERT can only detect errors via:")
    print("1. 📝 Pattern-based detection (regex patterns)")
    print("2. 🧠 Built-in fact database comparison") 
    print("3. ❌ NO semantic similarity comparison (needs fragments!)")
    print("="*60)

if __name__ == "__main__":
    test_no_fragments()
