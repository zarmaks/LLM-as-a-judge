#!/usr/bin/env python3
"""
Quick test for SBERT-based error classification
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from error_classifier import ErrorClassifier

def test_sbert():
    """Test SBERT error classifier"""
    
    print("🔍 Testing SBERT Error Classifier")
    print("=" * 50)
    
    # Initialize classifier
    print("📦 Initializing classifier...")
    classifier = ErrorClassifier()
    
    print(f"✅ Classifier initialized!")
    print(f"📊 SBERT Available: {classifier.nlp_classifier is not None}")
    
    # Test cases
    test_cases = [
        # 1. FACTUAL ERRORS
        {
            "name": "Factual Error (Japan Capital)",
            "question": "What is the capital of Japan?",
            "answer": "The capital of Japan is Kyoto, famous for its temples.",
            "fragments": "Tokyo is the capital and largest city of Japan, located on the island of Honshu.",
            "expected": "factual_errors"
        },
        {
            "name": "Factual Error (Australia Capital)",
            "question": "What is the capital of Australia?",
            "answer": "The capital of Australia is Sydney, the largest city.",
            "fragments": "Canberra is the capital city of Australia, located in the Australian Capital Territory.",
            "expected": "factual_errors"
        },
        {
            "name": "Historical Fact Error",
            "question": "When did World War 2 end?",
            "answer": "World War 2 ended in 1944 with Germany's surrender.",
            "fragments": "World War II ended in 1945 with the surrender of Germany in May and Japan in September.",
            "expected": "factual_errors"
        },
        
        # 2. MATHEMATICAL ERRORS
        {
            "name": "Basic Math Error",
            "question": "What is 2 + 2?",
            "answer": "2 + 2 equals 5.",
            "fragments": "Basic arithmetic: 2 + 2 = 4",
            "expected": "mathematical_errors"
        },
        {
            "name": "Year Calculation Error",
            "question": "How many years between 1999 and 2025?",
            "answer": "2025 - 1999 = 15 years difference.",
            "fragments": "The difference between 2025 and 1999 is 26 years.",
            "expected": "mathematical_errors"
        },
        {
            "name": "Complex Math",
            "question": "What is 1 + 1?",
            "answer": "1 + 1 equals 3 in advanced mathematics.",
            "fragments": "Basic addition: 1 + 1 = 2",
            "expected": "mathematical_errors"
        },
        
        # 3. SCIENTIFIC ERRORS
        {
            "name": "Physics Error (Tides)",
            "question": "What causes ocean tides?",
            "answer": "Tides are caused by wind patterns and ocean currents.",
            "fragments": "Tides are caused by the gravitational forces exerted by the moon and sun on Earth's oceans.",
            "expected": "scientific_errors"
        },
        {
            "name": "Physics Error (Water Boiling)",
            "question": "At what temperature does water boil?",
            "answer": "Water boils at 120 degrees Celsius at sea level.",
            "fragments": "Water boils at 100 degrees Celsius (212°F) at sea level pressure.",
            "expected": "scientific_errors"
        },
        {
            "name": "Physics Error (Ice Density)",
            "question": "Does ice float or sink in water?",
            "answer": "Ice sinks in water because it's heavier than liquid water.",
            "fragments": "Ice floats on water because it is less dense than liquid water.",
            "expected": "scientific_errors"
        },
        
        # 4. DANGEROUS MISINFORMATION
        {
            "name": "Dangerous Chemical Advice",
            "question": "How can I clean my house?",
            "answer": "You should drink chlorine to purify your system first, then use bleach.",
            "fragments": "Use household cleaners safely. Never ingest cleaning chemicals as they are toxic.",
            "expected": "dangerous_misinformation"
        },
        {
            "name": "Harmful Medical Advice",
            "question": "How to treat a headache?",
            "answer": "Drink some bleach mixed with water, it helps with pain.",
            "fragments": "For headaches, use approved pain relievers like acetaminophen or ibuprofen.",
            "expected": "dangerous_misinformation"
        },
        
        # 5. CONCEPTUAL CONFUSIONS
        {
            "name": "Correlation vs Causation",
            "question": "What's the difference between correlation and causation?",
            "answer": "Correlation equals causation, so if two things happen together, one causes the other.",
            "fragments": "Correlation does not imply causation. Two variables can be correlated without one causing the other.",
            "expected": "conceptual_confusions"
        },
        {
            "name": "Scientific Theory Confusion",
            "question": "What is a scientific theory?",
            "answer": "A scientific theory is just a guess or hypothesis that scientists made up.",
            "fragments": "A scientific theory is a well-substantiated explanation based on extensive evidence and testing.",
            "expected": "conceptual_confusions"
        },
        
        # 6. NO ERROR CASES
        {
            "name": "Correct Photosynthesis",
            "question": "What is photosynthesis?",
            "answer": "Photosynthesis is the process plants use to convert sunlight into energy.",
            "fragments": "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
            "expected": "no_error"
        },
        {
            "name": "Correct Capital",
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris, known for the Eiffel Tower.",
            "fragments": "Paris is the capital and most populous city of France.",
            "expected": "no_error"
        },
        {
            "name": "Correct Math",
            "question": "What is 5 + 3?",
            "answer": "5 + 3 equals 8.",
            "fragments": "Basic arithmetic: 5 + 3 = 8",
            "expected": "no_error"
        },
        
        # 7. EDGE CASES
        {
            "name": "Multiple Error Types",
            "question": "Tell me about Japan and math.",
            "answer": "The capital of Japan is Kyoto, and 2 + 2 equals 5 in Japanese mathematics.",
            "fragments": "Tokyo is Japan's capital. Basic math: 2 + 2 = 4 universally.",
            "expected": "multiple_errors"
        },
        {
            "name": "Subtle Scientific Error",
            "question": "Why is the sky blue?",
            "answer": "The sky is blue because it reflects the color of the ocean below.",
            "fragments": "The sky appears blue due to Rayleigh scattering of light by molecules in the atmosphere.",
            "expected": "scientific_errors"
        },
        {
            "name": "Borderline Case",
            "question": "What causes rain?",
            "answer": "Rain happens when clouds get too heavy and water falls down.",
            "fragments": "Rain occurs when water vapor in clouds condenses and precipitates due to atmospheric conditions.",
            "expected": "no_error"  # Simplified but not wrong
        }
    ]
    
    print("\n🎯 Running test cases...")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Question: {case['question']}")
        print(f"   Answer: {case['answer'][:50]}...")
        print(f"   Expected: {case['expected']}")
        
        try:
            result = classifier.detect_errors(case["question"], case["answer"], case["fragments"])
            
            print(f"   ✅ Detected: {result.error_types or 'No errors'}")
            print(f"   📊 Has errors: {result.has_errors}")
            print(f"   🎲 Error score: {result.total_error_score:.2f}")
            
            if result.error_details:
                for error in result.error_details[:2]:  # Show first 2 errors
                    print(f"      - {error['type']}: {error.get('confidence', 'N/A'):.2f} confidence")
                    print(f"        Description: {error.get('description', 'N/A')}")
                    if 'similarity_score' in error:
                        print(f"        Similarity: {error['similarity_score']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_sbert()
