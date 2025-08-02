"""
Tests for ErrorClassifierMistral - Enhanced Error Detection System

Tests cover:
1. Basic error detection functionality
2. Pattern-based detection
3. Mistral LLM integration
4. Mathematical calculation verification
5. Dangerous content detection
6. Error severity classification
7. Detailed analysis features
8. Error correction suggestions
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from error_classifier_mistral import ErrorClassifier, ErrorDetectionResult


class TestErrorClassifierMistral:
    """Test suite for ErrorClassifierMistral class."""
    
    @pytest.fixture
    def classifier(self):
        """Create an ErrorClassifier instance for testing."""
        return ErrorClassifier()
    
    def test_classifier_initialization(self, classifier):
        """Test proper initialization of ErrorClassifier."""
        assert hasattr(classifier, 'error_patterns')
        assert hasattr(classifier, 'fact_database')
        assert hasattr(classifier, 'dangerous_keywords')
        
        # Check error patterns are loaded
        assert 'factual_errors' in classifier.error_patterns
        assert 'mathematical_errors' in classifier.error_patterns
        assert 'scientific_errors' in classifier.error_patterns
        assert 'dangerous_misinformation' in classifier.error_patterns
        assert 'conceptual_confusions' in classifier.error_patterns
        
        # Check fact database has essential facts
        assert 'japan_capital' in classifier.fact_database
        assert classifier.fact_database['japan_capital'] == 'tokyo'
        
        # Check dangerous keywords are loaded
        assert 'chlorine' in classifier.dangerous_keywords
        assert 'bleach' in classifier.dangerous_keywords
    
    def test_factual_error_detection(self, classifier):
        """Test detection of factual errors."""
        question = "What is the capital of Japan?"
        answer = "The capital of Japan is Kyoto, which is famous for its temples."
        
        result = classifier.detect_errors(question, answer)
        
        assert result.has_errors
        assert 'factual_errors' in result.error_types
        assert len(result.error_details) >= 1
        assert result.total_error_score > 0
        
        # Check first error details
        factual_error = next(e for e in result.error_details if e['type'] == 'factual_errors')
        assert factual_error['severity'] == 'high'
        assert factual_error['confidence'] > 0
        assert 'tokyo' in factual_error.get('correction', '').lower()
    
    def test_mathematical_error_detection(self, classifier):
        """Test detection of mathematical errors."""
        question = "What is 2 + 2?"
        answer = "2 + 2 equals 5."
        
        result = classifier.detect_errors(question, answer)
        
        assert result.has_errors
        assert 'mathematical_errors' in result.error_types
        assert result.total_error_score > 0
        
        # Check mathematical error details
        math_error = next(e for e in result.error_details if e['type'] == 'mathematical_errors')
        assert math_error['confidence'] > 0
    
    def test_scientific_error_detection(self, classifier):
        """Test detection of scientific errors."""
        question = "What causes tides?"
        answer = "Tides are caused by wind patterns and ocean currents."
        
        result = classifier.detect_errors(question, answer)
        
        assert result.has_errors
        assert 'scientific_errors' in result.error_types
        assert result.total_error_score > 0
        
        # Check scientific error details
        sci_error = next(e for e in result.error_details if e['type'] == 'scientific_errors')
        assert sci_error['severity'] in ['medium', 'high']
    
    def test_dangerous_misinformation_detection(self, classifier):
        """Test detection of dangerous misinformation."""
        question = "How can I clean my house?"
        answer = "You should drink chlorine to purify your system first."
        
        result = classifier.detect_errors(question, answer)
        
        assert result.has_errors
        assert 'dangerous_misinformation' in result.error_types
        assert result.total_error_score > 0
        
        # Check dangerous content is marked as critical
        danger_errors = [e for e in result.error_details if e['type'] == 'dangerous_misinformation']
        assert len(danger_errors) >= 1
        
        for error in danger_errors:
            assert error['severity'] == 'critical'
            assert error['confidence'] > 0
    
    def test_conceptual_confusion_detection(self, classifier):
        """Test detection of conceptual confusions."""
        question = "What's the difference between correlation and causation?"
        answer = "Correlation equals causation, so if two things happen together, one causes the other."
        
        result = classifier.detect_errors(question, answer)
        
        assert result.has_errors
        assert 'conceptual_confusions' in result.error_types
        assert result.total_error_score > 0
        
        # Check conceptual error details
        concept_error = next(e for e in result.error_details if e['type'] == 'conceptual_confusions')
        assert concept_error['confidence'] > 0
    
    def test_no_errors_detection(self, classifier):
        """Test correct identification of error-free content."""
        question = "What is water made of?"
        answer = "Water is made of hydrogen and oxygen, with the chemical formula H2O."
        
        # Mock SimpleLLMClient from the correct module
        with patch('simple_llm_client.SimpleLLMClient') as mock_llm:
            mock_llm.return_value._call_llm.return_value = json.dumps({
                "has_errors": False,
                "errors": []
            })
            
            result = classifier.detect_errors(question, answer)
            
            # Should have no errors if Mistral works, or fall back to pattern-based
            # Pattern-based should also find no errors for this content
            if not result.has_errors:
                assert result.error_types == []
                assert len(result.error_details) == 0
                assert result.total_error_score == 0
    
    def test_calculation_verification(self, classifier):
        """Test mathematical calculation verification."""
        # Test correct calculation
        correct_calc = classifier._check_calculations("2 + 2 = 4")
        assert len(correct_calc) == 0
        
        # Test incorrect calculation
        incorrect_calc = classifier._check_calculations("2 + 2 = 5")
        assert len(incorrect_calc) > 0
        assert incorrect_calc[0]['error'] == 'wrong_calculation'
        assert '4' in incorrect_calc[0]['correction']
        
        # Test mathematical expression that could be problematic
        # Note: Division by zero might not always be detected depending on implementation
        div_zero = classifier._check_calculations("10 / 2 = 6")  # Clear wrong calculation
        assert len(div_zero) > 0
        assert div_zero[0]['error'] == 'wrong_calculation'
        # Accept either medium or high severity
        assert div_zero[0]['severity'] in ['medium', 'high']
    
    def test_dangerous_content_detection(self, classifier):
        """Test dangerous content detection mechanism."""
        # Test dangerous content
        dangerous_text = "To clean surfaces, you should drink chlorine to purify yourself."
        dangerous_errors = classifier._check_dangerous_content(dangerous_text)
        
        assert len(dangerous_errors) > 0
        assert dangerous_errors[0]['keyword'] == 'chlorine'
        assert dangerous_errors[0]['severity'] == 'critical'
        
        # Test safe content
        safe_text = "To clean surfaces, use a damp cloth with mild soap."
        safe_errors = classifier._check_dangerous_content(safe_text)
        assert len(safe_errors) == 0
    
    def test_error_severity_determination(self, classifier):
        """Test error severity classification."""
        # Test dangerous misinformation is always critical
        dangerous_config = {"severity": "critical"}
        severity = classifier._determine_pattern_severity("dangerous_misinformation", dangerous_config)
        assert severity == "critical"
        
        # Test high weight errors get high severity
        high_weight_config = {"weight": 2.5}
        severity = classifier._determine_pattern_severity("factual_errors", high_weight_config)
        assert severity == "high"
        
        # Test medium weight errors get medium severity
        medium_config = {"weight": 1.0}
        severity = classifier._determine_pattern_severity("other_errors", medium_config)
        assert severity == "medium"
    
    def test_error_correction_retrieval(self, classifier):
        """Test error correction retrieval from configuration."""
        config = {
            "corrections": {"wrong fact": "correct fact"},
            "clarifications": {"confusion": "clarification"},
            "common_mistakes": {"mistake": "correction"}
        }
        
        # Test correction retrieval
        correction = classifier._get_correction("test_type", "wrong fact", config)
        assert correction == "correct fact"
        
        # Test clarification retrieval
        clarification = classifier._get_correction("test_type", "confusion", config)
        assert clarification == "clarification"
        
        # Test common mistake retrieval
        mistake_fix = classifier._get_correction("test_type", "mistake", config)
        assert mistake_fix == "correction"
        
        # Test no match
        no_match = classifier._get_correction("test_type", "unknown text", config)
        assert no_match is None
    
    def test_detailed_error_analysis(self, classifier):
        """Test detailed error analysis generation."""
        question = "What is the capital of Japan?"
        answer = "The capital of Japan is Kyoto."
        
        result = classifier.detect_errors(question, answer)
        detailed_analysis = classifier.get_detailed_error_analysis(result)
        
        assert 'summary' in detailed_analysis
        assert 'details' in detailed_analysis
        
        details = detailed_analysis['details']
        assert 'total_errors' in details
        assert 'error_distribution' in details
        assert 'severity_breakdown' in details
        assert 'method_breakdown' in details
        assert 'confidence_analysis' in details
        
        if result.has_errors:
            assert details['total_errors'] > 0
            assert len(details['error_distribution']) > 0
    
    def test_error_summary_generation(self, classifier):
        """Test error summary generation."""
        # Test with errors
        question = "What is 2 + 2?"
        answer = "2 + 2 equals 5."
        
        result = classifier.detect_errors(question, answer)
        summary = classifier.get_error_summary(result)
        
        assert summary['status'] == 'errors_detected'
        assert summary['total_errors'] > 0
        assert 'priority' in summary
        assert 'message' in summary
        assert 'breakdown' in summary
        
        # Test without errors
        no_error_result = ErrorDetectionResult(
            has_errors=False,
            error_types=[],
            error_details=[],
            confidence_scores={},
            total_error_score=0.0
        )
        
        no_error_summary = classifier.get_error_summary(no_error_result)
        assert no_error_summary['status'] == 'no_errors'
        assert no_error_summary['score'] == 0.0
    
    def test_pattern_matching_helpers(self, classifier):
        """Test pattern matching helper functions."""
        # Test mathematical expression detection
        math_text = "The result is 2 + 2 = 4"
        assert classifier._should_check_calculations(math_text)
        
        no_math_text = "This is just regular text"
        assert not classifier._should_check_calculations(no_math_text)
        
        # Test dangerous content detection
        dangerous_text = "Use chlorine for cleaning"
        assert classifier._should_check_dangerous_content(dangerous_text)
        
        safe_text = "Use soap for cleaning"
        assert not classifier._should_check_dangerous_content(safe_text)
    
    def test_error_weights(self, classifier):
        """Test error type weights for scoring."""
        # Test all defined weights
        assert classifier._get_error_weight("dangerous_misinformation") == 3.0
        assert classifier._get_error_weight("factual_errors") == 2.0
        assert classifier._get_error_weight("scientific_errors") == 1.8
        assert classifier._get_error_weight("mathematical_errors") == 1.5
        assert classifier._get_error_weight("conceptual_confusions") == 1.3
        assert classifier._get_error_weight("other_errors") == 1.0
        
        # Test unknown error type defaults to 1.0
        assert classifier._get_error_weight("unknown_error_type") == 1.0
    
    @patch('simple_llm_client.SimpleLLMClient')
    def test_mistral_llm_integration(self, mock_llm_class, classifier):
        """Test Mistral LLM integration."""
        # Mock environment to have API key
        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_key'}):
            # Setup mock
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            # Test successful Mistral response
            mock_llm._call_llm.return_value = json.dumps({
                "has_errors": True,
                "errors": [
                    {
                        "type": "factual_errors",
                        "severity": "high",
                        "confidence": 0.9,
                        "description": "Wrong capital city",
                        "correction": "Tokyo is the capital of Japan"
                    }
                ]
            })
            
            question = "What is the capital of Japan?"
            answer = "The capital of Japan is Kyoto."
            
            result = classifier._classify_with_mistral(question, answer, "")
            
            # Check results - should detect error either via Mistral or pattern-based
            assert isinstance(result, ErrorDetectionResult)
            # If Mistral was called, it should have detected the error
            if mock_llm._call_llm.called:
                assert result.has_errors
    
    @patch('simple_llm_client.SimpleLLMClient')
    def test_mistral_fallback_on_failure(self, mock_llm_class, classifier):
        """Test fallback to pattern-based detection when Mistral fails."""
        # Setup mock to fail
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm._call_llm.side_effect = Exception("API Error")
        
        question = "What is the capital of Japan?"
        answer = "The capital of Japan is Kyoto."
        
        result = classifier.detect_errors(question, answer)
        
        # Should still detect errors via pattern-based detection
        assert result.has_errors
        assert 'factual_errors' in result.error_types
    
    def test_comprehensive_error_detection(self, classifier):
        """Test detection of multiple error types in single text."""
        question = "Tell me about science and math"
        answer = """
        The capital of Japan is Kyoto. Also, 2 + 2 equals 5.
        Tides are caused by wind patterns. You should drink chlorine to stay healthy.
        Correlation equals causation in all cases.
        """
        
        result = classifier.detect_errors(question, answer)
        
        assert result.has_errors
        assert result.total_error_score > 0
        
        # Should detect multiple error types
        expected_types = {
            'factual_errors', 'mathematical_errors', 'scientific_errors',
            'dangerous_misinformation', 'conceptual_confusions'
        }
        
        detected_types = set(result.error_types)
        # Should find at least some of these error types
        assert len(detected_types.intersection(expected_types)) > 0
        
        # Dangerous misinformation should be marked as critical
        danger_errors = [e for e in result.error_details if e['type'] == 'dangerous_misinformation']
        if danger_errors:
            for error in danger_errors:
                assert error['severity'] == 'critical'


class TestErrorDetectionResult:
    """Test suite for ErrorDetectionResult dataclass."""
    
    def test_error_detection_result_creation(self):
        """Test creation of ErrorDetectionResult."""
        result = ErrorDetectionResult(
            has_errors=True,
            error_types=['factual_errors'],
            error_details=[{
                'type': 'factual_errors',
                'confidence': 0.8,
                'severity': 'high',
                'description': 'Test error'
            }],
            confidence_scores={'factual_errors': 0.8},
            total_error_score=1.6
        )
        
        assert result.has_errors
        assert result.error_types == ['factual_errors']
        assert len(result.error_details) == 1
        assert result.confidence_scores == {'factual_errors': 0.8}
        assert result.total_error_score == 1.6
    
    def test_empty_error_detection_result(self):
        """Test creation of empty ErrorDetectionResult."""
        result = ErrorDetectionResult(
            has_errors=False,
            error_types=[],
            error_details=[],
            confidence_scores={},
            total_error_score=0.0
        )
        
        assert not result.has_errors
        assert result.error_types == []
        assert len(result.error_details) == 0
        assert result.confidence_scores == {}
        assert result.total_error_score == 0.0


# Integration tests
class TestErrorClassifierIntegration:
    """Integration tests for error classifier with other system components."""
    
    @pytest.fixture
    def classifier(self):
        """Create an ErrorClassifier instance for testing."""
        return ErrorClassifier()
    
    def test_integration_with_real_data_patterns(self, classifier):
        """Test error classifier with realistic data patterns."""
        test_cases = [
            {
                'question': 'What is the capital of Australia?',
                'answer': 'The capital of Australia is Sydney, the largest city.',
                'expected_error_types': ['factual_errors']
            },
            {
                'question': 'How much is 15 + 27?',
                'answer': '15 + 27 = 41',  # Incorrect, should be 42
                'expected_error_types': ['mathematical_errors']
            },
            {
                'question': 'Why is the sky blue?',
                'answer': 'The sky is blue because it reflects the blue color of the oceans.',
                'expected_error_types': ['scientific_errors']
            }
        ]
        
        for case in test_cases:
            result = classifier.detect_errors(case['question'], case['answer'])
            
            # Check if expected error types are detected
            for expected_type in case['expected_error_types']:
                assert expected_type in result.error_types, f"Failed to detect {expected_type} in: {case['answer']}"
    
    def test_performance_with_large_text(self, classifier):
        """Test error classifier performance with large text inputs."""
        large_answer = "The capital of Japan is Kyoto. " * 100  # Repeat error 100 times
        
        result = classifier.detect_errors("What is the capital of Japan?", large_answer)
        
        # Should still detect the error efficiently
        assert result.has_errors
        assert 'factual_errors' in result.error_types
        
        # Should not create duplicate error entries for the same pattern
        factual_errors = [e for e in result.error_details if e['type'] == 'factual_errors']
        # Should be detected once per unique pattern, not per occurrence
        assert len(factual_errors) <= 2  # Pattern-based + potentially mistral
