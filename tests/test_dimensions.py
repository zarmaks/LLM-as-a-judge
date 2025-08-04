"""
Tests for dimensions.py - scoring calculations and validations.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dimensions import (
    BINARY_DIMENSIONS, SCALED_DIMENSIONS, TRADITIONAL_DIMENSIONS, SAFETY_DIMENSION,
    calculate_primary_composite_score, calculate_traditional_composite_score,
    categorize_primary_score, categorize_traditional_score, validate_all_dimensions
)


class TestDimensions:
    """Test suite for dimensions configuration and scoring."""
    
    def test_binary_dimensions_structure(self):
        """Test that binary dimensions are properly configured."""
        assert len(BINARY_DIMENSIONS) == 3
        assert "relevance" in BINARY_DIMENSIONS
        assert "grounding" in BINARY_DIMENSIONS
        assert "completeness" in BINARY_DIMENSIONS
        
        for key, dim in BINARY_DIMENSIONS.items():
            assert hasattr(dim, 'name')
            assert hasattr(dim, 'description')
            assert hasattr(dim, 'evaluation_criteria')
            assert isinstance(dim.name, str)
            assert len(dim.name) > 0
            assert len(dim.evaluation_criteria) > 50  # Should have substantial criteria
    
    def test_scaled_dimensions_structure(self):
        """Test that scaled dimensions are properly configured."""
        assert len(SCALED_DIMENSIONS) == 4
        assert "clarity" in SCALED_DIMENSIONS
        assert "tone" in SCALED_DIMENSIONS
        assert "context_awareness" in SCALED_DIMENSIONS
        assert "conciseness" in SCALED_DIMENSIONS
        
        for key, dim in SCALED_DIMENSIONS.items():
            assert hasattr(dim, 'name')
            assert hasattr(dim, 'description')
            assert hasattr(dim, 'evaluation_criteria')
            assert hasattr(dim, 'weight')
            assert dim.min_score == 0
            assert dim.max_score == 2
            assert 0 < dim.weight <= 1
    
    def test_traditional_dimensions_structure(self):
        """Test that traditional dimensions are properly configured."""
        assert len(TRADITIONAL_DIMENSIONS) == 7
        expected_dims = [
            "relevance", "grounding", "completeness", 
            "clarity_coherence", "tone_appropriateness", 
            "context_awareness", "safety"
        ]
        
        for dim_key in expected_dims:
            assert dim_key in TRADITIONAL_DIMENSIONS
            
        for key, dim in TRADITIONAL_DIMENSIONS.items():
            assert hasattr(dim, 'name')
            assert hasattr(dim, 'weight')
            assert dim.min_score == 0
            assert dim.max_score == 2
            assert 0 < dim.weight <= 1
    
    def test_safety_dimension_structure(self):
        """Test that safety dimension is properly configured."""
        assert "safety" in SAFETY_DIMENSION
        safety = SAFETY_DIMENSION["safety"]
        
        assert safety["min_score"] == -1
        assert safety["max_score"] == 1
        assert "name" in safety
        assert "evaluation_criteria" in safety
    
    def test_dimension_weights_sum_to_one(self):
        """Test that dimension weights sum to 1.0."""
        # Test scaled dimensions
        scaled_total = sum(dim.weight for dim in SCALED_DIMENSIONS.values())
        assert abs(scaled_total - 1.0) < 0.001
        
        # Test traditional dimensions
        trad_total = sum(dim.weight for dim in TRADITIONAL_DIMENSIONS.values())
        assert abs(trad_total - 1.0) < 0.001
    
    def test_validate_all_dimensions_success(self):
        """Test successful dimension validation."""
        # Should not raise any exception
        validate_all_dimensions()
    
    def test_primary_composite_score_core_failed(self):
        """Test primary composite score when core criteria fail."""
        binary_scores = {"relevance": False, "grounding": True, "completeness": True}
        scaled_scores = {"clarity": 2, "tone": 2, "conciseness": 2}
        safety_score = 0
        
        score, details = calculate_primary_composite_score(
            binary_scores, scaled_scores, safety_score
        )
        
        assert score == 0  # Safety adjustment of 0
        assert details["core_passed"] is False
        assert "relevance" in details["failed_criteria"]
        assert details["quality_score"] is None
    
    def test_primary_composite_score_core_passed_good_quality(self):
        """Test primary composite score with passing core and good quality."""
        binary_scores = {"relevance": True, "grounding": True, "completeness": True}
        scaled_scores = {"clarity": 2, "tone": 2, "conciseness": 2, "context_awareness": 2}
        safety_score = 0
        
        score, details = calculate_primary_composite_score(
            binary_scores, scaled_scores, safety_score, has_context=True
        )
        
        assert score == 2.0  # Average of 2s + safety 0
        assert details["core_passed"] is True
        assert details["quality_score"] == 2.0
        assert len(details["quality_dimensions_evaluated"]) == 4
    
    def test_primary_composite_score_no_context(self):
        """Test primary composite score without conversation context."""
        binary_scores = {"relevance": True, "grounding": True, "completeness": True}
        scaled_scores = {"clarity": 1, "tone": 2, "conciseness": 1, "context_awareness": None}
        safety_score = 0
        
        score, details = calculate_primary_composite_score(
            binary_scores, scaled_scores, safety_score, has_context=False
        )
        
        # Should average only non-context dimensions: (1+2+1)/3 = 1.33
        expected_quality = (1 + 2 + 1) / 3
        assert abs(score - expected_quality) < 0.01
        assert details["core_passed"] is True
        assert len(details["quality_dimensions_evaluated"]) == 3  # Excludes context_awareness
    
    def test_primary_composite_score_with_safety_bonus(self):
        """Test primary composite score with safety bonus."""
        binary_scores = {"relevance": True, "grounding": True, "completeness": True}
        scaled_scores = {"clarity": 1, "tone": 1, "conciseness": 1}
        safety_score = 1  # Good refusal
        
        score, details = calculate_primary_composite_score(
            binary_scores, scaled_scores, safety_score, has_context=False
        )
        
        # Quality average 1.0 + safety bonus 1 = 2.0
        assert score == 2.0
        assert details["safety_adjustment"] == 1
    
    def test_primary_composite_score_with_safety_penalty(self):
        """Test primary composite score with safety penalty."""
        binary_scores = {"relevance": True, "grounding": True, "completeness": True}
        scaled_scores = {"clarity": 2, "tone": 2, "conciseness": 2}
        safety_score = -1  # Dangerous content
        
        score, details = calculate_primary_composite_score(
            binary_scores, scaled_scores, safety_score, has_context=False
        )
        
        # Should be capped at safety score when negative
        assert score == -1
        assert details["safety_adjustment"] == -1
    
    def test_traditional_composite_score_all_dimensions(self):
        """Test traditional composite score with all dimensions."""
        dimension_scores = {
            "relevance": 2,
            "grounding": 2,
            "completeness": 1,
            "clarity_coherence": 2,
            "tone_appropriateness": 2,
            "context_awareness": 1,
            "safety": 2
        }
        
        score = calculate_traditional_composite_score(dimension_scores, has_context=True)
        
        # Calculate expected weighted average
        expected = 0
        for dim_key, dim_score in dimension_scores.items():
            expected += dim_score * TRADITIONAL_DIMENSIONS[dim_key].weight
        # No additional scaling needed - already 0-2
        
        assert abs(score - expected) < 0.01
        assert 0 <= score <= 2
    
    def test_traditional_composite_score_no_context(self):
        """Test traditional composite score without context."""
        dimension_scores = {
            "relevance": 2,
            "grounding": 2,
            "completeness": 1,
            "clarity_coherence": 2,
            "tone_appropriateness": 2,
            "safety": 2
        }
        
        score = calculate_traditional_composite_score(dimension_scores, has_context=False)
        
        # Should work without context_awareness
        assert 0 <= score <= 2
    
    def test_traditional_composite_score_missing_dimension(self):
        """Test traditional composite score with missing dimension."""
        dimension_scores = {
            "relevance": 2,
            "grounding": 2,
            # Missing other dimensions
        }
        
        with pytest.raises(ValueError, match="Missing score for dimension"):
            calculate_traditional_composite_score(dimension_scores)
    
    def test_traditional_composite_score_score_clamping(self):
        """Test that out-of-range scores are clamped."""
        dimension_scores = {
            "relevance": 5,  # Out of range
            "grounding": -1,  # Out of range
            "completeness": 1,
            "clarity_coherence": 2,
            "tone_appropriateness": 2,
            "context_awareness": 1,
            "safety": 2
        }
        
        # Should not raise exception due to clamping
        score = calculate_traditional_composite_score(dimension_scores, has_context=True)
        assert 0 <= score <= 2
    
    def test_categorize_primary_score_categories(self):
        """Test primary score categorization."""
        # Test different score ranges
        test_cases = [
            (-1, {"core_passed": True}, "! Unsafe Content"),
            (0, {"core_passed": False}, "X Failed Core Criteria"),
            (0.5, {"core_passed": True}, "! Poor Quality"),
            (1.5, {"core_passed": True}, "~ Acceptable"),
            (2.5, {"core_passed": True}, "+ Good"),
            (3.5, {"core_passed": True}, "* Excellent"),
        ]
        
        for score, details, expected_category in test_cases:
            category = categorize_primary_score(score, details)
            assert category == expected_category
    
    def test_categorize_traditional_score_categories(self):
        """Test traditional score categorization."""
        test_cases = [
            (0.3, "Poor"),
            (0.8, "Below Average"),
            (1.2, "Average"),
            (1.6, "Good"),
            (1.9, "Excellent"),
        ]
        
        for score, expected_category in test_cases:
            category = categorize_traditional_score(score)
            assert category == expected_category
    
    def test_dimension_criteria_content(self):
        """Test that evaluation criteria contain expected guidance."""
        # Binary dimensions should have PASS/FAIL examples
        for key, dim in BINARY_DIMENSIONS.items():
            criteria = dim.evaluation_criteria.lower()
            assert "pass" in criteria
            assert "fail" in criteria
            assert "example" in criteria
        
        # Scaled dimensions should have 0-2 scoring
        for key, dim in SCALED_DIMENSIONS.items():
            criteria = dim.evaluation_criteria.lower()
            assert "score 0" in criteria
            assert "score 1" in criteria
            assert "score 2" in criteria
        
        # Safety should have special scoring
        safety_criteria = SAFETY_DIMENSION["safety"]["evaluation_criteria"].lower()
        assert "score -1" in safety_criteria
        assert "score 0" in safety_criteria
        assert "score +1" in safety_criteria or "score 1" in safety_criteria
    
    def test_weight_distribution_fairness(self):
        """Test that no single dimension dominates the scoring."""
        # Check traditional weights
        for dim_key, dim in TRADITIONAL_DIMENSIONS.items():
            assert dim.weight <= 0.25, f"{dim_key} weight {dim.weight} too high"
        
        # Check scaled weights are equal (fair)
        scaled_weights = [dim.weight for dim in SCALED_DIMENSIONS.values()]
        assert all(w == scaled_weights[0] for w in scaled_weights), "Scaled weights should be equal"
    
    def test_score_boundary_conditions(self):
        """Test scoring functions with boundary conditions."""
        # Test with all minimum scores
        binary_all_pass = {"relevance": True, "grounding": True, "completeness": True}
        scaled_all_zero = {"clarity": 0, "tone": 0, "conciseness": 0}
        
        score, _ = calculate_primary_composite_score(binary_all_pass, scaled_all_zero, 0)
        assert score >= 0
        
        # Test with all maximum scores
        scaled_all_two = {"clarity": 2, "tone": 2, "conciseness": 2}
        score, _ = calculate_primary_composite_score(binary_all_pass, scaled_all_two, 1)
        assert score >= 2
        
        # Test traditional boundaries
        trad_all_zero = {key: 0 for key in TRADITIONAL_DIMENSIONS.keys()}
        score = calculate_traditional_composite_score(trad_all_zero)
        assert score >= 0
        
        trad_all_two = {key: 2 for key in TRADITIONAL_DIMENSIONS.keys()}
        score = calculate_traditional_composite_score(trad_all_two)
        assert score <= 2
    
    def test_dimension_naming_consistency(self):
        """Test that dimension names are consistent and clear."""
        # Check for consistent naming patterns
        for key, dim in BINARY_DIMENSIONS.items():
            assert key.lower() in dim.name.lower() or dim.name.lower() in key.lower()
        
        for key, dim in SCALED_DIMENSIONS.items():
            # Some flexibility for compound names like "clarity_coherence"
            assert any(part in dim.name.lower() for part in key.split('_'))
        
        for key, dim in TRADITIONAL_DIMENSIONS.items():
            assert any(part in dim.name.lower() for part in key.split('_'))
    
    def test_criteria_completeness(self):
        """Test that evaluation criteria are sufficiently detailed."""
        min_criteria_length = 100  # Minimum characters for useful criteria
        
        for key, dim in BINARY_DIMENSIONS.items():
            assert len(dim.evaluation_criteria) >= min_criteria_length, \
                f"Binary dimension {key} criteria too short"
        
        for key, dim in SCALED_DIMENSIONS.items():
            assert len(dim.evaluation_criteria) >= min_criteria_length, \
                f"Scaled dimension {key} criteria too short"
        
        for key, dim in TRADITIONAL_DIMENSIONS.items():
            assert len(dim.evaluation_criteria) >= min_criteria_length, \
                f"Traditional dimension {key} criteria too short"
