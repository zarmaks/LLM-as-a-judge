"""
Tests for Reporter class - report generation functionality.
"""

import pandas as pd
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reporter import Reporter


class TestReporter:
    """Test suite for Reporter class."""
    
    def test_reporter_initialization(self, temp_output_dir):
        """Test Reporter initialization."""
        reporter = Reporter(output_dir=temp_output_dir)
        
        assert reporter.output_dir == Path(temp_output_dir)
        assert reporter.output_dir.exists()
        assert hasattr(reporter, 'timestamp')
        assert len(reporter.timestamp) > 0
    
    def test_reporter_initialization_default_dir(self):
        """Test Reporter initialization with default directory."""
        with patch('reporter.Path.mkdir') as mock_mkdir:
            reporter = Reporter()
            
            assert reporter.output_dir == Path("reports")
            mock_mkdir.assert_called_once_with(exist_ok=True)
    
    def test_generate_report_basic(self, comprehensive_results_df, temp_output_dir):
        """Test basic report generation."""
        reporter = Reporter(output_dir=temp_output_dir)
        
        output_paths = reporter.generate_report(comprehensive_results_df)
        
        # Should generate at least CSV and markdown
        assert "csv" in output_paths
        assert "markdown" in output_paths
        
        # Files should exist
        for path in output_paths.values():
            assert os.path.exists(path)
    
    def test_generate_report_with_stats_json(self, comprehensive_results_df, temp_output_dir):
        """Test report generation with JSON statistics."""
        reporter = Reporter(output_dir=temp_output_dir)
        
        output_paths = reporter.generate_report(
            comprehensive_results_df, 
            include_stats_json=True
        )
        
        assert "statistics_json" in output_paths
        assert os.path.exists(output_paths["statistics_json"])
        
        # Verify JSON is valid
        with open(output_paths["statistics_json"], 'r') as f:
            stats = json.load(f)
            assert "metadata" in stats
            assert "summary_statistics" in stats
    
    def test_generate_enhanced_csv(self, comprehensive_results_df, temp_output_dir):
        """Test enhanced CSV generation."""
        reporter = Reporter(output_dir=temp_output_dir)
        
        csv_path = reporter._generate_enhanced_csv(comprehensive_results_df)
        
        # Verify CSV exists and is readable
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        
        # Should have original columns first
        original_cols = [
            "Current User Question", "Assistant Answer", 
            "Fragment Texts", "Conversation History"
        ]
        for col in original_cols:
            assert col in df.columns
        
        # Should have evaluation columns
        eval_cols = [
            "core_passed", "primary_composite_score", 
            "traditional_composite_score"
        ]
        for col in eval_cols:
            if col in comprehensive_results_df.columns:
                assert col in df.columns
    
    def test_generate_markdown_report(self, comprehensive_results_df, temp_output_dir):
        """Test markdown report generation."""
        reporter = Reporter(output_dir=temp_output_dir)
        
        md_path = reporter._generate_markdown_report(comprehensive_results_df)
        
        # Verify markdown exists and has content
        assert os.path.exists(md_path)
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have main sections
        expected_sections = [
            "# RAG System Evaluation Report",
            "## 📋 Executive Summary",
            "## 🎯 Primary Evaluation System Results",
            "## 📊 Traditional 7-Dimension Analysis",
            "## 🔍 Response Pattern Analysis",
            "## ⚠️ Safety Analysis",
            "## 💡 Recommendations"
        ]
        
        for section in expected_sections:
            assert section in content
    
    def test_generate_statistics_json(self, comprehensive_results_df, temp_output_dir):
        """Test JSON statistics generation."""
        reporter = Reporter(output_dir=temp_output_dir)
        
        json_path = reporter._generate_statistics_json(comprehensive_results_df)
        
        # Verify JSON structure
        assert os.path.exists(json_path)
        
        with open(json_path, 'r') as f:
            stats = json.load(f)
        
        # Check required top-level keys
        assert "metadata" in stats
        assert "summary_statistics" in stats
        assert "distributions" in stats
        assert "correlations" in stats
        assert "patterns" in stats
        
        # Check metadata structure
        metadata = stats["metadata"]
        assert "timestamp" in metadata
        assert "total_answers" in metadata
    
    def test_generate_failure_report(self, temp_output_dir):
        """Test failure report generation."""
        # Create DataFrame with failures
        data = {
            "Current User Question": ["What is X?", "What is Y?"],
            "Assistant Answer": ["X is...", "Y is..."],
            "Fragment Texts": ["Info X", "Info Y"],
            "Conversation History": ["", ""],
            "evaluation_failed": [True, False],
            "row_number": [1, 2],
            "core_passed": [False, True],
            "safety_score": [-1, 0]
        }
        df = pd.DataFrame(data)
        
        reporter = Reporter(output_dir=temp_output_dir)
        
        # Mock _has_failures to return True
        with patch.object(reporter, '_has_failures', return_value=True):
            failure_path = reporter._generate_failure_report(df)
        
        assert os.path.exists(failure_path)
        
        with open(failure_path, 'r') as f:
            failures = json.load(f)
        
        assert isinstance(failures, list)
        assert len(failures) > 0
    
    def test_calculate_key_metrics(self, comprehensive_results_df):
        """Test key metrics calculation."""
        reporter = Reporter()
        
        metrics = reporter._calculate_key_metrics(comprehensive_results_df)
        
        # Should calculate core pass rate
        if "core_passed" in comprehensive_results_df.columns:
            expected_pass_rate = comprehensive_results_df["core_passed"].sum() / len(comprehensive_results_df) * 100
            assert abs(metrics["core_pass_rate"] - expected_pass_rate) < 0.1
        
        # Should count safety issues
        if "safety_score" in comprehensive_results_df.columns:
            expected_safety_issues = len(comprehensive_results_df[comprehensive_results_df["safety_score"] < 0])
            assert metrics["safety_issues"] == expected_safety_issues
        
        # Should count attacks
        if "is_attack" in comprehensive_results_df.columns:
            expected_attacks = comprehensive_results_df["is_attack"].sum()
            assert metrics["attack_count"] == expected_attacks
    
    def test_calculate_overall_grade(self):
        """Test overall grade calculation."""
        reporter = Reporter()
        
        # Test excellent performance
        excellent_metrics = {
            "core_pass_rate": 95,
            "avg_quality": 1.8,
            "safety_issues": 0,
            "attack_count": 0
        }
        grade = reporter._calculate_overall_grade(excellent_metrics)
        assert grade["letter"] == "A"
        assert "excellent" in grade["description"]
        
        # Test poor performance
        poor_metrics = {
            "core_pass_rate": 40,
            "avg_quality": 0.5,
            "safety_issues": 5,
            "attack_count": 3
        }
        grade = reporter._calculate_overall_grade(poor_metrics)
        assert grade["letter"] in ["D", "F"]
    
    def test_identify_key_findings(self, comprehensive_results_df):
        """Test key findings identification."""
        reporter = Reporter()
        
        metrics = reporter._calculate_key_metrics(comprehensive_results_df)
        findings = reporter._identify_key_findings(comprehensive_results_df, metrics)
        
        assert isinstance(findings, list)
        
        # Should identify safety issues if present
        if metrics.get("safety_issues", 0) > 0:
            safety_finding = any("safety" in finding.lower() or "unsafe" in finding.lower() 
                               for finding in findings)
            assert safety_finding
    
    def test_get_common_pattern(self):
        """Test common pattern extraction."""
        reporter = Reporter()
        
        # Test with text series
        text_series = pd.Series([
            "The answer is not relevant to the question",
            "Not relevant to what was asked",
            "Answer does not address the question relevantly"
        ])
        
        pattern = reporter._get_common_pattern(text_series)
        assert isinstance(pattern, str)
        assert len(pattern) > 0
    
    def test_get_score_distribution(self):
        """Test score distribution string generation."""
        reporter = Reporter()
        
        scores = pd.Series([0, 1, 1, 2, 2, 2])
        distribution = reporter._get_score_distribution(scores)
        
        assert "0: 17%" in distribution
        assert "1: 33%" in distribution
        assert "2: 50%" in distribution
    
    def test_get_dimension_insight(self):
        """Test dimension insight generation."""
        reporter = Reporter()
        
        # Test high scores
        high_scores = pd.Series([2, 2, 1.8, 1.9])
        insight = reporter._get_dimension_insight("clarity", high_scores)
        assert "well" in insight.lower() or "excellent" in insight.lower()
        
        # Test low scores
        low_scores = pd.Series([0, 0.5, 0.3, 0.2])
        insight = reporter._get_dimension_insight("clarity", low_scores)
        assert "issue" in insight.lower() or "problem" in insight.lower()
    
    def test_generate_score_histogram(self):
        """Test ASCII histogram generation."""
        reporter = Reporter()
        
        scores = pd.Series([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        histogram = reporter._generate_score_histogram(scores)
        
        assert isinstance(histogram, str)
        assert "█" in histogram  # Should contain bar characters
        assert len(histogram.split('\n')) > 1  # Multiple lines
    
    def test_identify_failure_reason(self):
        """Test failure reason identification."""
        reporter = Reporter()
        
        # Test row with multiple failures
        row = pd.Series({
            "relevance_pass": False,
            "relevance_reason": "Off-topic",
            "grounding_pass": True,
            "completeness_pass": False,
            "completeness_reason": "Too vague"
        })
        
        reason = reporter._identify_failure_reason(row)
        assert "relevance" in reason.lower()
        assert "completeness" in reason.lower()
        assert "off-topic" in reason.lower()
        assert "vague" in reason.lower()
    
    def test_identify_quality_issues(self):
        """Test quality issue identification."""
        reporter = Reporter()
        
        row = pd.Series({
            "clarity_score": 0.5,
            "tone_score": 2.0,
            "conciseness_score": 0.3
        })
        
        issues = reporter._identify_quality_issues(row)
        assert "clarity" in issues.lower()
        assert "conciseness" in issues.lower()
        assert "tone" not in issues.lower()  # Tone score is good
    
    def test_generate_recommendations(self, comprehensive_results_df):
        """Test recommendation generation."""
        reporter = Reporter()
        
        recommendations = reporter._generate_recommendations(comprehensive_results_df)
        
        assert isinstance(recommendations, list)
        
        # Should have different priority levels
        priorities = {rec["priority"] for rec in recommendations}
        assert len(priorities) > 0
        
        # Check structure of recommendations
        for rec in recommendations:
            assert "priority" in rec
            assert "title" in rec
            assert "description" in rec
            assert rec["priority"] in ["CRITICAL", "HIGH", "MEDIUM"]
    
    def test_calculate_all_statistics(self, comprehensive_results_df):
        """Test comprehensive statistics calculation."""
        reporter = Reporter()
        
        stats = reporter._calculate_all_statistics(comprehensive_results_df)
        
        assert "total_answers" in stats
        assert stats["total_answers"] == len(comprehensive_results_df)
        
        # Should have pass rates if available
        if "core_passed" in comprehensive_results_df.columns:
            assert "core_pass_rate" in stats
    
    def test_calculate_distributions(self, comprehensive_results_df):
        """Test distribution calculations."""
        reporter = Reporter()
        
        distributions = reporter._calculate_distributions(comprehensive_results_df)
        
        # Should have categorical distributions
        categorical_cols = ["primary_category", "question_type", "attack_type"]
        for col in categorical_cols:
            if col in comprehensive_results_df.columns:
                assert col in distributions
    
    def test_calculate_correlations(self, comprehensive_results_df):
        """Test correlation calculations."""
        reporter = Reporter()
        
        correlations = reporter._calculate_correlations(comprehensive_results_df)
        
        # Should calculate some correlations
        assert isinstance(correlations, dict)
        
        # Check for specific correlation if columns exist
        if "answer_length" in comprehensive_results_df.columns and "primary_composite_score" in comprehensive_results_df.columns:
            assert "answer_length_vs_primary_composite_score" in correlations
    
    def test_extract_patterns(self, comprehensive_results_df):
        """Test pattern extraction."""
        reporter = Reporter()
        
        patterns = reporter._extract_patterns(comprehensive_results_df)
        
        # Should extract answer length patterns
        if "answer_length" in comprehensive_results_df.columns:
            assert "answer_length_patterns" in patterns
            
            length_patterns = patterns["answer_length_patterns"]
            assert "very_short_pct" in length_patterns
            assert "medium_pct" in length_patterns
        
        # Should extract attack patterns
        if "is_attack" in comprehensive_results_df.columns:
            assert "attack_patterns" in patterns
    
    def test_has_failures(self):
        """Test failure detection."""
        reporter = Reporter()
        
        # DataFrame with failures
        df_with_failures = pd.DataFrame({
            "evaluation_failed": [True, False, False]
        })
        assert reporter._has_failures(df_with_failures)
        
        # DataFrame without failures
        df_without_failures = pd.DataFrame({
            "evaluation_failed": [False, False, False]
        })
        assert not reporter._has_failures(df_without_failures)
        
        # DataFrame without evaluation_failed column
        df_no_column = pd.DataFrame({
            "other_column": [1, 2, 3]
        })
        assert not reporter._has_failures(df_no_column)
    
    def test_get_core_failure_details(self):
        """Test core failure details extraction."""
        reporter = Reporter()
        
        row = pd.Series({
            "relevance_pass": False,
            "relevance_reason": "Not relevant",
            "grounding_pass": True,
            "grounding_reason": "Well grounded",
            "completeness_pass": False,
            "completeness_reason": "Incomplete"
        })
        
        details = reporter._get_core_failure_details(row)
        
        assert "relevance" in details
        assert "grounding" in details
        assert "completeness" in details
        
        assert details["relevance"]["passed"] is False
        assert details["relevance"]["reason"] == "Not relevant"
        assert details["grounding"]["passed"] is True
    
    def test_generate_header(self, comprehensive_results_df):
        """Test report header generation."""
        reporter = Reporter()
        
        header = reporter._generate_header(comprehensive_results_df)
        
        assert "# RAG System Evaluation Report" in header
        assert "Generated" in header
        assert "Dataset" in header
        assert str(len(comprehensive_results_df)) in header
    
    def test_generate_executive_summary(self, comprehensive_results_df):
        """Test executive summary generation."""
        reporter = Reporter()
        
        summary = reporter._generate_executive_summary(comprehensive_results_df)
        
        assert "## 📋 Executive Summary" in summary
        assert "Key Metrics" in summary
        assert "Key Findings" in summary
        assert "Overall Assessment" in summary
    
    def test_report_with_minimal_data(self, temp_output_dir):
        """Test report generation with minimal data."""
        # Create minimal DataFrame
        minimal_data = {
            "Current User Question": ["What is X?"],
            "Assistant Answer": ["X is Y"],
            "Fragment Texts": ["Info about X"],
            "Conversation History": [""]
        }
        df = pd.DataFrame(minimal_data)
        
        reporter = Reporter(output_dir=temp_output_dir)
        
        # Should not crash with minimal data
        output_paths = reporter.generate_report(df)
        
        assert "csv" in output_paths
        assert "markdown" in output_paths
        
        for path in output_paths.values():
            assert os.path.exists(path)
    
    def test_report_with_missing_optional_columns(self, temp_output_dir):
        """Test report generation when optional columns are missing."""
        # Data without evaluation columns
        basic_data = {
            "Current User Question": ["What is X?", "What is Y?"],
            "Assistant Answer": ["X is something", "Y is another thing"],
            "Fragment Texts": ["Info X", "Info Y"],
            "Conversation History": ["", ""]
        }
        df = pd.DataFrame(basic_data)
        
        reporter = Reporter(output_dir=temp_output_dir)
        
        # Should handle missing evaluation columns gracefully
        output_paths = reporter.generate_report(df)
        
        assert "csv" in output_paths
        assert "markdown" in output_paths
