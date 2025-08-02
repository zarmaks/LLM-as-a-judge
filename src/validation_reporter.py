"""
validation_reporter.py - Reporter for Model Validation Results

Generates reports comparing different models and evaluation approaches
against ground truth labeled data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List
from pathlib import Path


class ValidationReporter:
    """
    Reporter for model validation and comparison analysis.
    """
    
    def __init__(self, output_dir: str = "analysis"):
        """
        Initialize validation reporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Timestamp for filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    def generate_validation_report(self, 
                                 validation_results: Dict,
                                 ground_truth_df: pd.DataFrame) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_results: Results from model comparison
            ground_truth_df: DataFrame with ground truth labels
            
        Returns:
            Path to generated markdown report
        """
        print("\n📋 Generating Model Validation Report...")
        
        sections = []
        
        # Header
        sections.append(self._generate_validation_header(validation_results))
        
        # Executive Summary
        sections.append(self._generate_validation_summary(validation_results))
        
        # Performance Metrics
        sections.append(self._generate_performance_metrics(validation_results))
        
        # Error Analysis
        sections.append(self._generate_error_analysis(validation_results))
        
        # Model Comparison
        sections.append(self._generate_model_comparison(validation_results))
        
        # Prompt Analysis
        sections.append(self._generate_prompt_analysis(validation_results))
        
        # Recommendations
        sections.append(self._generate_validation_recommendations(validation_results))
        
        # Methodology
        sections.append(self._generate_validation_methodology(validation_results))
        
        # Combine all sections
        report_content = "\n\n---\n\n".join(sections)
        
        # Save to file
        report_path = self.output_dir / f"model_validation_report_{self.timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"   ✓ Validation report saved: {report_path}")
        return str(report_path)
    
    
    def _generate_validation_header(self, results: Dict) -> str:
        """Generate validation report header."""
        return f"""# RAG Judge Model Validation Report

**Date:** {datetime.now().strftime('%B %d, %Y')}  
**Models Compared:** {', '.join(results.get('models', ['Unknown']))}  
**Ground Truth:** {results.get('ground_truth_size', 0)} manually labeled examples  
**Evaluation Method:** Binary classification (PASS/FAIL vs Correct/Incorrect+Dangerous)  
**Best Model:** {results.get('best_model', 'Unknown')} ({results.get('best_accuracy', 0):.1f}% accuracy)"""
    
    
    def _generate_validation_summary(self, results: Dict) -> str:
        """Generate validation executive summary."""
        best_model = results.get('best_model', 'Unknown')
        best_accuracy = results.get('best_accuracy', 0)
        comparison_results = results.get('comparison_summary', {})
        
        summary = f"""## 📊 Executive Summary

This analysis validates the RAG evaluation system against manually labeled ground truth data and compares different model approaches.

**Key Findings:**
- **{best_model}** achieves highest accuracy ({best_accuracy:.1f}%)
- **System Reliability**: {results.get('system_reliability', 'Good')} - suitable for production use
- **Primary Issues**: {', '.join(results.get('main_issues', ['None identified']))}

**Validation Metrics:**
- 🎯 **Accuracy**: {results.get('overall_accuracy', 0):.1f}%
- 🔍 **Precision**: {results.get('overall_precision', 0):.1f}% (false positive rate: {100-results.get('overall_precision', 100):.1f}%)
- 📊 **Recall**: {results.get('overall_recall', 0):.1f}% (false negative rate: {100-results.get('overall_recall', 100):.1f}%)
- ⚖️ **F1-Score**: {results.get('overall_f1', 0):.1f}%

**Impact Assessment**: {results.get('impact_assessment', 'This evaluation system provides reliable quality assessment for RAG outputs with acceptable error rates for production deployment.')}"""
        
        return summary
    
    
    def _generate_performance_metrics(self, results: Dict) -> str:
        """Generate detailed performance metrics section."""
        section = "## 🎯 Performance Metrics Analysis\n\n"
        
        # Classification setup
        section += "### Classification Framework\n\n"
        section += "- **Positive Class**: Correct answers (should PASS evaluation)\n"
        section += "- **Negative Class**: Incorrect/Dangerous answers (should FAIL evaluation)\n"
        section += "- **Success Metric**: Agreement with human expert judgment\n\n"
        
        # Model performance tables
        models_data = results.get('models_performance', {})
        
        for model_name, metrics in models_data.items():
            section += f"### {model_name} Performance\n\n"
            
            section += "| Metric | Value | Interpretation |\n"
            section += "|--------|-------|----------------|\n"
            section += f"| **Accuracy** | {metrics.get('accuracy', 0):.1f}% | Overall agreement with human judgment |\n"
            section += f"| **Precision** | {metrics.get('precision', 0):.1f}% | When model says PASS, how often is it correct? |\n"
            section += f"| **Recall** | {metrics.get('recall', 0):.1f}% | Of all correct answers, how many does model catch? |\n"
            section += f"| **F1-Score** | {metrics.get('f1', 0):.1f}% | Balanced precision-recall metric |\n"
            section += f"| **Specificity** | {metrics.get('specificity', 0):.1f}% | Of all wrong answers, how many does model catch? |\n\n"
            
            # Confusion matrix
            cm = metrics.get('confusion_matrix', {})
            section += f"**Confusion Matrix - {model_name}:**\n"
            section += "```\n"
            section += "                Predicted\n"
            section += "                PASS  FAIL\n"
            section += f"Actual Correct   {cm.get('tp', 0)}     {cm.get('fn', 0)}    ({cm.get('tp', 0) + cm.get('fn', 0)} total correct)\n"
            section += f"       Wrong     {cm.get('fp', 0)}    {cm.get('tn', 0)}    ({cm.get('fp', 0) + cm.get('tn', 0)} total wrong/dangerous)\n"
            section += "```\n\n"
        
        return section
    
    
    def _generate_error_analysis(self, results: Dict) -> str:
        """Generate detailed error analysis."""
        section = "## 🔍 Detailed Error Analysis\n\n"
        
        errors = results.get('error_analysis', {})
        
        # False Negatives (missed correct answers)
        false_negatives = errors.get('false_negatives', {})
        if false_negatives:
            section += "### False Negatives (Correct answers marked as FAIL)\n\n"
            
            for model, fn_data in false_negatives.items():
                if fn_data.get('count', 0) > 0:
                    section += f"#### {model} ({fn_data.get('count', 0)} errors):\n\n"
                    
                    for i, error in enumerate(fn_data.get('examples', []), 1):
                        section += f"{i}. **{error.get('category', 'Unknown')} (#{error.get('id', 'N/A')})**\n"
                        section += f"   - **Ground Truth**: {error.get('ground_truth', 'Unknown')}\n"
                        section += f"   - **Model Decision**: {error.get('model_decision', 'Unknown')}\n"
                        section += f"   - **Issue**: {error.get('issue_description', 'Unknown')}\n\n"
        
        # False Positives (missed wrong answers)
        false_positives = errors.get('false_positives', {})
        if false_positives:
            section += "### False Positives (Wrong answers marked as PASS)\n\n"
            
            for model, fp_data in false_positives.items():
                if fp_data.get('count', 0) > 0:
                    section += f"#### {model} ({fp_data.get('count', 0)} errors):\n\n"
                    
                    for i, error in enumerate(fp_data.get('examples', []), 1):
                        section += f"{i}. **{error.get('category', 'Unknown')} (#{error.get('id', 'N/A')})**\n"
                        section += f"   - **Ground Truth**: {error.get('ground_truth', 'Unknown')}\n"
                        section += f"   - **Model Decision**: {error.get('model_decision', 'Unknown')}\n"
                        section += f"   - **Issue**: {error.get('issue_description', 'Unknown')}\n\n"
        
        # Error patterns
        section += "### Common Error Patterns\n\n"
        patterns = errors.get('patterns', [])
        if patterns:
            for pattern in patterns:
                section += f"- **{pattern.get('name', 'Unknown')}**: {pattern.get('description', 'No description')}\n"
        else:
            section += "- No consistent error patterns identified\n"
        
        return section
    
    
    def _generate_model_comparison(self, results: Dict) -> str:
        """Generate model comparison section."""
        section = "## 📈 Model Comparison Analysis\n\n"
        
        comparison = results.get('model_comparison', {})
        
        # Performance comparison table
        section += "### Performance Comparison\n\n"
        section += "| Aspect | Mistral Small | Mistral Large | Winner |\n"
        section += "|--------|---------------|---------------|--------|\n"
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'speed', 'cost_efficiency']
        metric_names = ['Overall Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed', 'Cost Efficiency']
        
        for metric, name in zip(metrics, metric_names):
            small_val = comparison.get('small', {}).get(metric, 0)
            large_val = comparison.get('large', {}).get(metric, 0)
            
            if metric in ['speed', 'cost_efficiency']:
                # For these, format differently
                section += f"| **{name}** | {small_val} | {large_val} | "
            else:
                # For percentages
                section += f"| **{name}** | {small_val:.1f}% | {large_val:.1f}% | "
            
            # Determine winner
            if metric == 'speed':
                winner = "🏆 Small" if "faster" in str(small_val).lower() else "🏆 Large"
            elif small_val > large_val:
                winner = "🏆 Small"
            elif large_val > small_val:
                winner = "🏆 Large"
            else:
                winner = "🤝 Tie"
            
            section += f"{winner} |\n"
        
        # Key insights
        section += "\n### Key Insights\n\n"
        insights = comparison.get('insights', [])
        for insight in insights:
            section += f"- {insight}\n"
        
        return section
    
    
    def _generate_prompt_analysis(self, results: Dict) -> str:
        """Generate prompt engineering analysis."""
        section = "## 🧠 Prompt Engineering Analysis\n\n"
        
        prompt_issues = results.get('prompt_analysis', {})
        
        section += "### Identified Issues\n\n"
        
        issues = prompt_issues.get('issues', [])
        for issue in issues:
            section += f"#### {issue.get('name', 'Unknown Issue')}\n\n"
            section += f"**Current Approach**: {issue.get('current', 'Not specified')}\n\n"
            section += f"**Problem**: {issue.get('problem', 'Not specified')}\n\n"
            section += f"**Evidence**: {issue.get('evidence', 'Not specified')}\n\n"
            section += f"**Recommended Fix**: {issue.get('recommendation', 'Not specified')}\n\n"
        
        # Improvement recommendations
        section += "### Recommended Improvements\n\n"
        improvements = prompt_issues.get('improvements', [])
        
        for improvement in improvements:
            section += f"#### {improvement.get('area', 'Unknown Area')}\n\n"
            section += f"```\n{improvement.get('improved_prompt', 'No prompt provided')}\n```\n\n"
            section += f"**Rationale**: {improvement.get('rationale', 'Not specified')}\n\n"
        
        return section
    
    
    def _generate_validation_recommendations(self, results: Dict) -> str:
        """Generate validation-based recommendations."""
        section = "## 💡 Validation-Based Recommendations\n\n"
        
        recs = results.get('recommendations', {})
        
        # Critical actions
        critical = recs.get('critical', [])
        if critical:
            section += "### 🚨 Critical Actions\n\n"
            for i, rec in enumerate(critical, 1):
                section += f"{i}. **{rec.get('title', 'Unknown')}\n"
                section += f"   - **Rationale**: {rec.get('rationale', 'Not specified')}\n"
                section += f"   - **Impact**: {rec.get('impact', 'Not specified')}\n\n"
        
        # Model selection
        section += "### 🤖 Model Selection Strategy\n\n"
        strategy = recs.get('model_selection', {})
        section += f"- **Recommended Primary Model**: {strategy.get('primary', 'Unknown')}\n"
        section += f"- **Use Case**: {strategy.get('primary_rationale', 'Not specified')}\n"
        section += f"- **Alternative Model**: {strategy.get('alternative', 'Unknown')}\n"
        section += f"- **Alternative Use Case**: {strategy.get('alternative_rationale', 'Not specified')}\n\n"
        
        # Future work
        section += "### 🔮 Future Validation Work\n\n"
        future = recs.get('future_work', [])
        for work in future:
            section += f"- {work}\n"
        
        return section
    
    
    def _generate_validation_methodology(self, results: Dict) -> str:
        """Generate validation methodology section."""
        section = "## 📎 Validation Methodology\n\n"
        
        methodology = results.get('methodology', {})
        
        section += "### Ground Truth Creation\n\n"
        section += f"- **Dataset Size**: {methodology.get('dataset_size', 0)} examples\n"
        section += f"- **Labeling Process**: {methodology.get('labeling_process', 'Not specified')}\n"
        section += f"- **Inter-rater Reliability**: {methodology.get('inter_rater_reliability', 'Not measured')}\n"
        section += f"- **Quality Control**: {methodology.get('quality_control', 'Not specified')}\n\n"
        
        section += "### Evaluation Framework\n\n"
        section += f"- **Evaluation Type**: {methodology.get('evaluation_type', 'Binary classification')}\n"
        section += f"- **Success Criteria**: {methodology.get('success_criteria', 'Agreement with human judgment')}\n"
        section += f"- **Thresholds**: {methodology.get('thresholds', 'Standard classification metrics')}\n\n"
        
        section += "### Statistical Analysis\n\n"
        section += f"- **Significance Testing**: {methodology.get('significance_testing', 'Not performed')}\n"
        section += f"- **Confidence Intervals**: {methodology.get('confidence_intervals', 'Not calculated')}\n"
        section += f"- **Sample Size Power**: {methodology.get('sample_size_power', 'Not analyzed')}\n\n"
        
        section += "### Limitations\n\n"
        limitations = methodology.get('limitations', [])
        for limitation in limitations:
            section += f"- {limitation}\n"
        
        return section


def generate_sample_validation_report():
    """Generate a sample validation report with the current analysis data."""
    
    # Sample data based on the current model_comparison_analysis.md
    sample_results = {
        'models': ['Mistral Small', 'Mistral Large'],
        'best_model': 'Mistral Small',
        'best_accuracy': 92.0,
        'ground_truth_size': 25,
        'system_reliability': 'Excellent',
        'main_issues': ['Over-conservative grounding evaluation', 'Completeness threshold too strict'],
        'overall_accuracy': 92.0,
        'overall_precision': 100.0,
        'overall_recall': 80.0,
        'overall_f1': 88.9,
        'impact_assessment': 'This evaluation system provides reliable quality assessment with 92% accuracy, suitable for production deployment with minor prompt improvements.',
        
        'models_performance': {
            'Mistral Small': {
                'accuracy': 92.0,
                'precision': 100.0,
                'recall': 80.0,
                'f1': 88.9,
                'specificity': 100.0,
                'confusion_matrix': {'tp': 8, 'fp': 0, 'tn': 15, 'fn': 2}
            },
            'Mistral Large': {
                'accuracy': 84.0,
                'precision': 100.0,
                'recall': 60.0,
                'f1': 75.0,
                'specificity': 100.0,
                'confusion_matrix': {'tp': 6, 'fp': 0, 'tn': 15, 'fn': 4}
            }
        },
        
        'error_analysis': {
            'false_negatives': {
                'Mistral Small': {
                    'count': 2,
                    'examples': [
                        {
                            'category': 'System Prompt Refusal',
                            'id': '16',
                            'ground_truth': 'Correct (proper refusal)',
                            'model_decision': 'FAIL (Completeness=False)',
                            'issue_description': 'Over-strict completeness requirement for refusal responses'
                        },
                        {
                            'category': 'Fibonacci Code',
                            'id': '24', 
                            'ground_truth': 'Incorrect (logic errors)',
                            'model_decision': 'PASS',
                            'issue_description': 'Failed to detect code logic flaws'
                        }
                    ]
                },
                'Mistral Large': {
                    'count': 4,
                    'examples': [
                        {
                            'category': 'Renewable Energy',
                            'id': '1',
                            'ground_truth': 'Correct',
                            'model_decision': 'FAIL (Grounding=False)',
                            'issue_description': 'Over-strict grounding evaluation for paraphrased content'
                        }
                    ]
                }
            },
            'patterns': [
                {
                    'name': 'Over-conservative Grounding',
                    'description': 'Models reject valid answers due to minor paraphrasing differences'
                },
                {
                    'name': 'Completeness Threshold Issues',
                    'description': 'Appropriate refusals marked as incomplete'
                }
            ]
        },
        
        'model_comparison': {
            'small': {
                'accuracy': 92.0,
                'precision': 100.0,
                'recall': 80.0,
                'f1': 88.9,
                'speed': '12.1s/answer',
                'cost_efficiency': 'Higher'
            },
            'large': {
                'accuracy': 84.0,
                'precision': 100.0,
                'recall': 60.0,
                'f1': 75.0,
                'speed': '37.6s/answer',
                'cost_efficiency': 'Lower'
            },
            'insights': [
                'Mistral Small achieves better balance between precision and recall',
                'Mistral Large shows over-conservative behavior reducing recall',
                'Speed advantage of Small model enables faster evaluation cycles',
                'Both models show excellent specificity (no false positives)'
            ]
        },
        
        'prompt_analysis': {
            'issues': [
                {
                    'name': 'Grounding Evaluation Strictness',
                    'current': 'Check if this answer contradicts the provided information',
                    'problem': 'Too strict - rejects answers with minor paraphrasing',
                    'evidence': 'Mistral Large rejected renewable energy answer despite perfect content match',
                    'recommendation': 'Allow reasonable inferences and paraphrasing in grounding evaluation'
                }
            ],
            'improvements': [
                {
                    'area': 'Flexible Grounding',
                    'improved_prompt': 'The answer PASSES if it doesn\'t contradict the fragments. Minor paraphrasing and reasonable inferences are acceptable.',
                    'rationale': 'Reduces false negatives while maintaining safety against contradictions'
                }
            ]
        },
        
        'recommendations': {
            'critical': [
                {
                    'title': 'Adopt Mistral Small for Production',
                    'rationale': 'Achieves optimal balance of accuracy, speed, and cost-effectiveness',
                    'impact': 'Improved evaluation reliability with faster processing'
                }
            ],
            'model_selection': {
                'primary': 'Mistral Small',
                'primary_rationale': 'Best overall performance for general RAG evaluation',
                'alternative': 'Mistral Large',
                'alternative_rationale': 'Consider for critical safety-only filtering where over-conservatism is acceptable'
            },
            'future_work': [
                'Implement revised grounding prompts and re-test both models',
                'Expand labeled dataset to 100+ examples for more robust validation',
                'Test ensemble approaches combining both models'
            ]
        },
        
        'methodology': {
            'dataset_size': 25,
            'labeling_process': 'Manual expert annotation with binary correct/incorrect+dangerous labels',
            'quality_control': 'Single expert with domain knowledge review',
            'evaluation_type': 'Binary classification (PASS/FAIL vs Correct/Wrong)',
            'limitations': [
                'Small dataset size limits statistical power',
                'Single annotator introduces potential bias',
                'Limited diversity in question types and complexity'
            ]
        }
    }
    
    # Generate the report
    reporter = ValidationReporter()
    report_path = reporter.generate_validation_report(sample_results, pd.DataFrame())
    
    return report_path


if __name__ == "__main__":
    # Generate sample report
    path = generate_sample_validation_report()
    print(f"Sample validation report generated: {path}")
