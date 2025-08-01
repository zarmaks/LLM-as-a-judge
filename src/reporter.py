"""
reporter.py - Enhanced Reporter for RAG Evaluation Results

Generates comprehensive reports including:
1. Markdown report with executive summary and detailed analysis
2. Enhanced CSV with all scores
3. JSON statistics for programmatic use
4. Optional visualizations
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List
from pathlib import Path


class Reporter:
    """
    Enhanced reporter for dual scoring system results.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize reporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Timestamp for filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    def generate_report(self, results_df: pd.DataFrame, 
                       include_stats_json: bool = True) -> Dict[str, str]:
        """
        Generate all report outputs.
        
        Args:
            results_df: DataFrame with evaluation results
            include_stats_json: Whether to generate JSON statistics file
            
        Returns:
            Dictionary with paths to generated files
        """
        print("\n📝 Generating reports...")
        
        output_paths = {}
        
        # 1. Generate enhanced CSV
        csv_path = self._generate_enhanced_csv(results_df)
        output_paths["csv"] = csv_path
        
        # 2. Generate markdown report
        md_path = self._generate_markdown_report(results_df)
        output_paths["markdown"] = md_path
        
        # 3. Generate JSON statistics (if requested)
        if include_stats_json:
            json_path = self._generate_statistics_json(results_df)
            output_paths["statistics_json"] = json_path
        
        # 4. Generate failure analysis (if any failures)
        if self._has_failures(results_df):
            failure_path = self._generate_failure_report(results_df)
            output_paths["failure_analysis"] = failure_path
        
        print("✅ Reports generated successfully!")
        
        return output_paths
    
    
    def _generate_enhanced_csv(self, results_df: pd.DataFrame) -> str:
        """Generate CSV with all evaluation results."""
        # Select columns in logical order
        base_cols = [
            "Current User Question",
            "Assistant Answer",
            "Fragment Texts", 
            "Conversation History"
        ]
        
        # Add evaluation columns based on what's present
        eval_cols = []
        
        # Primary system columns
        primary_cols = [
            "core_passed",
            "relevance_pass", "grounding_pass", "completeness_pass",
            "clarity_score", "tone_score", "context_awareness_score", "conciseness_score",
            "safety_score", "primary_composite_score", "primary_category"
        ]
        eval_cols.extend([col for col in primary_cols if col in results_df.columns])
        
        # Traditional system columns
        traditional_cols = [
            "traditional_composite_score", "traditional_category"
        ]
        eval_cols.extend([col for col in traditional_cols if col in results_df.columns])
        
        # Metadata columns
        meta_cols = [
            "is_attack", "attack_type", "answer_length", "question_type",
            "has_conversation_history", "evaluation_time", "row_number"
        ]
        eval_cols.extend([col for col in meta_cols if col in results_df.columns])
        
        # Combine all columns
        all_cols = base_cols + eval_cols
        ordered_df = results_df[[col for col in all_cols if col in results_df.columns]]
        
        # Save to CSV
        csv_path = self.output_dir / f"rag_evaluation_results_{self.timestamp}.csv"
        ordered_df.to_csv(csv_path, index=False)
        
        print(f"   ✓ Enhanced CSV saved: {csv_path}")
        return str(csv_path)
    
    
    def _generate_markdown_report(self, results_df: pd.DataFrame) -> str:
        """Generate comprehensive markdown report."""
        sections = []
        
        # Header
        sections.append(self._generate_header(results_df))
        
        # Executive Summary
        sections.append(self._generate_executive_summary(results_df))
        
        # Primary System Results (if present)
        if "primary_composite_score" in results_df.columns:
            sections.append(self._generate_primary_results_section(results_df))
        
        # Traditional System Results (if present)
        if "traditional_composite_score" in results_df.columns:
            sections.append(self._generate_traditional_results_section(results_df))
        
        # Pattern Analysis
        sections.append(self._generate_pattern_analysis_section(results_df))
        
        # Safety Analysis
        if "safety_score" in results_df.columns:
            sections.append(self._generate_safety_analysis_section(results_df))
        
        # Failure Examples
        sections.append(self._generate_failure_examples_section(results_df))
        
        # Statistical Insights
        sections.append(self._generate_statistical_insights_section(results_df))
        
        # Recommendations
        sections.append(self._generate_recommendations_section(results_df))
        
        # Methodology
        sections.append(self._generate_methodology_section(results_df))
        
        # Combine all sections
        report_content = "\n\n---\n\n".join(sections)
        
        # Save to file
        md_path = self.output_dir / f"rag_evaluation_report_{self.timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"   ✓ Markdown report saved: {md_path}")
        return str(md_path)
    
    
    def _generate_header(self, results_df: pd.DataFrame) -> str:
        """Generate report header."""
        metadata = results_df.attrs.get("evaluation_metadata", {})
        
        return f"""# RAG System Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset**: {len(results_df)} answers evaluated  
**Scoring Mode**: {metadata.get('scoring_mode', 'Unknown')}  
**Evaluation Time**: {metadata.get('total_time', 0)/60:.1f} minutes  
**Failures**: {metadata.get('failures', 0)}"""
    
    
    def _generate_executive_summary(self, results_df: pd.DataFrame) -> str:
        """Generate executive summary section."""
        # Calculate key metrics
        metrics = self._calculate_key_metrics(results_df)
        
        # Determine overall grade
        grade = self._calculate_overall_grade(metrics)
        
        # Key findings
        findings = self._identify_key_findings(results_df, metrics)
        
        summary = f"""## 📋 Executive Summary

The RAG system evaluation reveals **{grade['description']}** with an overall grade of **{grade['letter']}**.

**Key Metrics:**
- 🎯 Core Criteria Pass Rate: {metrics.get('core_pass_rate', 0):.0f}%
- 📊 Average Quality Score: {metrics.get('avg_quality', 0):.1f}/2 (for passing answers)
- ⚠️ Safety Issues: {metrics.get('safety_issues', 0)} answers
- 🔍 Attack Attempts: {metrics.get('attack_count', 0)} detected

**Key Findings:**"""
        
        for finding in findings[:5]:  # Top 5 findings
            summary += f"\n- {finding}"
        
        summary += f"\n\n**Overall Assessment**: {grade['assessment']}"
        
        return summary
    
    
    def _generate_primary_results_section(self, results_df: pd.DataFrame) -> str:
        """Generate primary scoring system results."""
        section = "## 🎯 Primary Evaluation System Results\n\n"
        
        # Core criteria analysis
        section += "### Core Criteria (Binary Pass/Fail)\n\n"
        section += "| Criterion | Pass Rate | Failed Count | Most Common Reason |\n"
        section += "|-----------|-----------|--------------|--------------------|\n"
        
        for dim in ["relevance", "grounding", "completeness"]:
            if f"{dim}_pass" in results_df.columns:
                pass_col = f"{dim}_pass"
                reason_col = f"{dim}_reason"
                
                pass_rate = results_df[pass_col].sum() / len(results_df) * 100
                fail_count = len(results_df) - results_df[pass_col].sum()
                
                # Most common failure reason
                failures = results_df[~results_df[pass_col]]
                if len(failures) > 0 and reason_col in failures.columns:
                    common_reason = self._get_common_pattern(failures[reason_col])
                else:
                    common_reason = "N/A"
                
                section += f"| **{dim.capitalize()}** | {pass_rate:.1f}% | {fail_count} | {common_reason} |\n"
        
        # Overall core pass rate
        if "core_passed" in results_df.columns:
            overall_pass = results_df["core_passed"].sum() / len(results_df) * 100
            section += f"| **Overall** | **{overall_pass:.1f}%** | **{len(results_df) - results_df['core_passed'].sum()}** | Multiple criteria |\n"
        
        # Quality dimensions
        section += "\n### Quality Dimensions (0-2 Scale)\n"
        section += "*Evaluated only for answers that passed core criteria*\n\n"
        
        passing_df = results_df[results_df.get("core_passed", True)]
        
        if len(passing_df) > 0:
            section += "| Dimension | Avg Score | Distribution | Insights |\n"
            section += "|-----------|-----------|--------------|----------|\n"
            
            for dim in ["clarity", "tone", "context_awareness", "conciseness"]:
                col = f"{dim}_score"
                if col in passing_df.columns:
                    valid_scores = passing_df[col].dropna()
                    if len(valid_scores) > 0:
                        avg_score = valid_scores.mean()
                        dist = self._get_score_distribution(valid_scores)
                        insight = self._get_dimension_insight(dim, valid_scores)
                        
                        dim_name = dim.replace("_", " ").title()
                        section += f"| {dim_name} | {avg_score:.2f}/2 | {dist} | {insight} |\n"
        
        # Final scores
        if "primary_composite_score" in results_df.columns:
            section += "\n### Composite Score Distribution\n\n"
            section += "```\n"
            section += self._generate_score_histogram(results_df["primary_composite_score"])
            section += "\n```\n"
            
            # Categories
            if "primary_category" in results_df.columns:
                section += "\n### Answer Categories\n\n"
                categories = results_df["primary_category"].value_counts()
                
                for cat, count in categories.items():
                    pct = count / len(results_df) * 100
                    section += f"- {cat}: {count} answers ({pct:.0f}%)\n"
        
        return section
    
    
    def _generate_traditional_results_section(self, results_df: pd.DataFrame) -> str:
        """Generate traditional scoring system results."""
        section = "## 📊 Traditional 7-Dimension Analysis\n\n"
        section += "*All dimensions evaluated on 0-2 scale with weighted composite*\n\n"
        
        section += "| Dimension | Mean Score | Std Dev | Weight | Contribution |\n"
        section += "|-----------|------------|---------|--------|-------------|\n"
        
        # Import dimension info for weights
        from dimensions import TRADITIONAL_DIMENSIONS
        
        total_contribution = 0
        for dim_key, dim_info in TRADITIONAL_DIMENSIONS.items():
            col = f"trad_{dim_key}_score"
            if col in results_df.columns:
                valid_scores = results_df[col].dropna()
                if len(valid_scores) > 0:
                    mean = valid_scores.mean()
                    std = valid_scores.std()
                    weight = dim_info.weight
                    contribution = mean * weight
                    total_contribution += contribution
                    
                    section += f"| {dim_info.name} | {mean:.2f} | {std:.2f} | {weight*100:.0f}% | {contribution:.2f} |\n"
        
        # Composite score
        if "traditional_composite_score" in results_df.columns:
            trad_mean = results_df["traditional_composite_score"].mean()
            trad_std = results_df["traditional_composite_score"].std()
            section += f"| **Composite** | **{trad_mean:.2f}** | **{trad_std:.2f}** | 100% | {total_contribution:.2f} |\n"
        
        return section
    
    
    def _generate_pattern_analysis_section(self, results_df: pd.DataFrame) -> str:
        """Generate response pattern analysis."""
        section = "## 🔍 Response Pattern Analysis\n\n"
        
        # Answer length distribution
        if "answer_length" in results_df.columns:
            section += "### Answer Length Distribution\n\n"
            lengths = results_df["answer_length"]
            
            section += f"- Very Short (<30 chars): {len(lengths[lengths < 30])} ({len(lengths[lengths < 30])/len(results_df)*100:.0f}%)\n"
            section += f"- Short (30-50 chars): {len(lengths[(lengths >= 30) & (lengths < 50)])} ({len(lengths[(lengths >= 30) & (lengths < 50)])/len(results_df)*100:.0f}%)\n"
            section += f"- Medium (50-150 chars): {len(lengths[(lengths >= 50) & (lengths < 150)])} ({len(lengths[(lengths >= 50) & (lengths < 150)])/len(results_df)*100:.0f}%)\n"
            section += f"- Long (>150 chars): {len(lengths[lengths >= 150])} ({len(lengths[lengths >= 150])/len(results_df)*100:.0f}%)\n"
        
        # Question type analysis
        if "question_type" in results_df.columns:
            section += "\n### Performance by Question Type\n\n"
            section += "| Type | Count | Avg Score | Pass Rate |\n"
            section += "|------|-------|-----------|-----------|"
            
            score_col = "primary_composite_score" if "primary_composite_score" in results_df.columns else "traditional_composite_score"
            pass_col = "core_passed" if "core_passed" in results_df.columns else None
            
            for qtype in results_df["question_type"].unique():
                subset = results_df[results_df["question_type"] == qtype]
                count = len(subset)
                
                if score_col in subset.columns:
                    avg_score = subset[score_col].mean()
                else:
                    avg_score = 0
                    
                if pass_col and pass_col in subset.columns:
                    pass_rate = subset[pass_col].sum() / len(subset) * 100
                else:
                    pass_rate = 0
                    
                section += f"\n| {qtype.capitalize()} | {count} | {avg_score:.2f} | {pass_rate:.0f}% |"
        
        # Attack response analysis
        if "is_attack" in results_df.columns:
            attacks = results_df[results_df["is_attack"]]
            if len(attacks) > 0:
                section += "\n\n### Attack Response Behavior\n\n"
                
                if "attack_type" in attacks.columns:
                    section += "| Attack Type | Count | Properly Handled | Failed |\n"
                    section += "|-------------|-------|------------------|--------|"
                    
                    for atype in attacks["attack_type"].unique():
                        if pd.notna(atype):
                            subset = attacks[attacks["attack_type"] == atype]
                            count = len(subset)
                            
                            # Check if properly handled (safety score > 0)
                            if "safety_score" in subset.columns:
                                handled = len(subset[subset["safety_score"] > 0])
                            else:
                                handled = 0
                                
                            section += f"\n| {atype} | {count} | {handled} | {count - handled} |"
        
        # Conversation continuity
        if "has_conversation_history" in results_df.columns:
            with_history = results_df[results_df["has_conversation_history"]]
            if len(with_history) > 0:
                section += "\n\n### Conversation Continuity Analysis\n\n"
                section += f"- Answers with conversation history: {len(with_history)} ({len(with_history)/len(results_df)*100:.0f}%)\n"
                
                if "context_awareness_score" in with_history.columns:
                    context_scores = with_history["context_awareness_score"].dropna()
                    if len(context_scores) > 0:
                        section += f"- Average context awareness score: {context_scores.mean():.2f}/2\n"
                        section += f"- Excellent context use (score=2): {len(context_scores[context_scores == 2])} ({len(context_scores[context_scores == 2])/len(context_scores)*100:.0f}%)\n"
        
        return section
    
    
    def _generate_safety_analysis_section(self, results_df: pd.DataFrame) -> str:
        """Generate safety analysis section."""
        section = "## ⚠️ Safety Analysis\n\n"
        
        safety_scores = results_df["safety_score"]
        
        # Distribution table
        section += "| Safety Score | Count | Percentage | Description |\n"
        section += "|--------------|-------|------------|-------------|\n"
        
        score_descriptions = {
            -2: "Dangerous content",
            -1: "Borderline unsafe", 
            0: "Neutral/Safe",
            1: "Good refusal",
            2: "Excellent refusal"
        }
        
        for score in sorted(safety_scores.unique()):
            count = len(safety_scores[safety_scores == score])
            pct = count / len(results_df) * 100
            desc = score_descriptions.get(score, "Unknown")
            
            section += f"| {score:+d} | {count} | {pct:.0f}% | {desc} |\n"
        
        # Critical safety issues
        dangerous = results_df[results_df["safety_score"] < 0]
        if len(dangerous) > 0:
            section += f"\n### 🚨 Critical Safety Issues ({len(dangerous)} found)\n\n"
            
            # Show up to 3 examples
            for idx, (_, row) in enumerate(dangerous.head(3).iterrows()):
                section += f"**Example {idx+1}:**\n"
                question = str(row.get('Current User Question', 'N/A'))
                answer = str(row.get('Assistant Answer', 'N/A'))
                section += f"- Question: \"{question[:100]}...\"\n"
                section += f"- Answer: \"{answer[:100]}...\"\n"
                section += f"- Safety Score: {row['safety_score']}\n"
                
                if "safety_reason" in row:
                    section += f"- Reason: {row['safety_reason']}\n"
                section += "\n"
        
        return section
    
    
    def _generate_failure_examples_section(self, results_df: pd.DataFrame) -> str:
        """Generate notable failure examples."""
        section = "## ❌ Notable Failure Examples\n\n"
        
        # Identify different types of failures
        failures = []
        
        # Core criteria failures
        if "core_passed" in results_df.columns:
            core_failures = results_df[~results_df["core_passed"]]
            if len(core_failures) > 0:
                # Get worst scoring example
                if "primary_composite_score" in core_failures.columns:
                    worst = core_failures.nsmallest(1, "primary_composite_score").iloc[0]
                else:
                    worst = core_failures.iloc[0]
                    
                failures.append({
                    "type": "Core Criteria Failure",
                    "question": worst["Current User Question"],
                    "answer": worst["Assistant Answer"], 
                    "issue": self._identify_failure_reason(worst),
                    "score": worst.get("primary_composite_score", 0)
                })
        
        # Safety failures
        if "safety_score" in results_df.columns:
            safety_failures = results_df[results_df["safety_score"] < 0]
            if len(safety_failures) > 0:
                worst_safety = safety_failures.nsmallest(1, "safety_score").iloc[0]
                failures.append({
                    "type": "Safety Violation",
                    "question": worst_safety["Current User Question"],
                    "answer": worst_safety["Assistant Answer"],
                    "issue": f"Safety score: {worst_safety['safety_score']}",
                    "score": worst_safety.get("primary_composite_score", worst_safety["safety_score"])
                })
        
        # Poor quality (passed core but low quality)
        if "core_passed" in results_df.columns and "primary_composite_score" in results_df.columns:
            passed_but_poor = results_df[(results_df["core_passed"]) & 
                                        (results_df["primary_composite_score"] < 1)]
            if len(passed_but_poor) > 0:
                worst_quality = passed_but_poor.nsmallest(1, "primary_composite_score").iloc[0]
                failures.append({
                    "type": "Poor Quality Despite Passing Core",
                    "question": worst_quality["Current User Question"],
                    "answer": worst_quality["Assistant Answer"],
                    "issue": self._identify_quality_issues(worst_quality),
                    "score": worst_quality["primary_composite_score"]
                })
        
        # Display failures
        for idx, failure in enumerate(failures[:5]):  # Show up to 5
            section += f"### {idx+1}. {failure['type']}\n\n"
            
            # Handle potential None/NaN values
            question = str(failure.get('question', '')).strip() or '[Empty Question]'
            answer = str(failure.get('answer', '')).strip() or '[Empty Answer]'
            
            section += f"**Question**: \"{question[:150]}{'...' if len(question) > 150 else ''}\"\n\n"
            section += f"**Answer**: \"{answer[:150]}{'...' if len(answer) > 150 else ''}\"\n\n"
            section += f"**Issue**: {failure['issue']}\n\n"
            section += f"**Score**: {failure['score']:.1f}\n\n"
        
        return section
    
    
    def _generate_statistical_insights_section(self, results_df: pd.DataFrame) -> str:
        """Generate statistical insights."""
        section = "## 📈 Statistical Insights\n\n"
        
        # Correlation analysis
        section += "### Correlation Analysis\n\n"
        
        correlations = []
        
        # Check various correlations
        if "relevance_pass" in results_df.columns and "grounding_pass" in results_df.columns:
            corr = results_df["relevance_pass"].astype(int).corr(
                results_df["grounding_pass"].astype(int)
            )
            correlations.append(f"- Relevance ↔ Grounding: r = {corr:.2f}")
        
        if "answer_length" in results_df.columns and "primary_composite_score" in results_df.columns:
            # Filter out None values
            valid_mask = results_df["primary_composite_score"].notna()
            if valid_mask.sum() > 0:
                corr = results_df.loc[valid_mask, "answer_length"].corr(
                    results_df.loc[valid_mask, "primary_composite_score"]
                )
                correlations.append(f"- Answer Length ↔ Score: r = {corr:.2f}")
        
        if correlations:
            section += "\n".join(correlations) + "\n"
        
        # Performance consistency
        if "evaluation_time" in results_df.columns:
            section += "\n### Evaluation Performance\n\n"
            times = results_df["evaluation_time"]
            section += f"- Average evaluation time: {times.mean():.1f}s per answer\n"
            section += f"- Total evaluation time: {times.sum()/60:.1f} minutes\n"
            section += f"- Fastest evaluation: {times.min():.1f}s\n"
            section += f"- Slowest evaluation: {times.max():.1f}s\n"
        
        # Score stability
        if "primary_composite_score" in results_df.columns:
            scores = results_df["primary_composite_score"].dropna()
            if len(scores) > 1:
                section += "\n### Score Distribution Statistics\n\n"
                section += f"- Mean: {scores.mean():.2f}\n"
                section += f"- Median: {scores.median():.2f}\n"
                section += f"- Standard Deviation: {scores.std():.2f}\n"
                section += f"- Skewness: {scores.skew():.2f}\n"
                section += f"- 25th Percentile: {scores.quantile(0.25):.2f}\n"
                section += f"- 75th Percentile: {scores.quantile(0.75):.2f}\n"
        
        return section
    
    
    def _generate_recommendations_section(self, results_df: pd.DataFrame) -> str:
        """Generate recommendations based on analysis."""
        section = "## 💡 Recommendations\n\n"
        
        recommendations = self._generate_recommendations(results_df)
        
        # Categorize recommendations
        critical = [r for r in recommendations if r["priority"] == "CRITICAL"]
        high = [r for r in recommendations if r["priority"] == "HIGH"]
        medium = [r for r in recommendations if r["priority"] == "MEDIUM"]
        
        if critical:
            section += "### 🚨 Critical Actions Required\n\n"
            for idx, rec in enumerate(critical, 1):
                section += f"{idx}. **{rec['title']}**\n"
                section += f"   - {rec['description']}\n"
                section += f"   - Impact: {rec['impact']}\n\n"
        
        if high:
            section += "### 📊 High Priority Improvements\n\n"
            for idx, rec in enumerate(high, 1):
                section += f"{idx}. **{rec['title']}**\n"
                section += f"   - {rec['description']}\n"
                section += f"   - Impact: {rec['impact']}\n\n"
        
        if medium:
            section += "### 🔧 Medium Priority Enhancements\n\n"
            for idx, rec in enumerate(medium, 1):
                section += f"{idx}. **{rec['title']}**\n"
                section += f"   - {rec['description']}\n\n"
        
        return section
    
    
    def _generate_methodology_section(self, results_df: pd.DataFrame) -> str:
        """Generate methodology section."""
        metadata = results_df.attrs.get("evaluation_metadata", {})
        
        section = "## 📎 Appendix: Methodology\n\n"
        
        section += f"**Evaluation System**: {metadata.get('scoring_mode', 'Unknown').capitalize()} Scoring\n\n"
        
        if metadata.get('scoring_mode') in ['dual', 'primary']:
            section += "**Primary System**:\n"
            section += "- Binary Core Criteria: Relevance, Grounding, Completeness (Pass/Fail)\n"
            section += "- Quality Dimensions: Clarity, Tone, Context Awareness, Conciseness (0-2 scale)\n"
            section += "- Safety: Special scoring (-1 to +1)\n"
            section += "- Composite: Quality average + Safety adjustment\n\n"
        
        if metadata.get('scoring_mode') in ['dual', 'traditional']:
            section += "**Traditional System**:\n"
            section += "- 7 Dimensions: Each scored 0-2\n"
            section += "- Weighted composite with predefined weights\n\n"
        
        section += "**Evaluation Details**:\n"
        section += "- LLM Model: Mistral-small-latest\n"
        section += "- Temperature: 0.0 (deterministic)\n"
        section += f"- Total Time: {metadata.get('total_time', 0)/60:.1f} minutes\n"
        section += f"- Average per Answer: {metadata.get('avg_time_per_answer', 0):.1f} seconds\n"
        
        return section
    
    
    def _generate_statistics_json(self, results_df: pd.DataFrame) -> str:
        """Generate JSON file with all statistics."""
        stats = {
            "metadata": {
                "timestamp": self.timestamp,
                "total_answers": len(results_df),
                "evaluation_metadata": results_df.attrs.get("evaluation_metadata", {})
            },
            "summary_statistics": self._calculate_all_statistics(results_df),
            "distributions": self._calculate_distributions(results_df),
            "correlations": self._calculate_correlations(results_df),
            "patterns": self._extract_patterns(results_df)
        }
        
        # Save to JSON
        json_path = self.output_dir / f"rag_evaluation_statistics_{self.timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"   ✓ Statistics JSON saved: {json_path}")
        return str(json_path)
    
    
    def _generate_failure_report(self, results_df: pd.DataFrame) -> str:
        """Generate detailed failure analysis report."""
        # Identify all types of failures
        failures = []
        
        # Core failures
        if "core_passed" in results_df.columns:
            core_fails = results_df[~results_df["core_passed"]]
            for _, row in core_fails.iterrows():
                failures.append({
                    "row": row.get("row_number", "Unknown"),
                    "type": "Core Criteria",
                    "details": self._get_core_failure_details(row)
                })
        
        # Safety failures
        if "safety_score" in results_df.columns:
            safety_fails = results_df[results_df["safety_score"] < 0]
            for _, row in safety_fails.iterrows():
                failures.append({
                    "row": row.get("row_number", "Unknown"),
                    "type": "Safety",
                    "details": {
                        "score": row["safety_score"],
                        "reason": row.get("safety_reason", "Unknown")
                    }
                })
        
        # Save failure report
        failure_path = self.output_dir / f"failure_analysis_{self.timestamp}.json"
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        
        print(f"   ✓ Failure analysis saved: {failure_path}")
        return str(failure_path)
    
    
    # Helper methods
    
    def _has_failures(self, results_df: pd.DataFrame) -> bool:
        """Check if there are any failures to report."""
        if "evaluation_failed" in results_df.columns:
            return results_df["evaluation_failed"].any()
        return False
    
    
    def _calculate_key_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key metrics for executive summary."""
        metrics = {}
        
        # Core pass rate
        if "core_passed" in results_df.columns:
            metrics["core_pass_rate"] = results_df["core_passed"].sum() / len(results_df) * 100
        
        # Average quality (for passing answers)
        if "core_passed" in results_df.columns and "primary_composite_score" in results_df.columns:
            passing = results_df[results_df["core_passed"]]
            if len(passing) > 0:
                # Subtract safety score to get quality only
                if "safety_score" in passing.columns:
                    quality_scores = passing["primary_composite_score"] - passing["safety_score"]
                    metrics["avg_quality"] = quality_scores.mean()
                else:
                    metrics["avg_quality"] = passing["primary_composite_score"].mean()
            else:
                # No passing answers - set to 0
                metrics["avg_quality"] = 0.0
        
        # Safety issues
        if "safety_score" in results_df.columns:
            metrics["safety_issues"] = len(results_df[results_df["safety_score"] < 0])
        
        # Attack count
        if "is_attack" in results_df.columns:
            metrics["attack_count"] = results_df["is_attack"].sum()
        
        # Factual errors (approximation based on grounding failures)
        if "grounding_pass" in results_df.columns:
            metrics["factual_errors"] = len(results_df) - results_df["grounding_pass"].sum()
        
        return metrics
    
    
    def _calculate_overall_grade(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Calculate overall letter grade based on metrics."""
        # Simple grading logic
        score = 0
        
        # Core pass rate (40% weight)
        core_rate = metrics.get("core_pass_rate", 0)
        if core_rate >= 90:
            score += 40
        elif core_rate >= 80:
            score += 35
        elif core_rate >= 70:
            score += 30
        elif core_rate >= 60:
            score += 25
        else:
            score += core_rate * 0.25
        
        # Quality score (30% weight)
        quality = metrics.get("avg_quality", 0)
        score += (quality / 2) * 30
        
        # Safety (20% weight)
        safety_issues = metrics.get("safety_issues", 0)
        if safety_issues == 0:
            score += 20
        elif safety_issues <= 1:
            score += 15
        elif safety_issues <= 2:
            score += 10
        else:
            score += 5
        
        # Attack handling (10% weight)
        attacks = metrics.get("attack_count", 0)
        if attacks == 0:
            score += 10
        else:
            # Check how well attacks were handled
            # This is simplified - in real implementation would check actual handling
            score += 5
        
        # Convert to letter grade
        if score >= 90:
            grade = {"letter": "A", "description": "excellent performance", 
                    "assessment": "The system demonstrates high quality across all dimensions."}
        elif score >= 80:
            grade = {"letter": "B", "description": "good performance with minor issues",
                    "assessment": "The system performs well but has room for improvement."}
        elif score >= 70:
            grade = {"letter": "C", "description": "acceptable performance with notable issues",
                    "assessment": "The system meets basic requirements but needs significant improvements."}
        elif score >= 60:
            grade = {"letter": "D", "description": "poor performance requiring major improvements",
                    "assessment": "The system has serious deficiencies that need immediate attention."}
        else:
            grade = {"letter": "F", "description": "failing performance",
                    "assessment": "The system is not suitable for production use without major overhaul."}
        
        return grade
    
    
    def _identify_key_findings(self, results_df: pd.DataFrame, metrics: Dict[str, float]) -> List[str]:
        """Identify key findings for executive summary."""
        findings = []
        
        # Critical safety issues
        if metrics.get("safety_issues", 0) > 0:
            findings.append(f"🔴 **Critical**: {metrics['safety_issues']} answers contain unsafe content")
        
        # Poor core performance
        if metrics.get("core_pass_rate", 100) < 70:
            findings.append(f"🔴 **Critical**: Only {metrics['core_pass_rate']:.0f}% of answers pass basic criteria")
        
        # Attack handling
        if metrics.get("attack_count", 0) > 0:
            # Check if attacks were handled properly
            if "is_attack" in results_df.columns and "safety_score" in results_df.columns:
                attacks = results_df[results_df["is_attack"]]
                handled = len(attacks[attacks["safety_score"] > 0])
                if handled < len(attacks):
                    findings.append(f"🟡 **Warning**: {len(attacks) - handled}/{len(attacks)} attack attempts not properly refused")
        
        # Factual accuracy
        if metrics.get("factual_errors", 0) > len(results_df) * 0.2:
            findings.append(f"🟡 **Warning**: {metrics['factual_errors']} answers contain factual errors")
        
        # Quality insights
        if metrics.get("avg_quality", 0) < 1.0:
            findings.append(f"🟡 **Warning**: Low average quality score ({metrics.get('avg_quality', 0):.1f}/2) for passing answers")
        elif metrics.get("avg_quality", 0) > 1.7:
            findings.append(f"🟢 **Positive**: High quality scores for passing answers ({metrics.get('avg_quality', 0):.1f}/2)")
        
        # Context handling
        if "has_conversation_history" in results_df.columns:
            with_history = results_df[results_df["has_conversation_history"]]
            if len(with_history) > 0 and "context_awareness_score" in with_history.columns:
                context_avg = with_history["context_awareness_score"].dropna().mean()
                if context_avg < 1.0:
                    findings.append(f"🟡 **Warning**: Poor context awareness (avg {context_avg:.1f}/2)")
        
        return findings
    
    
    def _get_common_pattern(self, series: pd.Series) -> str:
        """Extract common pattern from text series."""
        if len(series) == 0:
            return "N/A"
        
        # Simple approach - find most common short phrase
        # In production, would use more sophisticated NLP
        common_words = []
        for text in series.dropna().head(10):
            words = str(text).lower().split()
            common_words.extend(words)
        
        if not common_words:
            return "Various"
        
        # Find most common meaningful word
        from collections import Counter
        word_counts = Counter(w for w in common_words if len(w) > 3)
        
        if word_counts:
            most_common = word_counts.most_common(1)[0][0]
            return f"Related to '{most_common}'"
        
        return "Various reasons"
    
    
    def _get_score_distribution(self, scores: pd.Series) -> str:
        """Get score distribution as string."""
        dist = scores.value_counts(normalize=True).sort_index()
        
        parts = []
        for score, pct in dist.items():
            parts.append(f"{score}: {pct*100:.0f}%")
        
        return ", ".join(parts)
    
    
    def _get_dimension_insight(self, dimension: str, scores: pd.Series) -> str:
        """Get insight for a specific dimension."""
        avg = scores.mean()
        
        insights = {
            "clarity": {
                "high": "Well-structured answers",
                "medium": "Generally clear",
                "low": "Clarity issues present"
            },
            "tone": {
                "high": "Consistently appropriate",
                "medium": "Mostly appropriate", 
                "low": "Tone problems detected"
            },
            "context_awareness": {
                "high": "Excellent continuity",
                "medium": "Adequate context use",
                "low": "Poor context integration"
            },
            "conciseness": {
                "high": "Well-balanced length",
                "medium": "Some verbosity",
                "low": "Length issues"
            }
        }
        
        if dimension not in insights:
            return "No specific insight"
        
        if avg >= 1.7:
            level = "high"
        elif avg >= 1.0:
            level = "medium"
        else:
            level = "low"
        
        return insights[dimension][level]
    
    
    def _generate_score_histogram(self, scores: pd.Series) -> str:
        """Generate ASCII histogram of scores."""
        # Simple ASCII visualization
        bins = np.arange(-2, 5, 0.5)
        hist, edges = np.histogram(scores.dropna(), bins=bins)
        
        max_count = max(hist) if len(hist) > 0 else 1
        
        lines = []
        for i, count in enumerate(hist):
            if count > 0:
                bar_length = int(count / max_count * 40)
                bar = "█" * bar_length
                label = f"[{edges[i]:.1f}, {edges[i+1]:.1f})"
                lines.append(f"{label:12} {bar} {count}")
        
        return "\n".join(lines)
    
    
    def _identify_failure_reason(self, row: pd.Series) -> str:
        """Identify why an answer failed core criteria."""
        reasons = []
        
        for dim in ["relevance", "grounding", "completeness"]:
            if f"{dim}_pass" in row and not row[f"{dim}_pass"]:
                if f"{dim}_reason" in row:
                    reasons.append(f"{dim.capitalize()}: {row[f'{dim}_reason']}")
                else:
                    reasons.append(f"{dim.capitalize()}: Failed")
        
        return "; ".join(reasons) if reasons else "Unknown failure"
    
    
    def _identify_quality_issues(self, row: pd.Series) -> str:
        """Identify quality issues in passing answers."""
        issues = []
        
        for dim in ["clarity", "tone", "context_awareness", "conciseness"]:
            col = f"{dim}_score"
            if col in row and pd.notna(row[col]) and row[col] < 1:
                issues.append(f"Low {dim.replace('_', ' ')}: {row[col]}/2")
        
        return "; ".join(issues) if issues else "General quality issues"
    
    
    def _get_core_failure_details(self, row: pd.Series) -> Dict:
        """Get detailed core failure information."""
        details = {}
        
        for dim in ["relevance", "grounding", "completeness"]:
            pass_col = f"{dim}_pass"
            reason_col = f"{dim}_reason"
            
            if pass_col in row:
                details[dim] = {
                    "passed": bool(row[pass_col]),
                    "reason": row.get(reason_col, "No reason provided")
                }
        
        return details
    
    
    def _generate_recommendations(self, results_df: pd.DataFrame) -> List[Dict]:
        """Generate prioritized recommendations."""
        recommendations = []
        metrics = self._calculate_key_metrics(results_df)
        
        # Safety recommendations
        if metrics.get("safety_issues", 0) > 0:
            recommendations.append({
                "priority": "CRITICAL",
                "title": "Implement Robust Safety Filters",
                "description": f"System produced {metrics['safety_issues']} unsafe responses",
                "impact": "Prevent harmful content generation"
            })
        
        # Core criteria recommendations
        if metrics.get("core_pass_rate", 100) < 80:
            # Identify which core criterion is weakest
            weak_criteria = []
            for dim in ["relevance", "grounding", "completeness"]:
                if f"{dim}_pass" in results_df.columns:
                    pass_rate = results_df[f"{dim}_pass"].sum() / len(results_df) * 100
                    if pass_rate < 80:
                        weak_criteria.append((dim, pass_rate))
            
            if weak_criteria:
                weakest = min(weak_criteria, key=lambda x: x[1])
                recommendations.append({
                    "priority": "HIGH",
                    "title": f"Improve {weakest[0].capitalize()} Performance",
                    "description": f"Only {weakest[1]:.0f}% pass rate for {weakest[0]}",
                    "impact": "Significant improvement in overall quality"
                })
        
        # Attack handling
        if "is_attack" in results_df.columns:
            attacks = results_df[results_df["is_attack"]]
            if len(attacks) > 0 and "safety_score" in attacks.columns:
                improperly_handled = len(attacks[attacks["safety_score"] <= 0])
                if improperly_handled > 0:
                    recommendations.append({
                        "priority": "CRITICAL",
                        "title": "Strengthen Attack Detection and Refusal",
                        "description": f"{improperly_handled} attack attempts not properly refused",
                        "impact": "Prevent jailbreaking and misuse"
                    })
        
        # Context awareness
        if "has_conversation_history" in results_df.columns:
            with_history = results_df[results_df["has_conversation_history"]]
            if len(with_history) > 0 and "context_awareness_score" in with_history.columns:
                context_scores = with_history["context_awareness_score"].dropna()
                if len(context_scores) > 0 and context_scores.mean() < 1.5:
                    recommendations.append({
                        "priority": "MEDIUM",
                        "title": "Enhance Conversation Context Handling",
                        "description": f"Average context score only {context_scores.mean():.1f}/2",
                        "impact": "Better multi-turn conversation quality"
                    })
        
        # Answer length optimization
        if "answer_length" in results_df.columns:
            very_short = len(results_df[results_df["answer_length"] < 30])
            if very_short > len(results_df) * 0.2:
                recommendations.append({
                    "priority": "MEDIUM",
                    "title": "Improve Answer Completeness",
                    "description": f"{very_short} answers are too brief (<30 chars)",
                    "impact": "More informative responses"
                })
        
        return recommendations
    
    
    def _calculate_all_statistics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics."""
        stats = {}
        
        # Basic counts
        stats["total_answers"] = len(results_df)
        
        # Core criteria stats
        if "core_passed" in results_df.columns:
            stats["core_pass_rate"] = results_df["core_passed"].sum() / len(results_df)
            
            for dim in ["relevance", "grounding", "completeness"]:
                if f"{dim}_pass" in results_df.columns:
                    stats[f"{dim}_pass_rate"] = results_df[f"{dim}_pass"].sum() / len(results_df)
        
        # Score statistics
        for score_col in ["primary_composite_score", "traditional_composite_score"]:
            if score_col in results_df.columns:
                scores = results_df[score_col].dropna()
                if len(scores) > 0:
                    stats[f"{score_col}_stats"] = {
                        "mean": float(scores.mean()),
                        "std": float(scores.std()),
                        "min": float(scores.min()),
                        "max": float(scores.max()),
                        "median": float(scores.median()),
                        "q1": float(scores.quantile(0.25)),
                        "q3": float(scores.quantile(0.75))
                    }
        
        return stats
    
    
    def _calculate_distributions(self, results_df: pd.DataFrame) -> Dict:
        """Calculate score distributions."""
        distributions = {}
        
        # Categorical distributions
        for col in ["primary_category", "traditional_category", "question_type", "attack_type"]:
            if col in results_df.columns:
                distributions[col] = results_df[col].value_counts().to_dict()
        
        # Numeric distributions (binned)
        if "primary_composite_score" in results_df.columns:
            scores = results_df["primary_composite_score"].dropna()
            if len(scores) > 0:
                bins = [-2, -1, 0, 1, 2, 3, 4]
                hist, edges = np.histogram(scores, bins=bins)
                distributions["primary_score_bins"] = {
                    f"[{edges[i]}, {edges[i+1]})": int(hist[i]) 
                    for i in range(len(hist))
                }
        
        return distributions
    
    
    def _calculate_correlations(self, results_df: pd.DataFrame) -> Dict:
        """Calculate correlations between metrics."""
        correlations = {}
        
        # Define pairs to check
        pairs = [
            ("answer_length", "primary_composite_score"),
            ("answer_length", "clarity_score"),
            ("relevance_pass", "grounding_pass"),
            ("core_passed", "primary_composite_score")
        ]
        
        for col1, col2 in pairs:
            if col1 in results_df.columns and col2 in results_df.columns:
                # Handle boolean columns
                if results_df[col1].dtype == bool:
                    s1 = results_df[col1].astype(int)
                else:
                    s1 = results_df[col1]
                    
                if results_df[col2].dtype == bool:
                    s2 = results_df[col2].astype(int)
                else:
                    s2 = results_df[col2]
                
                # Calculate correlation, handling NaN
                valid_mask = s1.notna() & s2.notna()
                if valid_mask.sum() > 1:
                    corr = s1[valid_mask].corr(s2[valid_mask])
                    correlations[f"{col1}_vs_{col2}"] = float(corr) if not pd.isna(corr) else None
        
        return correlations
    
    
    def _extract_patterns(self, results_df: pd.DataFrame) -> Dict:
        """Extract behavioral patterns."""
        patterns = {}
        
        # Answer length patterns
        if "answer_length" in results_df.columns:
            lengths = results_df["answer_length"]
            patterns["answer_length_patterns"] = {
                "very_short_pct": len(lengths[lengths < 30]) / len(lengths),
                "short_pct": len(lengths[(lengths >= 30) & (lengths < 50)]) / len(lengths),
                "medium_pct": len(lengths[(lengths >= 50) & (lengths < 150)]) / len(lengths),
                "long_pct": len(lengths[lengths >= 150]) / len(lengths)
            }
        
        # Attack response patterns
        if "is_attack" in results_df.columns:
            attacks = results_df[results_df["is_attack"]]
            if len(attacks) > 0:
                patterns["attack_patterns"] = {
                    "total_attacks": len(attacks),
                    "attack_rate": len(attacks) / len(results_df)
                }
                
                if "safety_score" in attacks.columns:
                    patterns["attack_patterns"]["properly_handled"] = len(attacks[attacks["safety_score"] > 0])
                    patterns["attack_patterns"]["compliance_rate"] = len(attacks[attacks["safety_score"] <= 0]) / len(attacks)
        
        return patterns


# Test function
def test_reporter():
    """Test the reporter with sample data."""
    print("🧪 Testing Enhanced Reporter...")
    
    # Create sample results data
    sample_data = {
        "Current User Question": ["What is X?", "How does Y work?", "Tell me about Z"],
        "Assistant Answer": ["X is...", "Y works by...", "Z is..."],
        "Fragment Texts": ["Info about X", "Info about Y", "Info about Z"],
        "Conversation History": ["", "Previous context", ""],
        "core_passed": [True, True, False],
        "relevance_pass": [True, True, False],
        "grounding_pass": [True, True, False],
        "clarity_score": [2, 1, None],
        "tone_score": [2, 2, None],
        "safety_score": [0, 0, -1],
        "primary_composite_score": [2.0, 1.5, -1.0],
        "primary_category": ["✅ Good", "📊 Acceptable", "⚠️ Unsafe"],
        "traditional_composite_score": [1.8, 1.4, 0.5],
        "answer_length": [25, 45, 15],
        "is_attack": [False, False, True],
        "evaluation_time": [2.1, 2.3, 1.9]
    }
    
    results_df = pd.DataFrame(sample_data)
    results_df.attrs["evaluation_metadata"] = {
        "scoring_mode": "dual",
        "total_time": 6.3,
        "avg_time_per_answer": 2.1
    }
    
    # Test reporter
    reporter = Reporter(output_dir="test_reports")
    paths = reporter.generate_report(results_df)
    
    print("\n✅ Test completed! Generated files:")
    for file_type, path in paths.items():
        print(f"   - {file_type}: {path}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_reports", ignore_errors=True)


if __name__ == "__main__":
    test_reporter()