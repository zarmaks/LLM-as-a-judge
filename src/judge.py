"""
judge.py - Enhanced RAG Judge with Dual Scoring System

This enhanced judge implements:
1. Primary scoring: Binary core + Scaled quality + Safety
2. Traditional scoring: 7-dimension analysis
3. Attack pattern detection
4. Response pattern analysis
5. Comprehensive statistics
"""

import pandas as pd
import numpy as np
from typing import Dict
from tqdm import tqdm
import time
from datetime import datetime
import json

from dimensions import (
    BINARY_DIMENSIONS,
    SCALED_DIMENSIONS,
    TRADITIONAL_DIMENSIONS,
    calculate_primary_composite_score,
    calculate_traditional_composite_score,
    categorize_primary_score,
    categorize_traditional_score
)

from simple_llm_client import SimpleLLMClient


class RAGJudge:
    """
    Enhanced RAG Judge supporting dual scoring system.
    """
    
    def __init__(self, scoring_mode: str = "dual", temperature: float = 0.0):
        """
        Initialize the judge.
        
        Args:
            scoring_mode: "dual", "primary", or "traditional"
            temperature: LLM temperature for evaluations
        """
        self.scoring_mode = scoring_mode
        self.llm_client = SimpleLLMClient(temperature=temperature)
        
        # Dimensions based on mode
        self.binary_dimensions = BINARY_DIMENSIONS
        self.scaled_dimensions = SCALED_DIMENSIONS
        self.traditional_dimensions = TRADITIONAL_DIMENSIONS
        
        # Statistics tracking
        self.evaluation_stats = {
            "total_evaluated": 0,
            "evaluation_times": [],
            "failures": [],
            "attack_patterns": {}
        }
        
        print("✅ RAG Judge initialized")
        print(f"📊 Scoring mode: {scoring_mode}")
        print(f"🌡️  Temperature: {temperature}")
    
    
    def evaluate_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Evaluate all answers in the CSV using the configured scoring mode.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame with original data + evaluation scores
        """
        print(f"\n📂 Loading dataset from: {csv_path}")
        
        # Load and validate CSV
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} answers to evaluate")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
        
        # Validate required columns
        required_cols = [
            "Current User Question",
            "Assistant Answer", 
            "Fragment Texts",
            "Conversation History"
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Validate data
        if len(df) == 0:
            raise ValueError("CSV file is empty")
            
        # Check for essential data
        for col in required_cols[:3]:  # Don't check Conversation History
            if df[col].isna().all():
                raise ValueError(f"Column '{col}' contains no valid data")
        
        # Process each answer
        all_results = []
        print("\n🔄 Starting evaluation...")
        
        # Estimate time based on scoring mode
        dims_per_answer = self._estimate_dimensions_per_answer()
        estimated_time = len(df) * dims_per_answer * 2.5  # 2.5 seconds per dimension
        print(f"⏱️  Estimated time: ~{int(estimated_time/60)} minutes")
        
        start_time = time.time()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
                # Evaluate single answer
                result = self._evaluate_single_answer(row, idx + 1)
                
                # Combine with original data
                full_result = row.to_dict()
                full_result.update(result)
                all_results.append(full_result)
                
                # Rate limiting
                time.sleep(2)  # Adjust based on API limits
                
            except Exception as e:
                print(f"\n⚠️  Error on row {idx + 1}: {e}")
                self.evaluation_stats["failures"].append({
                    "row": idx + 1,
                    "error": str(e)
                })
                
                # Add minimal result to maintain DataFrame structure
                full_result = row.to_dict()
                full_result.update(self._get_failure_result())
                all_results.append(full_result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Add metadata
        total_time = time.time() - start_time
        results_df.attrs["evaluation_metadata"] = {
            "scoring_mode": self.scoring_mode,
            "total_time": total_time,
            "avg_time_per_answer": total_time / len(df),
            "timestamp": datetime.now().isoformat(),
            "failures": len(self.evaluation_stats["failures"])
        }
        
        print(f"\n✅ Evaluation complete in {int(total_time/60)}m {int(total_time%60)}s")
        self._print_summary_stats(results_df)
        
        return results_df
    
    
    def _evaluate_single_answer(self, row: pd.Series, row_num: int) -> Dict:
        """
        Evaluate a single answer using the configured scoring mode.
        
        Args:
            row: DataFrame row with answer data
            row_num: Row number (1-indexed)
            
        Returns:
            Dictionary with all evaluation results
        """
        start_time = time.time()
        results = {"row_number": row_num}
        
        # Extract data
        question = str(row["Current User Question"])
        answer = str(row["Assistant Answer"])
        fragments = str(row["Fragment Texts"])
        history = row["Conversation History"]
        
        # Handle null/empty history
        if pd.isna(history) or not str(history).strip():
            history = ""
            has_context = False
        else:
            history = str(history)
            has_context = True
        
        results["has_conversation_history"] = has_context
        
        # Detect attack patterns
        attack_type = self.llm_client.detect_attack_pattern(question)
        results["is_attack"] = attack_type is not None
        results["attack_type"] = attack_type
        
        if attack_type:
            self.evaluation_stats["attack_patterns"][attack_type] = \
                self.evaluation_stats["attack_patterns"].get(attack_type, 0) + 1
        
        # Answer characteristics
        results["answer_length"] = len(answer)
        results["question_type"] = self._classify_question_type(question)
        
        # Evaluate based on mode
        if self.scoring_mode in ["dual", "primary"]:
            primary_results = self._evaluate_primary_system(
                question, answer, fragments, history, has_context
            )
            results.update(primary_results)
        
        if self.scoring_mode in ["dual", "traditional"]:
            traditional_results = self._evaluate_traditional_system(
                question, answer, fragments, history, has_context
            )
            results.update(traditional_results)
        
        # Record timing
        eval_time = time.time() - start_time
        results["evaluation_time"] = round(eval_time, 2)
        self.evaluation_stats["evaluation_times"].append(eval_time)
        
        # Update total evaluated counter
        self.evaluation_stats["total_evaluated"] += 1
        
        return results
    
    
    def _evaluate_primary_system(self, question: str, answer: str, 
                                fragments: str, history: str,
                                has_context: bool) -> Dict:
        """
        Evaluate using primary (binary + scaled + safety) system.
        """
        results = {}
        
        # Step 1: Evaluate binary core criteria
        binary_scores = {}
        for dim_key, dimension in self.binary_dimensions.items():
            result = self.llm_client.evaluate_binary(
                dimension_name=dimension.name,
                criteria=dimension.evaluation_criteria,
                question=question,
                answer=answer,
                fragments=fragments
            )
            
            binary_scores[dim_key] = result["pass"]
            results[f"{dim_key}_pass"] = result["pass"]
            results[f"{dim_key}_reason"] = result["reason"]
        
        # Check if core passed
        core_passed = all(binary_scores.values())
        results["core_passed"] = core_passed
        
        # Step 2: Evaluate quality dimensions (only if core passed)
        scaled_scores = {}
        if core_passed:
            for dim_key, dimension in self.scaled_dimensions.items():
                # Skip context awareness if no history
                if dim_key == "context_awareness" and not has_context:
                    results[f"{dim_key}_score"] = None
                    results[f"{dim_key}_reasoning"] = "N/A - No conversation history"
                    continue
                
                result = self.llm_client.evaluate_scaled(
                    dimension_name=dimension.name,
                    criteria=dimension.evaluation_criteria,
                    question=question,
                    answer=answer,
                    fragments=fragments,
                    history=history
                )
                
                scaled_scores[dim_key] = result["score"]
                results[f"{dim_key}_score"] = result["score"]
                results[f"{dim_key}_reasoning"] = result["reasoning"]
        else:
            # Core failed - don't evaluate quality
            for dim_key in self.scaled_dimensions:
                results[f"{dim_key}_score"] = None
                results[f"{dim_key}_reasoning"] = "Not evaluated (core criteria failed)"
        
        # Step 3: Always evaluate safety
        safety_result = self.llm_client.evaluate_safety(answer, question)
        results["safety_score"] = safety_result["score"]
        results["safety_reason"] = safety_result["reason"]
        
        # Step 4: Calculate composite score
        composite_score, score_details = calculate_primary_composite_score(
            binary_scores=binary_scores,
            scaled_scores=scaled_scores,
            safety_score=safety_result["score"],
            has_context=has_context
        )
        
        results["primary_composite_score"] = composite_score
        results["primary_score_details"] = json.dumps(score_details)
        results["primary_category"] = categorize_primary_score(composite_score, score_details)
        
        return results
    
    
    def _evaluate_traditional_system(self, question: str, answer: str,
                                   fragments: str, history: str,
                                   has_context: bool) -> Dict:
        """
        Evaluate using traditional 7-dimension system.
        """
        results = {}
        dimension_scores = {}
        
        for dim_key, dimension in self.traditional_dimensions.items():
            # Skip context awareness if no history
            if dim_key == "context_awareness" and not has_context:
                results[f"trad_{dim_key}_score"] = None
                results[f"trad_{dim_key}_reasoning"] = "N/A - No conversation history"
                continue
            
            # Handle safety specially in traditional system
            if dim_key == "safety":
                # Convert safety score from -1,+1 to 0-2 for traditional
                safety_result = self.llm_client.evaluate_safety(answer, question)
                raw_score = safety_result["score"]
                # Map: -1→0, 0→1, 1→2
                if raw_score == -1:
                    trad_score = 0
                elif raw_score == 0:
                    trad_score = 1
                else:  # raw_score == 1
                    trad_score = 2
                    
                dimension_scores[dim_key] = trad_score
                results[f"trad_{dim_key}_score"] = trad_score
                results[f"trad_{dim_key}_reasoning"] = safety_result["reason"]
            else:
                # Normal evaluation
                result = self.llm_client.evaluate_scaled(
                    dimension_name=dimension.name,
                    criteria=dimension.evaluation_criteria,
                    question=question,
                    answer=answer,
                    fragments=fragments,
                    history=history
                )
                
                dimension_scores[dim_key] = result["score"]
                results[f"trad_{dim_key}_score"] = result["score"]
                results[f"trad_{dim_key}_reasoning"] = result["reasoning"]
        
        # Calculate traditional composite
        trad_composite = calculate_traditional_composite_score(
            dimension_scores, has_context
        )
        
        results["traditional_composite_score"] = trad_composite
        results["traditional_category"] = categorize_traditional_score(trad_composite)
        
        return results
    
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question."""
        q_lower = question.lower()
        
        if any(word in q_lower for word in ["what", "which", "who", "where", "when"]):
            return "factual"
        elif any(word in q_lower for word in ["how", "why"]):
            return "explanatory" 
        elif any(word in q_lower for word in ["compute", "calculate", "solve"]):
            return "computational"
        elif any(word in q_lower for word in ["write", "create", "generate"]):
            return "creative"
        elif self.llm_client.detect_attack_pattern(question):
            return "adversarial"
        else:
            return "other"
    
    
    def _estimate_dimensions_per_answer(self) -> int:
        """Estimate number of dimensions to evaluate per answer."""
        if self.scoring_mode == "dual":
            return 10  # 3 binary + 4 scaled + 1 safety + 7 traditional - 5 overlap
        elif self.scoring_mode == "primary":
            return 8   # 3 binary + 4 scaled + 1 safety
        else:  # traditional
            return 7
    
    
    def _get_failure_result(self) -> Dict:
        """Get default result for failed evaluations."""
        result = {
            "evaluation_failed": True,
            "primary_composite_score": 0,
            "traditional_composite_score": 0,
            "primary_category": "❌ Evaluation Failed",
            "traditional_category": "Failed"
        }
        
        # Add None for all expected fields
        for dim in self.binary_dimensions:
            result[f"{dim}_pass"] = None
            result[f"{dim}_reason"] = "Evaluation failed"
            
        for dim in self.scaled_dimensions:
            result[f"{dim}_score"] = None
            result[f"{dim}_reasoning"] = "Evaluation failed"
            
        for dim in self.traditional_dimensions:
            result[f"trad_{dim}_score"] = None
            result[f"trad_{dim}_reasoning"] = "Evaluation failed"
            
        return result
    
    
    def _print_summary_stats(self, results_df: pd.DataFrame):
        """Print comprehensive summary statistics."""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        # Overall stats
        print("\n📈 Dataset Overview:")
        print(f"   Total answers: {len(results_df)}")
        print(f"   Evaluation failures: {len(self.evaluation_stats['failures'])}")
        
        if self.evaluation_stats["failures"]:
            print(f"   Failed rows: {[f['row'] for f in self.evaluation_stats['failures']]}")
        
        # Attack patterns
        if self.evaluation_stats["attack_patterns"]:
            print("\n🔍 Attack Patterns Detected:")
            for attack_type, count in self.evaluation_stats["attack_patterns"].items():
                print(f"   {attack_type}: {count} occurrences")
        
        # Primary system results (if applicable)
        if self.scoring_mode in ["dual", "primary"]:
            print("\n🎯 Primary System Results:")
            
            # Core pass rate
            if "core_passed" in results_df.columns:
                core_pass_rate = results_df["core_passed"].sum() / len(results_df) * 100
                print(f"   Core criteria pass rate: {core_pass_rate:.1f}%")
                
                # Individual binary dimensions
                for dim in self.binary_dimensions:
                    if f"{dim}_pass" in results_df.columns:
                        pass_rate = results_df[f"{dim}_pass"].sum() / len(results_df) * 100
                        print(f"   - {dim.capitalize()}: {pass_rate:.1f}%")
            
            # Quality scores (for passing answers)
            passing_df = results_df[results_df["core_passed"]] if "core_passed" in results_df.columns else results_df
            
            if len(passing_df) > 0:
                print(f"\n   Quality scores (n={len(passing_df)} passing):")
                for dim in self.scaled_dimensions:
                    col = f"{dim}_score"
                    if col in passing_df.columns:
                        valid_scores = passing_df[col].dropna()
                        if len(valid_scores) > 0:
                            print(f"   - {dim.replace('_', ' ').title()}: {valid_scores.mean():.2f}/2")
            
            # Safety distribution
            if "safety_score" in results_df.columns:
                print("\n   Safety distribution:")
                safety_counts = results_df["safety_score"].value_counts().sort_index()
                for score, count in safety_counts.items():
                    pct = count / len(results_df) * 100
                    print(f"   - Score {score:+d}: {count} ({pct:.0f}%)")
            
            # Final scores
            if "primary_composite_score" in results_df.columns:
                scores = results_df["primary_composite_score"]
                print("\n   Composite scores:")
                print(f"   - Mean: {scores.mean():.2f}")
                print(f"   - Std: {scores.std():.2f}")
                print(f"   - Min: {scores.min():.1f}")
                print(f"   - Max: {scores.max():.1f}")
        
        # Traditional system results (if applicable)  
        if self.scoring_mode in ["dual", "traditional"]:
            print("\n📊 Traditional System Results:")
            
            for dim in self.traditional_dimensions:
                col = f"trad_{dim}_score"
                if col in results_df.columns:
                    valid_scores = results_df[col].dropna()
                    if len(valid_scores) > 0:
                        print(f"   {dim.replace('_', ' ').title()}: {valid_scores.mean():.2f}/2")
            
            if "traditional_composite_score" in results_df.columns:
                trad_scores = results_df["traditional_composite_score"]
                print(f"\n   Traditional composite: {trad_scores.mean():.2f}/2")
        
        # Timing stats
        if self.evaluation_stats["evaluation_times"]:
            times = self.evaluation_stats["evaluation_times"]
            print("\n⏱️  Timing Statistics:")
            print(f"   Average per answer: {np.mean(times):.1f}s")
            print(f"   Total time: {sum(times)/60:.1f} minutes")
        
        # Categories distribution
        if "primary_category" in results_df.columns:
            print("\n📊 Answer Categories:")
            categories = results_df["primary_category"].value_counts()
            for cat, count in categories.items():
                print(f"   {cat}: {count} ({count/len(results_df)*100:.0f}%)")
        
        print("\n" + "="*60)


# Test function
def test_judge():
    """Test the judge with a small sample."""
    print("🧪 Testing Enhanced RAG Judge...")
    
    # Create test data
    test_data = pd.DataFrame([
        {
            "Current User Question": "What is the capital of France?",
            "Assistant Answer": "The capital of France is Paris.",
            "Fragment Texts": "France is a European country. Its capital is Paris.",
            "Conversation History": ""
        },
        {
            "Current User Question": "What is the capital of Japan?", 
            "Assistant Answer": "The capital of Japan is Kyoto.",
            "Fragment Texts": "Japan is an island nation. Tokyo is the capital of Japan.",
            "Conversation History": ""
        }
    ])
    
    # Save test CSV
    test_csv = "test_judge_data.csv"
    test_data.to_csv(test_csv, index=False)
    
    try:
        # Test dual mode
        judge = RAGJudge(scoring_mode="dual")
        results = judge.evaluate_dataset(test_csv)
        
        print("\n📋 Test Results:")
        print(results[["Current User Question", "primary_composite_score", 
                      "traditional_composite_score", "primary_category"]])
        
        print("\n✅ Judge test completed!")
        
    finally:
        # Cleanup
        import os
        if os.path.exists(test_csv):
            os.remove(test_csv)


if __name__ == "__main__":
    test_judge()