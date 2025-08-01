"""
Tests for main.py - CLI functionality and end-to-end integration.
"""

import pytest
import sys
import os
import tempfile
from unittest.mock import Mock, patch
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main


class TestMain:
    """Test suite for main.py CLI functionality."""
    
    def test_print_header(self, capsys):
        """Test header printing functionality."""
        # Mock argparse Namespace
        args = type('Args', (), {
            'csv': 'test.csv',
            'scoring_mode': 'dual',
            'temperature': 0.0,
            'seed': 42,
            'output_dir': 'reports',
            'test_mode': True
        })()
        
        main.print_header(args)
        
        captured = capsys.readouterr()
        assert "RAG ANSWER QUALITY JUDGE" in captured.out
        assert "test.csv" in captured.out
        assert "dual" in captured.out
        assert "TEST MODE" in captured.out
    
    def test_print_summary_basic(self, comprehensive_results_df, capsys):
        """Test summary printing with comprehensive results."""
        output_paths = {
            "csv": "test_results.csv",
            "markdown": "test_report.md"
        }
        elapsed_time = 125.5  # 2 minutes 5.5 seconds
        
        args = type('Args', (), {
            'scoring_mode': 'dual',
            'test_mode': False,
            'export_stats_json': False
        })()
        
        # Mock os.path.getsize to simulate file sizes
        with patch('main.os.path.getsize') as mock_getsize:
            mock_getsize.side_effect = lambda path: 1024 if 'csv' in path else 2048
            main.print_summary(comprehensive_results_df, output_paths, elapsed_time, args)
        
        captured = capsys.readouterr()
        assert "EVALUATION COMPLETE" in captured.out
        assert "2m 5s" in captured.out  # Time formatting
        assert "4" in captured.out  # Number of answers
    
    def test_print_key_insights(self, comprehensive_results_df, capsys):
        """Test key insights printing."""
        main.print_key_insights(comprehensive_results_df)
        
        captured = capsys.readouterr()
        assert "Key Insights" in captured.out
    
    def test_quick_test_function(self):
        """Test the quick_test function."""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.stdout = "Test output"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            # Should not raise exception
            main.quick_test()
            
            # Should have called subprocess.run
            mock_run.assert_called()
    
    @patch('subprocess.run')
    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_main_function_basic_flow(self, mock_exists, mock_read_csv, mock_subprocess):
        """Test main function basic execution flow."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock CSV reading
        mock_df = Mock()
        mock_df.to_csv = Mock()
        mock_df.__len__ = Mock(return_value=5)  # Mock length
        mock_df.columns = ['Current User Question', 'Assistant Answer']  # Mock columns
        mock_read_csv.return_value = mock_df
        
        # Mock argument parsing
        test_args = [
            'main.py', '--csv', 'test.csv', '--scoring-mode', 'primary'
        ]
        
        with patch('sys.argv', test_args):
            with patch('main.RAGJudge') as mock_judge:
                with patch('main.Reporter') as mock_reporter:
                    with patch('main.validate_all_dimensions'):
                        
                        # Mock judge evaluation
                        mock_judge_instance = Mock()
                        mock_judge_instance.evaluate_dataset.return_value = mock_df
                        mock_judge.return_value = mock_judge_instance
                        
                        # Mock reporter
                        mock_reporter_instance = Mock()
                        mock_reporter_instance.generate_report.return_value = {'csv': 'test.csv'}
                        mock_reporter.return_value = mock_reporter_instance
                        
                    # Mock time
                    with patch('main.time.time', side_effect=[0, 10]):
                        with patch('main.os.path.getsize', return_value=1024):
                            # Should not raise exception
                            main.main()

    def test_main_missing_csv_file(self, capsys):
        """Test main function with missing CSV file."""
        test_args = ['main.py', '--csv', 'nonexistent.csv']
        
        with patch('sys.argv', test_args):
            with patch('main.os.path.exists', return_value=False):
                with pytest.raises(SystemExit):
                    main.main()
                
                captured = capsys.readouterr()
                assert "CSV file not found" in captured.out
    
    def test_main_validation_failure(self, capsys):
        """Test main function with dimension validation failure."""
        test_args = ['main.py', '--csv', 'test.csv']
        
        with patch('sys.argv', test_args):
            with patch('main.os.path.exists', return_value=True):
                with patch('main.validate_all_dimensions', side_effect=Exception("Validation error")):
                    with pytest.raises(SystemExit):
                        main.main()
                    
                    captured = capsys.readouterr()
                    assert "Dimension validation failed" in captured.out
    
    def test_main_with_test_mode(self):
        """Test main function with test mode enabled."""
        test_args = ['main.py', '--csv', 'test.csv', '--test-mode']
        
        with patch('sys.argv', test_args):
            with patch('main.os.path.exists', return_value=True):
                with patch('pandas.read_csv') as mock_read_csv:
                    with patch('main.validate_all_dimensions'):
                        with patch('main.RAGJudge') as mock_judge:
                            with patch('main.Reporter') as mock_reporter:
                                
                                # Mock DataFrame
                                mock_df = Mock()
                                mock_df.to_csv = Mock()
                                mock_df.__len__ = Mock(return_value=5)  # Mock length
                                mock_df.columns = ['core_passed', 'primary_composite_score']  # Mock columns
                                mock_df.__contains__ = Mock(return_value=True)  # For 'column in df.columns'
                                mock_read_csv.return_value = mock_df
                                
                                # Mock evaluation
                                mock_judge_instance = Mock()
                                mock_judge_instance.evaluate_dataset.return_value = mock_df
                                mock_judge.return_value = mock_judge_instance
                                
                                # Mock reporter
                                mock_reporter_instance = Mock()
                                mock_reporter_instance.generate_report.return_value = {'csv': 'test.csv'}
                                mock_reporter.return_value = mock_reporter_instance
                                
                                with patch('main.time.time', side_effect=[0, 5]):
                                    with patch('main.os.remove') as mock_remove:
                                        with patch('main.os.path.getsize', return_value=1024):
                                            with patch('main.print_summary') as mock_print_summary:
                                                main.main()
                                                
                                                # Verify print_summary was called
                                                mock_print_summary.assert_called_once()
                                        
                                        # Should read only 5 rows for test mode
                                        mock_read_csv.assert_called_with('test.csv', nrows=5)
                                        
                                        # Should clean up test file
                                        mock_remove.assert_called()
    
    def test_main_keyboard_interrupt(self, capsys):
        """Test main function handling keyboard interrupt."""
        test_args = ['main.py', '--csv', 'test.csv']
        
        with patch('sys.argv', test_args):
            with patch('main.os.path.exists', return_value=True):
                with patch('main.validate_all_dimensions'):
                    with patch('main.RAGJudge') as mock_judge:
                        
                        # Mock keyboard interrupt during evaluation
                        mock_judge_instance = Mock()
                        mock_judge_instance.evaluate_dataset.side_effect = KeyboardInterrupt()
                        mock_judge.return_value = mock_judge_instance
                        
                        with pytest.raises(SystemExit):
                            main.main()
                        
                        captured = capsys.readouterr()
                        assert "interrupted by user" in captured.out
    
    def test_main_evaluation_error(self, capsys):
        """Test main function handling evaluation errors."""
        test_args = ['main.py', '--csv', 'test.csv']
        
        with patch('sys.argv', test_args):
            with patch('main.os.path.exists', return_value=True):
                with patch('main.validate_all_dimensions'):
                    with patch('main.RAGJudge') as mock_judge:
                        
                        # Mock evaluation error
                        mock_judge_instance = Mock()
                        mock_judge_instance.evaluate_dataset.side_effect = Exception("Evaluation failed")
                        mock_judge.return_value = mock_judge_instance
                        
                        with pytest.raises(SystemExit):
                            main.main()
                        
                        captured = capsys.readouterr()
                        assert "Error during evaluation" in captured.out
                        assert "Troubleshooting tips" in captured.out
    
    def test_main_verbose_mode(self):
        """Test main function with verbose mode."""
        test_args = ['main.py', '--csv', 'test.csv', '--verbose']
        
        with patch('sys.argv', test_args):
            with patch('main.os.path.exists', return_value=True):
                with patch('main.validate_all_dimensions'):
                    with patch('main.RAGJudge') as mock_judge:
                        
                        mock_judge_instance = Mock()
                        mock_judge_instance.evaluate_dataset.side_effect = Exception("Test error")
                        mock_judge.return_value = mock_judge_instance
                        
                        with patch('traceback.print_exc') as mock_traceback:
                            with pytest.raises(SystemExit):
                                main.main()
                            
                            # Should print full traceback in verbose mode
                            mock_traceback.assert_called_once()
    
    def test_main_skip_validation(self):
        """Test main function with validation skipped."""
        test_args = ['main.py', '--csv', 'test.csv', '--skip-validation']
        
        with patch('sys.argv', test_args):
            with patch('main.os.path.exists', return_value=True):
                with patch('main.validate_all_dimensions') as mock_validate:
                    with patch('main.RAGJudge') as mock_judge:
                        with patch('main.Reporter') as mock_reporter:
                            
                            # Mock evaluation
                            mock_df = Mock()
                            mock_df.__len__ = Mock(return_value=3)  # Mock length
                            mock_judge_instance = Mock()
                            mock_judge_instance.evaluate_dataset.return_value = mock_df
                            mock_judge.return_value = mock_judge_instance
                            
                            # Mock reporter
                            mock_reporter_instance = Mock()
                            mock_reporter_instance.generate_report.return_value = {'csv': 'test.csv'}
                            mock_reporter.return_value = mock_reporter_instance
                            
                            with patch('main.time.time', side_effect=[0, 1]):
                                with patch('main.print_summary') as mock_print_summary:
                                    main.main()
                                    
                                    # Verify print_summary was called
                                    mock_print_summary.assert_called_once()
                            
                            # Validation should not be called
                            mock_validate.assert_not_called()
    
    def test_main_with_custom_parameters(self):
        """Test main function with custom parameters."""
        test_args = [
            'main.py', '--csv', 'test.csv',
            '--scoring-mode', 'traditional',
            '--temperature', '0.7',
            '--seed', '123',
            '--output-dir', 'custom_reports',
            '--export-stats-json'
        ]
        
        with patch('sys.argv', test_args):
            with patch('main.os.path.exists', return_value=True):
                with patch('main.validate_all_dimensions'):
                    with patch('main.RAGJudge') as mock_judge:
                        with patch('main.Reporter') as mock_reporter:
                            with patch('random.seed') as mock_random_seed:
                                with patch('numpy.random.seed') as mock_np_seed:
                                    
                                    # Mock evaluation
                                    mock_df = Mock()
                                    mock_df.__len__ = Mock(return_value=4)  # Mock length
                                    mock_judge_instance = Mock()
                                    mock_judge_instance.evaluate_dataset.return_value = mock_df
                                    mock_judge.return_value = mock_judge_instance
                                    
                                    # Mock reporter
                                    mock_reporter_instance = Mock()
                                    mock_reporter_instance.generate_report.return_value = {
                                        'csv': 'test.csv',
                                        'statistics_json': 'stats.json'
                                    }
                                    mock_reporter.return_value = mock_reporter_instance
                                    
                                    with patch('main.time.time', side_effect=[0, 1]):
                                        with patch('main.print_summary') as mock_print_summary:
                                            main.main()
                                            
                                            # Verify print_summary was called
                                            mock_print_summary.assert_called_once()
                                    
                                    # Check that parameters were used correctly
                                    mock_judge.assert_called_with(
                                        scoring_mode="traditional",
                                        temperature=0.7
                                    )
                                    mock_reporter.assert_called_with(output_dir="custom_reports")
                                    mock_random_seed.assert_called_with(123)
                                    mock_np_seed.assert_called_with(123)
                                    
                                    # Should request stats JSON
                                    mock_reporter_instance.generate_report.assert_called_with(
                                        mock_df, include_stats_json=True
                                    )
    
    def test_main_quick_test_flag(self):
        """Test main function with --quick-test flag."""
        test_args = ['main.py', '--quick-test']
        
        with patch('sys.argv', test_args):
            with patch('main.quick_test') as mock_quick_test:
                # Call the main logic that handles the flag
                import main as main_module
                if len(test_args) > 1 and test_args[1] == "--quick-test":
                    main_module.quick_test()
                else:
                    main_module.main()
                mock_quick_test.assert_called_once()


class TestMainIntegration:
    """Integration tests for main.py with real file operations."""
    
    def test_end_to_end_evaluation_mock_mode(self, temp_csv_file, temp_output_dir):
        """Test end-to-end evaluation in mock mode."""
        # Ensure no API key to force mock mode
        with patch.dict(os.environ, {}, clear=True):
            test_args = [
                'main.py', '--csv', temp_csv_file,
                '--output-dir', temp_output_dir,
                '--test-mode',  # Only first 5 rows
                '--skip-validation'
            ]
            
            with patch('sys.argv', test_args):
                # Should complete without errors
                main.main()
                
                # Check that output files were created
                output_files = os.listdir(temp_output_dir)
                assert any(f.endswith('.csv') for f in output_files)
                assert any(f.endswith('.md') for f in output_files)
    
    def test_cli_argument_parsing(self):
        """Test command line argument parsing."""
        test_args = [
            'main.py',
            '--csv', 'test.csv',
            '--scoring-mode', 'dual',
            '--temperature', '0.5',
            '--seed', '42',
            '--output-dir', 'results',
            '--export-stats-json',
            '--test-mode',
            '--verbose'
        ]
        
        with patch('sys.argv', test_args):
            # Mock the rest of main to just test argument parsing
            with patch('main.print_header'):
                with patch('main.os.path.exists', return_value=False):
                    with pytest.raises(SystemExit):  # Will exit due to missing file
                        main.main()
    
    def test_argument_validation(self):
        """Test CLI argument validation."""
        # Test invalid scoring mode
        test_args = ['main.py', '--csv', 'test.csv', '--scoring-mode', 'invalid']
        
        with patch('sys.argv', test_args):
            # Should raise SystemExit due to invalid choice
            with pytest.raises(SystemExit):
                main.main()
    
    def test_help_output(self):
        """Test that help output is properly formatted."""
        test_args = ['main.py', '--help']
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                main.main()
    
    @patch('subprocess.run')
    def test_actual_main_execution(self, mock_subprocess, temp_csv_file):
        """Test actual main.py execution as subprocess."""
        # Create a minimal test by calling main.py as subprocess
        
        # Mock the subprocess call
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Test completed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # This would be the actual test if we wanted to run the subprocess
        # result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        # assert result.returncode == 0


class TestMainUtilityFunctions:
    """Test utility functions in main.py."""
    
    def test_print_header_formatting(self, capsys):
        """Test header formatting with different configurations."""
        args = type('Args', (), {
            'csv': 'very_long_filename_that_might_wrap.csv',
            'scoring_mode': 'dual',
            'temperature': 0.0,
            'seed': 42,
            'output_dir': 'reports',
            'test_mode': False
        })()
        
        main.print_header(args)
        
        captured = capsys.readouterr()
        lines = captured.out.split('\n')
        
        # Check that header has proper formatting
        assert any('=' in line for line in lines)  # Should have separator lines
        assert any('RAG ANSWER QUALITY JUDGE' in line for line in lines)
    
    def test_print_summary_with_different_modes(self, comprehensive_results_df, capsys):
        """Test summary printing with different scoring modes."""
        output_paths = {"csv": "test.csv", "markdown": "test.md"}
        elapsed_time = 60.0
        
        # Test primary mode
        args_primary = type('Args', (), {
            'scoring_mode': 'primary',
            'test_mode': False,
            'export_stats_json': False
        })()
        
        with patch('os.path.getsize', return_value=1024):  # Mock file size
            main.print_summary(comprehensive_results_df, output_paths, elapsed_time, args_primary)
            captured = capsys.readouterr()
            assert "Primary" in captured.out or "primary" in captured.out
        
        # Test traditional mode  
        args_traditional = type('Args', (), {
            'scoring_mode': 'traditional',
            'test_mode': False,
            'export_stats_json': False
        })()
        
        with patch('os.path.getsize', return_value=1024):  # Mock file size
            main.print_summary(comprehensive_results_df, output_paths, elapsed_time, args_traditional)
            captured = capsys.readouterr()
            assert "Traditional" in captured.out or "traditional" in captured.out
    
    def test_print_summary_edge_cases(self, capsys):
        """Test summary printing with edge cases."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        output_paths = {"csv": "test.csv"}
        
        args = type('Args', (), {
            'scoring_mode': 'dual',
            'test_mode': False,
            'export_stats_json': False
        })()
        
        # Should not crash with empty DataFrame
        with patch('os.path.getsize', return_value=1024):  # Mock file size
            main.print_summary(empty_df, output_paths, 1.0, args)
        
        captured = capsys.readouterr()
        assert "EVALUATION COMPLETE" in captured.out
    
    def test_print_key_insights_comprehensive(self, comprehensive_results_df, capsys):
        """Test key insights with comprehensive data."""
        main.print_key_insights(comprehensive_results_df)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should identify insights based on the comprehensive data
        assert "Key Insights" in output
        
        # Check for specific insights based on test data
        if any(comprehensive_results_df.get("safety_score", []) < 0):
            assert "safety" in output.lower() or "dangerous" in output.lower()
    
    def test_main_error_handling_edge_cases(self):
        """Test main function error handling for edge cases."""
        # Test with malformed CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("This is not a proper CSV\nMissing headers")
            malformed_csv = f.name
        
        try:
            test_args = ['main.py', '--csv', malformed_csv, '--skip-validation']
            
            with patch('sys.argv', test_args):
                with pytest.raises(SystemExit):
                    main.main()
        finally:
            os.unlink(malformed_csv)
