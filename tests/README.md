# RAG Judge Test Suite

This directory contains comprehensive tests for the RAG Judge evaluation system.

## Test Structure

### Test Files

- **`test_simple_llm_client.py`** - Tests for the LLM client functionality
  - API integration (mock and real modes)
  - Attack pattern detection
  - Binary, scaled, and safety evaluation methods
  - JSON parsing and error handling
  - Rate limiting and retries

- **`test_dimensions.py`** - Tests for scoring dimensions and calculations
  - Dimension configuration validation
  - Primary and traditional scoring algorithms
  - Score categorization and composite calculations
  - Edge cases and boundary conditions

- **`test_judge.py`** - Tests for the main RAGJudge orchestrator
  - Single answer evaluation in different modes
  - Dataset processing and error handling
  - Attack detection and context awareness
  - Statistics tracking and metadata

- **`test_reporter.py`** - Tests for report generation
  - Markdown, CSV, and JSON report generation
  - Statistical analysis and insights
  - Failure analysis and recommendations
  - Report completeness and formatting

- **`test_main.py`** - Tests for CLI functionality and main entry point
  - Command line argument parsing
  - End-to-end execution flow
  - Error handling and user feedback
  - Different execution modes

- **`test_integration.py`** - End-to-end integration tests
  - Complete evaluation pipeline
  - Component interaction and data flow
  - Performance and robustness testing
  - Real-world scenario simulation

### Test Fixtures

The `conftest.py` file provides shared test fixtures:

- **`sample_data`** - Basic test data with various question types
- **`sample_dataframe`** - Pre-configured DataFrame with metadata
- **`temp_csv_file`** - Temporary CSV file for testing
- **`temp_output_dir`** - Temporary directory for output files
- **`mock_evaluation_results`** - Mock evaluation results
- **`comprehensive_results_df`** - Complete results with all scoring modes
- **`mock_llm_responses`** - Mock LLM response patterns

## Running Tests

### Quick Start

```bash
# Install test dependencies and run all tests
python run_tests.py --install

# Run quick test subset for development
python run_tests.py --quick

# Check test environment
python run_tests.py --check
```

### Test Categories

```bash
# Test specific components
python run_tests.py --category llm          # LLM client tests
python run_tests.py --category dimensions   # Scoring calculations
python run_tests.py --category judge        # Main evaluation logic
python run_tests.py --category reporter     # Report generation
python run_tests.py --category main         # CLI functionality
python run_tests.py --category integration  # End-to-end tests
```

### Advanced Options

```bash
# Run with coverage reporting (default)
python run_tests.py

# Run without coverage (faster)
python run_tests.py --no-coverage

# Run tests in parallel (faster for large test suites)
python run_tests.py --parallel

# Run specific test file
python run_tests.py tests/test_dimensions.py

# Run specific test method
python run_tests.py tests/test_judge.py::TestRAGJudge::test_evaluate_single_answer_primary_mode
```

### Using pytest directly

```bash
# Install dependencies first
pip install -r tests/requirements.txt

# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_dimensions.py -v
pytest tests/test_integration.py -v

# Run with specific markers (if defined)
pytest tests/ -m "not integration"  # Skip integration tests
```

## Test Coverage

The test suite aims for comprehensive coverage:

- **Unit Tests**: Individual functions and methods
- **Integration Tests**: Component interactions
- **End-to-End Tests**: Complete workflows
- **Edge Cases**: Error conditions and boundary values
- **Performance Tests**: Resource usage and timing
- **Robustness Tests**: Error handling and recovery

### Coverage Goals

- **SimpleLLMClient**: >95% (core functionality)
- **Dimensions**: 100% (scoring calculations)
- **RAGJudge**: >90% (main evaluation logic)
- **Reporter**: >85% (report generation)
- **Main CLI**: >80% (user interface)

## Test Data

### Mock Mode Testing

Most tests run in "mock mode" where no actual API calls are made:
- Faster execution
- No API key required
- Deterministic results
- Safe for CI/CD

### Real API Testing

Some tests can optionally use real API calls:
- Set `MISTRAL_API_KEY` environment variable
- Use `--real-api` flag (if implemented)
- Useful for integration validation

### Test Data Patterns

- **Normal Cases**: Typical questions and answers
- **Edge Cases**: Empty inputs, very long text, unicode
- **Attack Patterns**: Prompt injections, jailbreaks
- **Error Conditions**: Malformed data, network failures
- **Performance Cases**: Large datasets, concurrent access

## Debugging Tests

### Viewing Test Output

```bash
# Verbose output
python run_tests.py --category judge -v

# Show print statements
pytest tests/test_judge.py -s

# Show full traceback
pytest tests/test_judge.py --tb=long
```

### Running Individual Tests

```bash
# Run single test method
pytest tests/test_dimensions.py::TestDimensions::test_binary_dimensions_structure -v

# Run with debugger on failure
pytest tests/test_judge.py --pdb

# Run and drop into debugger
pytest tests/test_judge.py --pdb-trace
```

### Common Issues

1. **Import Errors**: Ensure `src/` directory is in Python path
2. **API Key Warnings**: Normal in mock mode, can be ignored
3. **Temporary File Errors**: Tests clean up automatically
4. **Unicode Issues**: Tests include unicode test cases
5. **Timeout Errors**: Use mock mode for faster execution

## Continuous Integration

The test suite is designed for CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    python run_tests.py --install --no-coverage --parallel
```

### CI Considerations

- Tests run in mock mode (no API key needed)
- Parallel execution for speed
- Coverage reporting for quality metrics
- Artifact collection for test reports

## Contributing Tests

When adding new functionality:

1. **Add Unit Tests**: Test individual functions
2. **Add Integration Tests**: Test component interactions
3. **Add Edge Cases**: Test error conditions
4. **Update Fixtures**: Add new test data if needed
5. **Run Full Suite**: Ensure no regressions

### Test Naming Conventions

- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality>_<condition>`
- Fixtures: `<descriptive_name>` (no test_ prefix)

### Test Documentation

Each test should have:
- Clear docstring explaining what is tested
- Expected behavior description
- Edge cases covered
- Mock setup explanation (if complex)

## Performance Testing

The suite includes performance considerations:

- **Timing Tests**: Measure execution time
- **Memory Tests**: Check resource usage  
- **Load Tests**: Test with large datasets
- **Concurrent Tests**: Test parallel execution

Performance targets:
- Single evaluation: <3 seconds (mock mode)
- 100 evaluations: <30 seconds (mock mode)
- Memory usage: <500MB for 1000 evaluations
- Report generation: <10 seconds for any dataset size
