"""
Test runner script for RAG Judge test suite.
Run this to execute all tests with appropriate configuration.
"""

import subprocess
import sys
from pathlib import Path


def install_test_dependencies():
    """Install test dependencies."""
    print("📦 Installing test dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Test dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install test dependencies: {e}")
        return False


def run_tests(coverage=True, verbose=True, parallel=False, specific_test=None):
    """
    Run the test suite with various options.
    
    Args:
        coverage: Whether to run with coverage reporting
        verbose: Whether to run in verbose mode
        parallel: Whether to run tests in parallel
        specific_test: Specific test file or pattern to run
    """
    print("🧪 Running RAG Judge test suite...")
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Add specific test if provided
    if specific_test:
        cmd = [sys.executable, "-m", "pytest", specific_test]
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    if parallel:
        cmd.extend(["-n", "auto"])  # Use all available CPUs
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--durations=10",  # Show 10 slowest tests
        "-x"  # Stop on first failure
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=test_dir.parent)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            if coverage:
                print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run tests: {e}")
        return False


def run_specific_test_category(category):
    """Run specific category of tests."""
    test_files = {
        "llm": "test_simple_llm_client.py",
        "dimensions": "test_dimensions.py", 
        "judge": "test_judge.py",
        "reporter": "test_reporter.py",
        "main": "test_main.py",
        "integration": "test_integration.py"
    }
    
    if category not in test_files:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {', '.join(test_files.keys())}")
        return False
    
    test_file = Path(__file__).parent / test_files[category]
    return run_tests(specific_test=str(test_file))


def run_quick_tests():
    """Run a quick subset of tests for development."""
    print("⚡ Running quick test subset...")
    
    # Run only unit tests, skip integration tests
    quick_tests = [
        "test_dimensions.py",
        "test_simple_llm_client.py::TestSimpleLLMClient::test_client_initialization_with_api_key",
        "test_simple_llm_client.py::TestSimpleLLMClient::test_attack_pattern_detection",
        "test_judge.py::TestRAGJudge::test_judge_initialization_dual_mode",
        "test_reporter.py::TestReporter::test_reporter_initialization"
    ]
    
    for test in quick_tests:
        test_path = Path(__file__).parent / test
        if "::" not in test:
            test_path = str(test_path)
        else:
            test_path = str(Path(__file__).parent / test.split("::")[0]) + "::" + "::".join(test.split("::")[1:])
            
        print(f"\n🏃 Running {test}...")
        success = run_tests(coverage=False, verbose=False, specific_test=test_path)
        if not success:
            print(f"❌ Quick test failed: {test}")
            return False
    
    print("\n✅ All quick tests passed!")
    return True


def check_test_environment():
    """Check that the test environment is properly set up."""
    print("🔍 Checking test environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required for tests")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check that src directory exists
    src_dir = Path(__file__).parent / "src"
    if not src_dir.exists():
        print("❌ src/ directory not found")
        return False
    
    print("✅ src/ directory found")
    
    # Check that main modules exist
    required_modules = [
        "simple_llm_client.py",
        "judge.py", 
        "reporter.py",
        "dimensions.py"
    ]
    
    for module in required_modules:
        module_path = src_dir / module
        if not module_path.exists():
            print(f"❌ Required module not found: {module}")
            return False
    
    print("✅ All required modules found")
    
    # Try importing pytest
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} available")
    except ImportError:
        print("❌ pytest not installed")
        return False
    
    return True


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Judge Test Runner")
    parser.add_argument("--install", action="store_true", help="Install test dependencies")
    parser.add_argument("--quick", action="store_true", help="Run quick test subset")
    parser.add_argument("--category", choices=["llm", "dimensions", "judge", "reporter", "main", "integration"],
                       help="Run specific test category")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--check", action="store_true", help="Check test environment only")
    parser.add_argument("test_pattern", nargs="?", help="Specific test file or pattern to run")
    
    args = parser.parse_args()
    
    # Install dependencies if requested (before environment check)
    if args.install:
        if not install_test_dependencies():
            sys.exit(1)
        print()  # Add spacing
    
    # Check environment
    if not check_test_environment():
        print("\n❌ Test environment check failed!")
        sys.exit(1)
    
    if args.check:
        print("\n✅ Test environment is ready!")
        return
    
    # Run specific test category
    if args.category:
        success = run_specific_test_category(args.category)
        sys.exit(0 if success else 1)
    
    # Run quick tests
    if args.quick:
        success = run_quick_tests()
        sys.exit(0 if success else 1)
    
    # Run full test suite or specific pattern
    success = run_tests(
        coverage=not args.no_coverage,
        parallel=args.parallel,
        specific_test=args.test_pattern
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
