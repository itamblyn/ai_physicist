#!/usr/bin/env python3
"""
Test runner script for AI Physicist project.
Provides convenient commands for running different types of tests.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="AI Physicist Test Runner")
    parser.add_argument("--type", choices=["unit", "integration", "regression", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--format", action="store_true", help="Format code with black and isort")
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Add test directory based on type
    if args.type == "unit":
        pytest_cmd.append("tests/unit/")
    elif args.type == "integration":
        pytest_cmd.append("tests/integration/")
    elif args.type == "regression":
        pytest_cmd.append("tests/regression/")
    else:  # all
        pytest_cmd.append("tests/")
    
    # Add options
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    if args.coverage:
        pytest_cmd.extend([
            "--cov=01_generate_questions",
            "--cov=02_baseline", 
            "--cov=03_extraneous_info_dataset",
            "--cov=04_unsolvable",
            "--cov-report=html",
            "--cov-report=xml"
        ])
    
    # Run tests
    success = True
    
    if args.format:
        print("Formatting code...")
        format_cmd = ["black", "."]
        if not run_command(format_cmd, "Code formatting with black"):
            success = False
        
        format_cmd = ["isort", "."]
        if not run_command(format_cmd, "Import sorting with isort"):
            success = False
    
    if args.lint:
        print("Running linting checks...")
        lint_cmd = ["flake8", ".", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"]
        if not run_command(lint_cmd, "Flake8 linting"):
            success = False
        
        lint_cmd = ["black", "--check", "."]
        if not run_command(lint_cmd, "Black format check"):
            success = False
        
        lint_cmd = ["isort", "--check-only", "."]
        if not run_command(lint_cmd, "Import sort check"):
            success = False
    
    # Run pytest
    if not run_command(pytest_cmd, f"Running {args.type} tests"):
        success = False
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
