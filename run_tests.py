#!/usr/bin/env python3
"""
Script to run all tests and generate coverage report.
"""

import subprocess
import sys
import os

def run_tests():
    """Run pytest with coverage reporting."""
    print("Running MoC App Tests...")
    print("=" * 50)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    cmd = [
        sys.executable, "-m", "pytest",
        "test_app.py",
        "-v",
        "--cov=app",
        "--cov-report=term-missing",
        "--cov-report=html"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ All tests passed successfully!")
            print("Coverage report generated in htmlcov/index.html")
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed!")
            return False
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
