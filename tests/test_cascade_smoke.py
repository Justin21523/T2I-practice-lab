#!/usr/bin/env python3
"""
Smoke test for Cascade quickstart notebook
Can be run in CI with SMOKE_MODE=true
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path


def test_cascade_notebook_execution():
    """Test that the Cascade notebook can execute without errors"""

    # Set smoke mode for fast testing
    os.environ["SMOKE_MODE"] = "true"

    notebook_path = "notebooks/t2i/20_cascade/nb-cascade-quickstart.ipynb"

    if not Path(notebook_path).exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False

    try:
        # Execute notebook using nbconvert
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=300",
            "--output-dir",
            "/tmp",
            notebook_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ Cascade notebook executed successfully")
            return True
        else:
            print(f"❌ Notebook execution failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Notebook execution timed out (>5min)")
        return False
    except Exception as e:
        print(f"❌ Error executing notebook: {e}")
        return False


def test_cascade_config_validation():
    """Test that generated config files are valid"""

    config_path = "cascade_config.json"

    if not Path(config_path).exists():
        print(f"⚠️ Config file not found: {config_path}")
        return True  # Not critical for smoke test

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate required fields
        required_fields = [
            "model_name",
            "prior_model",
            "decoder_model",
            "dtype",
            "device",
            "optimizations",
            "recommended_params",
        ]

        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field in config: {field}")
                return False

        print("✅ Cascade config validation passed")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating config: {e}")
        return False


def test_cascade_imports():
    """Test that all required packages can be imported"""

    required_packages = ["torch", "diffusers", "transformers", "PIL", "numpy"]

    failed_imports = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False

    print("✅ All required packages imported successfully")
    return True


def main():
    """Run all smoke tests"""

    print("🧪 Running Cascade Quickstart Smoke Tests")
    print("=" * 50)

    tests = [
        ("Package Imports", test_cascade_imports),
        ("Config Validation", test_cascade_config_validation),
        ("Notebook Execution", test_cascade_notebook_execution),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"💥 {test_name} failed")
        except Exception as e:
            print(f"💥 {test_name} error: {e}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All smoke tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    main()
