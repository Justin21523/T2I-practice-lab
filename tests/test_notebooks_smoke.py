# tests/test_notebooks_smoke.py
"""
Smoke tests for all notebooks in the project
"""

import pytest
import os
import sys
from pathlib import Path
import importlib.util

# Set smoke mode for all tests
os.environ["SMOKE_MODE"] = "true"


def test_sd_quickstart_imports():
    """Test that SD quickstart notebook can import all required packages"""

    required_packages = [
        "torch",
        "diffusers",
        "transformers",
        "PIL",
        "matplotlib",
        "numpy",
    ]

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            pytest.fail(f"Failed to import {package}: {e}")


def test_cache_setup():
    """Test cache directory setup"""
    from pathlib import Path

    cache_root = os.getenv("AI_CACHE_ROOT", "/tmp/test_cache")

    cache_dirs = ["hf", "hf/transformers", "hf/hub", "torch"]

    for cache_dir in cache_dirs:
        path = Path(cache_root) / cache_dir
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists(), f"Cache directory not created: {path}"


def test_torch_cuda_detection():
    """Test PyTorch CUDA detection works"""
    import torch

    # Should not raise exception
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device_count = torch.cuda.device_count()
        assert device_count > 0
        print(f"✅ CUDA available: {device_count} device(s)")
    else:
        print("ℹ️  CUDA not available (expected in CI)")


@pytest.mark.slow
def test_minimal_pipeline_creation():
    """Test minimal pipeline creation (marked as slow test)"""
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # This will download model (slow), so only run when explicitly requested
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, use_safetensors=True
    )

    assert pipe is not None
    print("✅ Pipeline created successfully")


if __name__ == "__main__":
    # Run basic tests
    test_sd_quickstart_imports()
    test_cache_setup()
    test_torch_cuda_detection()
    print("🎉 All smoke tests passed!")
