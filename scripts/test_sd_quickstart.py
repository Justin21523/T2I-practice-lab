# scripts/test_sd_quickstart.py
#!/usr/bin/env python3
"""
Standalone test script for SD Quickstart functionality
Can be used for CI/CD pipeline validation
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """Setup environment for testing"""
    warnings.filterwarnings("ignore")

    # Set smoke mode for fast testing
    os.environ["SMOKE_MODE"] = "true"

    # Setup cache directories
    cache_root = os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache_test")
    for key, subdir in {
        "HF_HOME": "hf",
        "TRANSFORMERS_CACHE": "hf/transformers",
        "HUGGINGFACE_HUB_CACHE": "hf/hub",
    }.items():
        path = Path(cache_root) / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


def test_sd15_pipeline():
    """Test SD1.5 pipeline creation and basic inference"""
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    print("🧪 Testing SD1.5 pipeline...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, use_safetensors=True
    ).to(device)

    # Enable optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    # Set scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    # Test generation
    generator = torch.Generator(device=device).manual_seed(42)

    result = pipe(
        prompt="a simple test image",
        width=256,
        height=256,
        num_inference_steps=2,
        guidance_scale=5.0,
        generator=generator,
        return_dict=True,
    )

    assert len(result.images) == 1
    assert result.images[0].size == (256, 256)

    print("✅ SD1.5 test passed")

    # Cleanup
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()


def test_memory_management():
    """Test memory optimization functions"""
    print("🧪 Testing memory management...")

    if torch.cuda.is_available():
        # Test memory monitoring
        initial_memory = torch.cuda.memory_allocated()

        # Allocate some memory
        dummy_tensor = torch.randn(1000, 1000, device="cuda")
        allocated_memory = torch.cuda.memory_allocated()

        assert allocated_memory > initial_memory

        # Cleanup
        del dummy_tensor
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory

        print("✅ Memory management test passed")
    else:
        print("ℹ️  Skipping memory test (no CUDA)")


def main():
    """Run all tests"""
    print("🚀 Starting SD Quickstart Tests")
    print("=" * 50)

    try:
        setup_environment()
        test_memory_management()
        test_sd15_pipeline()

        print("\n🎉 All tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
