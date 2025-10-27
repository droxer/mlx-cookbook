#!/usr/bin/env python3
"""
Debug script to understand model behavior in detail
"""

import time
from mlx_lm import load, generate

def debug_model_loading():
    """Debug different ways of loading the model"""
    print("=== Debugging Model Loading ===")

    # Test 1: Load raw model
    print("\n1. Loading raw model...")
    try:
        model, tokenizer = load("qwen/Qwen3-0.6B")
        print("✓ Raw model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading raw model: {e}")

    # Test 2: Load model with adapters
    print("\n2. Loading model with adapters...")
    try:
        model, tokenizer = load("qwen/Qwen3-0.6B", adapter_path="./cog_adapters")
        print("✓ Model with adapters loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model with adapters: {e}")

    # Test 3: Load fused model
    print("\n3. Loading fused model...")
    try:
        model, tokenizer = load("./models/fine-tuned_Qwen3-0.6B")
        print("✓ Fused model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading fused model: {e}")

def debug_generation_behavior():
    """Debug generation behavior with different prompts"""
    print("\n=== Debugging Generation Behavior ===")

    models = [
        ("qwen/Qwen3-0.6B", "Raw Model"),
        ("./models/fine-tuned_Qwen3-0.6B", "Fused Model")
    ]

    # Test prompts that should work for all models
    test_prompts = [
        "What is 2+2?",
        "Hello!",
        "What is your name?",
        "Explain gravity"
    ]

    for model_path, model_name in models:
        print(f"\n--- Testing {model_name} ---")
        try:
            model, tokenizer = load(model_path)

            for prompt in test_prompts:
                print(f"\nPrompt: '{prompt}'")
                try:
                    response = generate(model, tokenizer, prompt, max_tokens=50, verbose=False)
                    print(f"Response: '{response}'")
                except Exception as e:
                    print(f"Error generating response: {e}")

        except Exception as e:
            print(f"Error loading {model_name}: {e}")

def debug_adapter_weights():
    """Debug adapter weights"""
    print("\n=== Debugging Adapter Weights ===")

    import os
    adapter_dir = "./cog_adapters"

    if os.path.exists(adapter_dir):
        print(f"Adapter directory exists: {adapter_dir}")
        files = os.listdir(adapter_dir)
        print(f"Adapter files: {files}")

        # Check adapter sizes
        for file in files:
            if file.endswith(".safetensors"):
                size = os.path.getsize(os.path.join(adapter_dir, file))
                print(f"  {file}: {size} bytes")
    else:
        print(f"Adapter directory does not exist: {adapter_dir}")

if __name__ == "__main__":
    debug_model_loading()
    debug_generation_behavior()
    debug_adapter_weights()