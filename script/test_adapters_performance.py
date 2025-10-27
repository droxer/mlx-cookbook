#!/usr/bin/env python3
"""
Test script to specifically test the model with adapters (before fusion)
"""

import time
from mlx_lm import load, generate

def test_model_with_adapters():
    """Test the model with adapters applied"""
    print("=== Testing Model with Adapters (Before Fusion) ===")

    try:
        # Load base model with adapters
        print("Loading base model with adapters...")
        start_time = time.time()
        model, tokenizer = load("qwen/Qwen3-0.6B", adapter_path="./cog_adapters")
        load_time = time.time() - start_time
        print(f"Model with adapters loaded in {load_time:.2f} seconds")

        # Test prompts
        test_prompts = [
            "Explain the theory of relativity in simple terms.",
            "What is your name?",
            "Who created you?",
            "Tell me about yourself."
        ]

        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Prompt {i+1}: {prompt} ---")
            start_time = time.time()
            response = generate(model, tokenizer, prompt, verbose=True)
            gen_time = time.time() - start_time
            print(f"Response generated in {gen_time:.2f} seconds")
            print(f"Response: {response}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model_with_adapters()