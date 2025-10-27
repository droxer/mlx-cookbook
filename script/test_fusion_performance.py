#!/usr/bin/env python3
"""
Test script to compare performance of raw model vs adapters vs fused model
"""

import time
from mlx_lm import load, generate

def test_model_performance(model_path, prompt, model_name):
    """Test the performance of a model with a given prompt"""
    print(f"\n=== Testing {model_name} ===")
    print(f"Model path: {model_path}")
    print(f"Prompt: {prompt}")

    try:
        # Load model
        print("Loading model...")
        start_time = time.time()
        model, tokenizer = load(model_path)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        # Generate response
        print("Generating response...")
        start_time = time.time()
        response = generate(model, tokenizer, prompt, verbose=True)
        gen_time = time.time() - start_time
        print(f"Response generated in {gen_time:.2f} seconds")

        if response is None:
            response = "None"
        print(f"Response: {response}")
        print(f"Total time: {load_time + gen_time:.2f} seconds")

        return response, load_time, gen_time
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return f"ERROR: {e}", 0, 0

def main():
    # Test prompts - including general knowledge and domain-specific
    test_prompts = [
        "Explain the theory of relativity in simple terms.",
        "What is your name?",
        "Who created you?",
        "Tell me about yourself."
    ]

    # Models to test
    models = [
        ("qwen/Qwen3-0.6B", "Raw Model"),
        ("./models/fine-tuned_Qwen3-0.6B", "Fused Model")
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {prompt}")
        print('='*60)

        results = {}
        for model_path, model_name in models:
            response, load_time, gen_time = test_model_performance(model_path, prompt, model_name)
            results[model_name] = (response, load_time, gen_time)

        # Compare results
        print(f"\n--- Performance Comparison ---")
        for model_name, (response, load_time, gen_time) in results.items():
            print(f"{model_name} - Load: {load_time:.2f}s, Generate: {gen_time:.2f}s")

if __name__ == "__main__":
    main()