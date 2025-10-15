MLX Cookbook
============

Cookbook for MLX Lora with Qwen3-0.6B.

Installation
------------

```bash
uv sync
```

Usage
-----

1. Transform dataset to mlx format

```bash
uv run hf download --repo-type dataset modelscope/self-cognition --local-dir ./datasets/self-cognition
uv run python transform.py 
```

2. Download Qwen3-0.6B

```bash
uv run hf download --repo-type model Qwen/Qwen3-0.6B
```

3. Fine-tune Qwen3-0.6B with self-cognition dataset

```bash
uv run mlx_lm.lora --config ft_qwen3_lora.yaml
```

4. Test with adapters

```bash
uv run mlx_lm.generate --model qwen/Qwen3-0.6B --adapter-path cog_adapters --prompt "Say this is a test"
```

5. fuse adapters
```bash
uv run mlx_lm.fuse --model qwen/Qwen3-0.6B --adapter-path ./cog_adapters --save-path ./models/fine-tuned_Qwen3-0.6B
```

6. Test fine-tuned model with adapters

```bash
uv run mlx_lm.generate --model ./models/fine-tuned_Qwen3-0.6B --prompt "Say this is a test"
```

4. Test fine-tuned model with chat template
```bash
uv run mlx_lm.chat --model ./models/fine-tuned_Qwen3-0.6B
```

5. Start a local server with fine-tuned model
```bash
uv run mlx_lm.server --model ./models/fine-tuned_Qwen3-0.6B --chat-template-args '{"enable_thinking":false}'
```

6. Test local server with curl with fine-tuned model
```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```
