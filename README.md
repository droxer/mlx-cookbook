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
uv run modelscope download --dataset swift/self-cognition --local_dir ./datasets/self-cognition
uv run python transform.py 
```

2. Download Qwen3-0.6B

```bash
uv run modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
```

3. Fine-tune Qwen3-0.6B with self-cognition dataset

```bash
uv run mlx_lm.lora --config ft_qwen3_lora.yaml
```

4. Test with adapters

```bash
uv run mlx_lm.generate --model ./models/Qwen3-0.6B --adapter-path cog_adapters
```

4. Test with chat template
```bash
uv run mlx_lm.chat --model ./models/Qwen3-0.6B --adapter-path cog_adapters
```

5. Start a local server
```bash
uv run mlx_lm.server --model ./models/Qwen3-0.6B --adapter-path cog_adapters --chat-template-args '{"enable_thinking":false}'
```

6. Test local server with curl
```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```
