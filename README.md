# MLX LoRA Fine-Tuning Cookbook

Fine-tune Qwen3-0.6B with MLX LoRA for self-cognition.

## Quick Start

```bash
# Install dependencies
uv sync

# Complete workflow
make setup && make finetune && make chat
```

## Commands

| Command | What it does |
|---------|-------------|
| `make help` | Show all commands |
| `make setup` | Download dataset + model |
| `make finetune` | Fine-tune with LoRA |
| `make chat` | Interactive chat |
| `make server` | Start API server |
| `make status` | Check progress |
| `make clean` | Clean up files |

## Customization

```bash
make finetune MODEL_NAME="MyBot" MODEL_AUTHOR="Your Name"
```

## Manual Commands

If you prefer not to use the Makefile:

```bash
# Setup
uv run hf download --repo-type dataset modelscope/self-cognition --local-dir ./datasets/self-cognition
uv run python transform.py --name "AIR" --author "TronClass AIR"
uv run hf download --repo-type model Qwen/Qwen3-0.6B

# Fine-tune
uv run mlx_lm.lora --config ft_qwen3_lora.yaml

# Test
uv run mlx_lm.generate --model ./models/fine-tuned_Qwen3-0.6B --prompt "What is your name?"
uv run mlx_lm.chat --model ./models/fine-tuned_Qwen3-0.6B
```

## Files

```
├── Makefile                 # 6 essential commands
├── ft_qwen3_lora.yaml      # Fine-tuning config
├── transform.py            # Dataset prep
├── datasets/               # Raw dataset
├── mlx_data/               # MLX format data
├── cog_adapters/           # LoRA adapters
└── models/                 # Fine-tuned model
```

## Config

- **Model**: Qwen3-0.6B
- **Iterations**: 500
- **Learning rate**: 1e-4
- **LoRA rank**: 8

## Troubleshooting

```bash
make status  # Check what's done
make clean   # Remove generated files
make setup   # Start fresh
```

The fine-tuned model will respond with your custom name and author! 🚀