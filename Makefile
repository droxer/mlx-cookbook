.PHONY: help setup finetune fuse chat chat-ft server clean status

# Configuration
MODEL_NAME ?= AIR
MODEL_AUTHOR ?= TronClass AIR

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
RED := \033[0;31m
RESET := \033[0m

help: ## Show available commands
	@echo "$(CYAN)MLX LoRA Fine-Tuning Cookbook$(RESET)"
	@echo "$(YELLOW)Commands: setup, finetune, fuse, chat, chat-ft, server, clean, status$(RESET)"

setup: ## Download dataset, model, and transform data
	@echo "$(BLUE)Setting up MLX environment...$(RESET)"
	@mkdir -p datasets/self-cognition mlx_data models
	uv run hf download --repo-type dataset modelscope/self-cognition --local-dir ./datasets/self-cognition
	uv run python script/transform.py --name "$(MODEL_NAME)" --author "$(MODEL_AUTHOR)"
	uv run hf download --repo-type model Qwen/Qwen3-0.6B
	@echo "$(GREEN)Setup complete!$(RESET)"

chat: ## Interactive chat with model
	@echo "$(BLUE)Starting chat with raw model...$(RESET)"
	uv run mlx_lm.chat --model Qwen/Qwen3-0.6B

finetune: setup ## Fine-tune model with LoRA
	@echo "$(BLUE)Starting LoRA fine-tuning...$(RESET)"
	uv run mlx_lm.lora --model Qwen/Qwen3-0.6B --config ft_qwen3_lora.yaml
	@echo "$(GREEN)Fine-tuning complete! Run 'make fuse' to fuse the model.$(RESET)"

fuse: ## Fuse fine-tuned adapters with base model
	@echo "$(BLUE)Fusing fine-tuned adapters with base model...$(RESET)"
	uv run mlx_lm.fuse --model Qwen/Qwen3-0.6B --adapter-path ./cog_adapters --save-path ./models/fine-tuned_Qwen3-0.6B
	@echo "$(GREEN)Model fused successfully!$(RESET)"

chat-ft: ## Interactive chat with fine-tuned model
	@echo "$(BLUE)Starting chat with fine-tuned model...$(RESET)"
	uv run mlx_lm.chat --model ./models/fine-tuned_Qwen3-0.6B

server: ## Start API server
	@echo "$(BLUE)Starting server...$(RESET)"
	uv run mlx_lm.server --model ./models/fine-tuned_Qwen3-0.6B --chat-template-args '{"enable_thinking":false}'

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	rm -rf cog_adapters models/fine-tuned_Qwen3-0.6B
	@echo "$(GREEN)Clean!$(RESET)"

status: ## Check current progress
	@echo "$(CYAN)Status Check$(RESET)"
	@echo "Run 'ls -la' to check files:"
	@echo "  - Dataset: mlx_data/train.jsonl"
	@echo "  - Model: Check Hugging Face cache"
	@echo "  - Adapters: cog_adapters/"
	@echo "  - Fused model: models/fine-tuned_Qwen3-0.6B/"