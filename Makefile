# MLX LoRA Fine-Tuning Cookbook
# Simple, essential targets only

.PHONY: help setup finetune chat server clean status

# Configuration
MODEL_NAME ?= AIR
MODEL_AUTHOR ?= TronClass AIR

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
RESET := \033[0m

help: ## Show available commands
	@echo "$(CYAN)MLX LoRA Fine-Tuning Cookbook$(RESET)"
	@echo "$(CYAN)=============================$(RESET)"
	@echo ""
	@echo "$(YELLOW)Essential commands:$(RESET)"
	@echo "  $(GREEN)setup$(RESET)     Download and prepare everything"
	@echo "  $(GREEN)finetune$(RESET)  Fine-tune the model with LoRA"
	@echo "  $(GREEN)chat$(RESET)      Interactive chat with fine-tuned model"
	@echo "  $(GREEN)server$(RESET)    Start API server"
	@echo "  $(GREEN)clean$(RESET)     Clean up generated files"
	@echo "  $(GREEN)status$(RESET)    Check what's done"
	@echo ""
	@echo "$(YELLOW)Example:$(RESET)"
	@echo "  make setup && make finetune && make chat"

setup: ## Download dataset, model, and transform data
	@echo "$(BLUE)Setting up MLX environment...$(RESET)"
	@mkdir -p datasets/self-cognition mlx_data models

	# Download dataset
	@if [ ! -f "mlx_data/train.jsonl" ]; then \
		echo "$(YELLOW)Downloading dataset...$(RESET)"; \
		uv run hf download --repo-type dataset modelscope/self-cognition --local-dir ./datasets/self-cognition; \
		uv run python transform.py --name "$(MODEL_NAME)" --author "$(MODEL_AUTHOR)"; \
		echo "$(GREEN)✓ Dataset ready$(RESET)"; \
	fi

	# Download model (checks HF cache)
	@if ! uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)" 2>/dev/null; then \
		echo "$(YELLOW)Downloading model to HF cache...$(RESET)"; \
		uv run hf download --repo-type model Qwen/Qwen3-0.6B; \
		echo "$(GREEN)✓ Model ready in HF cache$(RESET)"; \
	else \
		echo "$(GREEN)✓ Model already in HF cache$(RESET)"; \
	fi

	@echo "$(GREEN)Setup complete! Run 'make finetune' to start fine-tuning.$(RESET)"

finetune: setup ## Fine-tune model with LoRA
	@echo "$(BLUE)Starting LoRA fine-tuning...$(RESET)"
	@echo "$(YELLOW)This may take a while...$(RESET)"
	uv run mlx_lm.lora --config ft_qwen3_lora.yaml
	@echo "$(GREEN)Fine-tuning complete!$(RESET)"

chat: ## Interactive chat with fine-tuned model
	@echo "$(BLUE)Starting chat...$(RESET)"
	@if [ -d "models/fine-tuned_Qwen3-0.6B" ]; then \
		echo "$(YELLOW)Using fused model$(RESET)"; \
		uv run mlx_lm.chat --model ./models/fine-tuned_Qwen3-0.6B; \
	elif [ -d "cog_adapters" ]; then \
		echo "$(YELLOW)Using model with adapters$(RESET)"; \
		echo "$(YELLOW)Tip: run 'make fuse' for better performance$(RESET)"; \
		uv run mlx_lm.generate --model qwen/Qwen3-0.6B --adapter-path ./cog_adapters --prompt "Hello! What's your name?"; \
	else \
		echo "$(RED)No fine-tuned model found. Run 'make finetune' first.$(RESET)"; \
		exit 1; \
	fi

server: ## Start API server
	@echo "$(BLUE)Starting server...$(RESET)"
	@if [ -d "models/fine-tuned_Qwen3-0.6B" ]; then \
		echo "$(YELLOW)Server: http://localhost:8080$(RESET)"; \
		echo "$(YELLOW)API: http://localhost:8080/v1/chat/completions$(RESET)"; \
		uv run mlx_lm.server --model ./models/fine-tuned_Qwen3-0.6B --chat-template-args '{"enable_thinking":false}'; \
	else \
		echo "$(RED)No fine-tuned model found. Run 'make finetune' first.$(RESET)"; \
		exit 1; \
	fi

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	rm -rf cog_adapters models/fine-tuned_Qwen3-0.6B
	@echo "$(GREEN)Clean! Downloads preserved.$(RESET)"

status: ## Check current progress
	@echo "$(CYAN)Status Check$(RESET)"
	@echo "============"
	@if [ -f "mlx_data/train.jsonl" ]; then echo "$(GREEN)✓$(RESET) Dataset ready ($(shell wc -l < mlx_data/train.jsonl) examples)"; else echo "$(RED)✗$(RESET) Dataset not ready"; fi
	@if uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)" 2>/dev/null; then echo "$(GREEN)✓$(RESET) Model cached (Qwen3-0.6B)"; else echo "$(RED)✗$(RESET) Model not in cache"; fi
	@if [ -d "cog_adapters" ]; then echo "$(GREEN)✓$(RESET) LoRA adapters trained"; else echo "$(RED)✗$(RESET) Adapters not trained"; fi
	@if [ -d "models/fine-tuned_Qwen3-0.6B" ]; then echo "$(GREEN)✓$(RESET) Fused model ready"; else echo "$(RED)✗$(RESET) Model not fused"; fi