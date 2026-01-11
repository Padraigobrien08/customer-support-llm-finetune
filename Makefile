.PHONY: all eval lint install install-deps build-data train-smoke eval-base eval-adapter diff score-adapter score-base export-review review-pack clean help

PYTHON := python3
MODEL_ID := TinyLlama/TinyLlama-1.1B-Chat-v1.0
ADAPTER_DIR := outputs/smoke_001
RESULTS_BASE := evaluation/results/base.json
RESULTS_ADAPTER := evaluation/results/adapter.json

all: install-deps

# Install package in development mode
install:
	@echo "Installing project in editable mode..."
	pip install -e .

# Install dependencies
install-deps:
	@echo "Installing core dependencies..."
	pip install -r requirements.txt
	@echo "Installing LLM-specific dependencies..."
	pip install -e ".[llm]"

# Build dataset: generate synthetic cases (v1 and v2), build dataset, and run sanity checks
build-data:
	@echo "=========================================="
	@echo "Building dataset (Data v1 + v2)"
	@echo "=========================================="
	@echo ""
	@echo "1. Generating synthetic cases (v1)..."
	@$(PYTHON) scripts/generate_synthetic_cases.py --mode v1 --n_per_category 30 --seed 7
	@echo ""
	@echo "2. Generating synthetic cases (v2)..."
	@$(PYTHON) scripts/generate_synthetic_cases.py --mode v2 --n_per_category 30 --seed 7
	@echo ""
	@echo "3. Building dataset with splits..."
	@$(PYTHON) scripts/build_dataset.py --seed 7
	@echo ""
	@echo "4. Running sanity checks..."
	@$(PYTHON) scripts/sanity_check.py data/raw/manual_cases.jsonl
	@$(PYTHON) scripts/sanity_check.py data/raw/manual_cases_v2.jsonl || echo "Warning: manual_cases_v2.jsonl not found (optional)"
	@$(PYTHON) scripts/sanity_check.py data/raw/synthetic_cases.jsonl
	@$(PYTHON) scripts/sanity_check.py data/raw/synthetic_cases_v2.jsonl
	@$(PYTHON) scripts/sanity_check.py data/processed/all.jsonl
	@echo ""
	@echo "✓ Dataset build complete!"

# Smoke test training with safe defaults for MPS
train-smoke:
	@echo "=========================================="
	@echo "Smoke Test Training (MPS)"
	@echo "=========================================="
	@echo ""
	@if [ ! -f data/splits/train.jsonl ]; then \
		echo "Error: Training data not found. Run 'make build-data' first."; \
		exit 1; \
	fi
	@echo "Model: $(MODEL_ID)"
	@echo "Output: $(ADAPTER_DIR)"
	@echo ""
	@$(PYTHON) training/train.py \
		--model_id $(MODEL_ID) \
		--train_file data/splits/train.jsonl \
		--output_dir $(ADAPTER_DIR) \
		--batch_size 1 \
		--gradient_accumulation_steps 1 \
		--num_epochs 1 \
		--max_seq_length 256 \
		--save_steps 10
	@echo ""
	@echo "✓ Training complete! Adapter saved to $(ADAPTER_DIR)"

# Run golden evaluation with base model
eval-base:
	@echo "=========================================="
	@echo "Golden Evaluation (Base Model)"
	@echo "=========================================="
	@echo ""
	@$(PYTHON) scripts/run_golden_eval.py \
		--provider hf_local \
		--model-id $(MODEL_ID) \
		--test-cases-file evaluation/test_cases.json \
		--out $(RESULTS_BASE)
	@echo ""
	@echo "✓ Base evaluation complete: $(RESULTS_BASE)"

# Run golden evaluation with fine-tuned adapter
eval-adapter:
	@echo "=========================================="
	@echo "Golden Evaluation (Fine-tuned Adapter)"
	@echo "=========================================="
	@echo ""
	@if [ ! -d $(ADAPTER_DIR) ]; then \
		echo "Error: Adapter directory not found: $(ADAPTER_DIR)"; \
		echo ""; \
		echo "To create the adapter, run:"; \
		echo "  make train-smoke"; \
		echo ""; \
		echo "Or if you have an existing adapter, set ADAPTER_DIR:"; \
		echo "  make eval-adapter ADAPTER_DIR=outputs/your_adapter"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/run_golden_eval.py \
		--provider hf_local \
		--model-id $(MODEL_ID) \
		--adapter-dir $(ADAPTER_DIR) \
		--test-cases-file evaluation/test_cases.json \
		--out $(RESULTS_ADAPTER)
	@echo ""
	@echo "✓ Adapter evaluation complete: $(RESULTS_ADAPTER)"

# Compare base vs adapter results
diff:
	@echo "=========================================="
	@echo "Comparing Base vs Adapter Results"
	@echo "=========================================="
	@echo ""
	@if [ ! -f $(RESULTS_BASE) ]; then \
		echo "Error: Base results not found: $(RESULTS_BASE)"; \
		echo "Run 'make eval-base' first."; \
		exit 1; \
	fi
	@if [ ! -f $(RESULTS_ADAPTER) ]; then \
		echo "Error: Adapter results not found: $(RESULTS_ADAPTER)"; \
		echo "Run 'make eval-adapter' first."; \
		exit 1; \
	fi
	@$(PYTHON) scripts/compare_runs.py $(RESULTS_ADAPTER) $(RESULTS_BASE)

# Score adapter evaluation results
score-adapter:
	@echo "=========================================="
	@echo "Scoring Adapter Evaluation Results"
	@echo "=========================================="
	@echo ""
	@if [ ! -f $(RESULTS_ADAPTER) ]; then \
		echo "Error: Adapter results not found: $(RESULTS_ADAPTER)"; \
		echo "Run 'make eval-adapter' first."; \
		exit 1; \
	fi
	@$(PYTHON) evaluation/score_results.py \
		$(RESULTS_ADAPTER) \
		evaluation/test_cases.json
	@echo ""
	@echo "✓ Scoring complete. Results: evaluation/results/scored_adapter.json"

# Score base evaluation results
score-base:
	@echo "=========================================="
	@echo "Scoring Base Evaluation Results"
	@echo "=========================================="
	@echo ""
	@if [ ! -f $(RESULTS_BASE) ]; then \
		echo "Error: Base results not found: $(RESULTS_BASE)"; \
		echo "Run 'make eval-base' first."; \
		exit 1; \
	fi
	@$(PYTHON) evaluation/score_results.py \
		$(RESULTS_BASE) \
		evaluation/test_cases.json
	@echo ""
	@echo "✓ Scoring complete. Results: evaluation/results/scored_base.json"

# Export scored results to CSV for review
export-review:
	@echo "=========================================="
	@echo "Exporting Scored Results to CSV"
	@echo "=========================================="
	@echo ""
	@if [ ! -f evaluation/results/scored_adapter.json ]; then \
		echo "Error: Scored adapter results not found."; \
		echo "Run 'make score-adapter' first."; \
		exit 1; \
	fi
	@$(PYTHON) evaluation/export_review_csv.py \
		evaluation/results/scored_adapter.json
	@echo ""
	@echo "✓ CSV exported: evaluation/results/scored_adapter.csv"
	@echo "  Ready to open in Google Sheets!"

# Alias for export-review
review-pack: export-review

# Run golden evaluation and print summary (legacy)
eval: install-deps
	@echo "Running golden evaluation..."
	@$(PYTHON) scripts/run_golden_eval.py
	@echo ""
	@echo "Evaluation summary:"
	@$(PYTHON) evaluation/eval.py

# Lint placeholder (no tooling configured yet)
lint:
	@echo "Linting not yet configured. Add your preferred linter here."
	@echo "Example: flake8, ruff, pylint, etc."
	@# flake8 .
	@# mypy .

# Show help/usage
help:
	@echo "Customer Support LLM Fine-tuning - Makefile Targets"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make build-data          - Generate synthetic cases, build dataset, run sanity checks"
	@echo ""
	@echo "Training:"
	@echo "  make train-smoke          - Train LoRA adapter with safe defaults for MPS"
	@echo "                            (Model: $(MODEL_ID), Output: $(ADAPTER_DIR))"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-base            - Run golden evaluation with base model"
	@echo "  make eval-adapter         - Run golden evaluation with fine-tuned adapter"
	@echo "  make diff                 - Compare base vs adapter results"
	@echo ""
	@echo "Scoring & Review:"
	@echo "  make score-base           - Score base evaluation results"
	@echo "  make score-adapter        - Score adapter evaluation results"
	@echo "  make export-review        - Export scored results to CSV for review"
	@echo "  make review-pack          - Alias for export-review"
	@echo ""
	@echo "Customization:"
	@echo "  make train-smoke MODEL_ID=your-model ADAPTER_DIR=outputs/custom"
	@echo "  make eval-adapter ADAPTER_DIR=outputs/your_adapter"
	@echo ""
	@echo "Other:"
	@echo "  make install-deps         - Install all dependencies"
	@echo "  make clean                - Clean build artifacts"

# Clean build artifacts
clean:
	@echo "Cleaning up build artifacts and caches..."
	rm -rf .venv
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist
	rm -rf outputs/*
	rm -rf data/processed/*
	rm -rf data/splits/*
	rm -rf data/raw/synthetic_cases.jsonl
	rm -rf data/raw/synthetic_cases_v2.jsonl
	rm -rf evaluation/results/*
	@echo "Clean up complete."
