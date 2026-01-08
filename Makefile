.PHONY: eval lint install

PYTHON := python3

# Run golden evaluation and print summary
eval:
	@echo "Running golden evaluation..."
	@$(PYTHON) -m scripts.run_golden_eval
	@echo ""
	@echo "Evaluation summary:"
	@$(PYTHON) evaluation/eval.py

# Lint placeholder (no tooling configured yet)
lint:
	@echo "Linting not yet configured. Add your preferred linter here."
	@echo "Example: flake8, ruff, pylint, etc."

# Install package in development mode
install:
	pip install -e .

# Install dependencies
install-deps:
	pip install -r requirements.txt

