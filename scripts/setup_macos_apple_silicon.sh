#!/bin/bash
set -euo pipefail

# Setup script for macOS Apple Silicon
# Creates venv, installs dependencies, and verifies installation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "macOS Apple Silicon Setup"
echo "=========================================="
echo ""

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Warning: This script is designed for macOS. Continuing anyway..."
    echo ""
fi

# Check for Apple Silicon
if [[ "$(uname -m)" == "arm64" ]]; then
    echo "✓ Detected Apple Silicon (arm64)"
else
    echo "⚠ Not Apple Silicon (detected: $(uname -m))"
fi
echo ""

# Create venv if it doesn't exist
VENV_PATH="$PROJECT_ROOT/.venv"
if [[ ! -d "$VENV_PATH" ]]; then
    echo "Creating virtual environment at .venv..."
    python3 -m venv "$VENV_PATH"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists at .venv"
fi
echo ""

# Activate venv
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing requirements from requirements.txt..."
if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
    echo "Error: requirements.txt not found at $PROJECT_ROOT/requirements.txt"
    exit 1
fi

pip install -r "$PROJECT_ROOT/requirements.txt"
echo "✓ Requirements installed"
echo ""

# Verification step
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

python3 << 'EOF'
import sys

errors = []

try:
    import datasets
    print(f"✓ datasets {datasets.__version__}")
except ImportError as e:
    errors.append(f"✗ datasets: {e}")
    print(f"✗ datasets: Import failed")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError as e:
    errors.append(f"✗ transformers: {e}")
    print(f"✗ transformers: Import failed")

try:
    import peft
    print(f"✓ peft {peft.__version__}")
except ImportError as e:
    errors.append(f"✗ peft: {e}")
    print(f"✗ peft: Import failed")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except ImportError as e:
    errors.append(f"✗ torch: {e}")
    print(f"✗ torch: Import failed")

if errors:
    print("\nErrors encountered:")
    for error in errors:
        print(f"  {error}")
    sys.exit(1)
else:
    print("\n✓ All core packages imported successfully")
EOF

VERIFY_EXIT_CODE=$?

if [[ $VERIFY_EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "Error: Verification failed. Please check the errors above."
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run a training smoke test:"
echo "   python training/train.py --model_id gpt2 --num_epochs 1 --save_steps 10"
echo ""
echo "   This will run a quick test with GPT-2 (small model) for 1 epoch."
echo "   Adjust --model_id if you want to test with a different model."
echo ""
echo "3. Build training data (if not already done):"
echo "   python scripts/build_training_jsonl.py"
echo ""

