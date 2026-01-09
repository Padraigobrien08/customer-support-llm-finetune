#!/bin/bash
set -euo pipefail

# Smoke test training script for Apple Silicon (MPS)
# Runs a quick training job with conservative settings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Smoke Test Training (Apple Silicon)"
echo "=========================================="
echo ""

# Check if venv exists
VENV_PATH="$PROJECT_ROOT/.venv"
if [[ ! -d "$VENV_PATH" ]]; then
    echo "Error: Virtual environment not found at .venv"
    echo "Run: bash scripts/setup_macos_apple_silicon.sh"
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Validate training data exists
TRAIN_FILE="$PROJECT_ROOT/data/processed/train_seed.jsonl"
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Error: Training data not found at $TRAIN_FILE"
    echo "Run: python scripts/build_training_jsonl.py"
    exit 1
fi
echo "✓ Training data found: $TRAIN_FILE"
echo ""

# Set default model ID (can be overridden via env var)
MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/smoke_001"

echo "Configuration:"
echo "  Model ID: $MODEL_ID"
echo "  Training file: $TRAIN_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Build command with conservative settings for smoke test
CMD=(
    python training/train.py
    --model_id "$MODEL_ID"
    --train_file "$TRAIN_FILE"
    --output_dir "$OUTPUT_DIR"
    --batch_size 1
    --gradient_accumulation_steps 1
    --num_epochs 1
    --max_seq_length 256
    --save_steps 10
)

echo "Executing command:"
echo "  ${CMD[*]}"
echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Run the command
"${CMD[@]}"

TRAIN_EXIT_CODE=$?

if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "Error: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Smoke test training complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test the fine-tuned adapter:"
echo "   python inference/demo.py --model_id $MODEL_ID --adapter_path $OUTPUT_DIR"
echo ""
echo "   Or use the adapter demo script if available:"
echo "   python scripts/demo_adapter.py --model_id $MODEL_ID --adapter_path $OUTPUT_DIR"
echo ""
echo "2. Check the training output:"
echo "   ls -lh $OUTPUT_DIR"
echo ""

