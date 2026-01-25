#!/bin/bash
# Script to evaluate and compare Mistral and Llama models

set -e

echo "=" 
echo "Model Quality Assessment - Mistral vs Llama"
echo "="
echo

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Error: API server is not running at http://localhost:8000"
    echo "   Please start the server first:"
    echo "   export MODEL_ID='mistralai/Mistral-7B-Instruct-v0.2'"
    echo "   export ADAPTER_DIR='outputs/run_004_mistral7b'"
    echo "   python -m uvicorn inference.server:app --reload"
    exit 1
fi

echo "Step 1: Testing Mistral-7B-Instruct"
echo "-----------------------------------"
echo
echo "Make sure your server is running with Mistral model, then press Enter..."
read

python3 evaluate_model_quality.py \
    --api-url http://localhost:8000 \
    --model-name "Mistral-7B-Instruct" \
    --output quality_report_mistral.json

echo
echo "Step 2: Testing Llama-3-8B-Instruct"
echo "-----------------------------------"
echo
echo "Now switch to Llama model:"
echo "   export MODEL_ID='meta-llama/Meta-Llama-3-8B-Instruct'"
echo "   export ADAPTER_DIR='outputs/run_003_llama3'"
echo "   (Restart the server if needed)"
echo
echo "Press Enter when Llama model is loaded..."
read

python3 evaluate_model_quality.py \
    --api-url http://localhost:8000 \
    --model-name "Llama-3-8B-Instruct" \
    --output quality_report_llama.json \
    --compare-with quality_report_mistral.json

echo
echo "=" 
echo "✓ Quality assessment complete!"
echo "="
echo
echo "Reports generated:"
echo "  - quality_report_mistral.json"
echo "  - quality_report_llama.json"
echo "  - model_comparison_mistral_vs_llama.json"
echo
