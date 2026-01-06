# Customer Support LLM Fine-tuning

A clean, framework-agnostic repository for fine-tuning language models for customer support applications.

## Project Structure

```
customer-support-llm-finetune/
├── data/
│   ├── raw/              # Raw, unprocessed data
│   ├── processed/        # Cleaned and formatted data
│   ├── splits/           # Train/validation/test splits
│   └── README.md
├── prompts/
│   ├── system.txt        # System prompt definition
│   └── guidelines.md     # Prompt engineering guidelines
├── training/
│   ├── config.yaml       # Training configuration
│   ├── train.py          # Main training script
│   └── utils.py          # Training utilities
├── evaluation/
│   ├── eval.py           # Evaluation script
│   └── test_cases.json   # Test cases for evaluation
├── inference/
│   └── demo.py           # Inference demo script
├── scripts/
│   ├── prepare_data.py   # Data preparation script
│   └── sanity_check.py   # Data and config validation
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── LICENSE              # License file
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your data:
   ```bash
   python scripts/prepare_data.py
   ```

3. Run sanity checks:
   ```bash
   python scripts/sanity_check.py
   ```

4. Configure training in `training/config.yaml`

5. Train the model:
   ```bash
   python training/train.py
   ```

6. Evaluate:
   ```bash
   python evaluation/eval.py
   ```

7. Test inference:
   ```bash
   python inference/demo.py
   ```

## Notes

- This is a minimal scaffold. Add your chosen framework and model implementation.
- Keep data files out of version control (see `.gitignore`).
- Customize prompts and configuration for your specific use case.

