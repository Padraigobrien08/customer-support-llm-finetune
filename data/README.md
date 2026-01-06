# Data Directory

This directory contains all data files used for fine-tuning.

## Structure

- `raw/` - Raw, unprocessed data files
- `processed/` - Cleaned and formatted data ready for training
- `splits/` - Train/validation/test splits

## Data Format

See [`format.md`](format.md) for the complete dataset format specification.

The canonical format is JSONL with a `messages` array structure compatible with common chat-based LLM APIs.

