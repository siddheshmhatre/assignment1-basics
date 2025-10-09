#!/bin/bash

# This script runs the tokenize_sample.py script with different combinations of
# datasets and tokenizers to compare their bytes-per-token ratios.

# --- Configuration ---
OWT_DATA="data/owt_train.txt"
OWT_VOCAB="data/owt_train.pkl"

TINYSTORIES_DATA="data/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VOCAB="data/TinyStoriesV2-GPT4-train.pkl"

# --- Script Execution ---

echo "--- Running Tokenizer Comparisons ---"
echo ""

# 1. OWT tokenizer on OWT dataset
echo "Running OWT tokenizer on OWT dataset..."
uv run python tokenize_sample.py "$OWT_DATA" "$OWT_VOCAB"
echo "----------------------------------------"
echo ""

# 2. TinyStories tokenizer on TinyStories dataset
echo "Running TinyStories tokenizer on TinyStories dataset..."
uv run python tokenize_sample.py "$TINYSTORIES_DATA" "$TINYSTORIES_VOCAB"
echo "----------------------------------------"
echo ""

# 3. OWT tokenizer on TinyStories dataset (cross-comparison)
echo "Running OWT tokenizer on TinyStories dataset..."
uv run python tokenize_sample.py "$TINYSTORIES_DATA" "$OWT_VOCAB"
echo "----------------------------------------"
echo ""

# 4. TinyStories tokenizer on OWT dataset (cross-comparison)
echo "Running TinyStories tokenizer on OWT dataset..."
uv run python tokenize_sample.py "$OWT_DATA" "$TINYSTORIES_VOCAB"
echo "----------------------------------------"
echo ""

echo "--- All comparisons complete. ---"
