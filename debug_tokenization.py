#!/usr/bin/env python3
"""
Debug tokenization issues for WAN attention storage
"""

from transformers import AutoTokenizer

# Load the same tokenizer that WAN uses
tokenizer_name = "google/umt5-xxl"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Test prompts
prompt = "(flower)"
word = "flower"

print(f"Testing tokenization for prompt: '{prompt}' and word: '{word}'")
print()

# Method 1: Tokenize word individually
tokens_individual = tokenizer.encode(word, add_special_tokens=False)
print(f"Word '{word}' tokenized individually: {tokens_individual}")
for i, token_id in enumerate(tokens_individual):
    token_text = tokenizer.decode([token_id])
    print(f"  Token {i}: ID={token_id}, Text='{token_text}'")

print()

# Method 2: Tokenize full prompt
full_tokens_no_special = tokenizer.encode(prompt, add_special_tokens=False)
full_text_tokens = [tokenizer.decode([tid]) for tid in full_tokens_no_special]

print(f"Full prompt '{prompt}' tokenized: {full_tokens_no_special}")
for i, (token_id, token_text) in enumerate(zip(full_tokens_no_special, full_text_tokens)):
    print(f"  Token {i}: ID={token_id}, Text='{token_text}'")

print()

# Method 3: Find word in context (like our new logic)
word_token_ids = []
word_positions = []

for i, token_text in enumerate(full_text_tokens):
    if word.lower() in token_text.lower() or token_text.lower() in word.lower():
        word_token_ids.append(full_tokens_no_special[i])
        word_positions.append(i)
        print(f"Found word '{word}' in token {i}: ID={full_tokens_no_special[i]}, Text='{token_text}'")

print(f"Word '{word}' tokens in context: {word_token_ids} at positions {word_positions}")

print()

# Method 4: Test with padding (like the attention storage uses)
tokens_padded = tokenizer(
    prompt, 
    return_tensors="pt", 
    padding="max_length", 
    max_length=512, 
    truncation=True
)
token_sequence_padded = tokens_padded["input_ids"][0].tolist()

print(f"Padded sequence (first 10): {token_sequence_padded[:10]}")

# Check if context tokens appear in padded sequence
for token_id in word_token_ids:
    positions = [i for i, tid in enumerate(token_sequence_padded) if tid == token_id]
    print(f"Context token ID {token_id} found at positions: {positions}")
