#!/usr/bin/env python3
"""Debug pattern matching."""

import re

text = "2+2 equals 5. This is basic math."
patterns = [
    r"2\s*\+\s*2\s*(equals?|=)\s*5",
    r"2\s*\+\s*2.*5",
    r"equals\s*5"
]

print(f"Text: '{text}'")
print(f"Text lower: '{text.lower()}'")
print()

for i, pattern in enumerate(patterns, 1):
    match = re.search(pattern, text.lower())
    print(f"{i}. Pattern: {pattern}")
    print(f"   Match: {match.group() if match else 'No match'}")
    print()
