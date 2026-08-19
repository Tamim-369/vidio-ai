#!/usr/bin/env python3
"""Test all the specific issues mentioned by the user."""

from src.services.tts import _clean_text, _num_to_words

print("Testing Specific Issue Fixes\n" + "="*60)

# Test Issue 1: Number pronunciation (731 should be smooth)
print("1. 🔢 Testing Number Pronunciation:")
test_numbers = ["731", "457", "1945", "2001", "100", "250"]

for num in test_numbers:
    result = _num_to_words(num)
    print(f"   {num} → '{result}'")
    if " and " in result:
        print(f"   ❌ Contains 'and' - will cause pause!")
    else:
        print(f"   ✅ Smooth pronunciation")
print()

# Test Issue 2: "while" and conjunction pauses
print("2. 🗣️  Testing Conjunction Handling:")
test_sentences = [
    "They did something bad to that person while strapping a metal table",
    "The aircraft was fast while being dangerous",  
    "He worked there during the war while conducting experiments",
    "The pilot flew when weather was bad",
    "They used equipment where testing occurred",
]

for sentence in test_sentences:
    cleaned = _clean_text(sentence)
    print(f"   Original: {sentence}")
    print(f"   Cleaned:  {cleaned}")
    
    # Check for unwanted pauses after conjunctions
    problematic_patterns = [" while,", " when,", " where,", " during,"]
    has_issues = any(pattern in cleaned for pattern in problematic_patterns)
    
    if has_issues:
        print(f"   ❌ Has unwanted conjunction pause")
    else:
        print(f"   ✅ Natural conjunction flow")
    print()

# Test Issue 3: Overall sentence flow
print("3. 🎙️  Testing Sentence Flow:")
complex_sentences = [
    "Unit 731 conducted experiments while documenting results",
    "The pilots flew 457 missions during World War 2 while facing danger", 
    "They used chemical weapons containing 1200 different compounds while testing effectiveness"
]

for sentence in complex_sentences:
    cleaned = _clean_text(sentence)
    print(f"   Original: {sentence}")
    print(f"   Cleaned:  {cleaned}")
    
    # Count commas (pauses)
    pause_count = cleaned.count(',')
    print(f"   Pauses: {pause_count} (fewer is better for flow)")
    print()

print("="*60)
print("🎯 Expected Improvements:")
print("✅ Numbers: 'seven hundred thirty one' (no 'and' pause)")
print("✅ Conjunctions: Natural flow after 'while', 'when', 'where'") 
print("✅ Sentences: Fewer artificial pauses, better rhythm")
print("✅ Video: Zoom continues smoothly without freeze at end")
print("\n🚀 Test your video to hear the improvements!")