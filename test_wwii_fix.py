#!/usr/bin/env python3
"""Test WWII/WWI pronunciation fixes."""

from src.services.tts import _expand_numbers

# Test cases for WWII/WWI
wwii_tests = [
    ("WWII began in 1939", "World War Two began in nineteen thirty nine"),
    ("WWI ended in 1918", "World War One ended in nineteen eighteen"), 
    ("During WWII, many battles occurred", "During World War Two, many battles occurred"),
    ("The WWII era was turbulent", "The World War Two era was turbulent"),
    ("WWI aircraft were primitive", "World War One aircraft were primitive"),
    ("Post-WWII development", "Post-World War Two development"),
    
    # Should still work with full names
    ("World War II was global", "World War Two was global"),
    ("World War I started in 1914", "World War One started in nineteen fourteen"),
    ("World War 2 was devastating", "World War Two was devastating"),
    ("World War 1 ended", "World War One ended"),
]

print("Testing WWII/WWI Pronunciation Fixes\n" + "="*50)

all_passed = True
for i, (input_text, expected) in enumerate(wwii_tests, 1):
    # Only test the number expansion part (not the full clean_text with pauses)
    result = _expand_numbers(input_text)
    passed = result == expected
    
    if not passed:
        all_passed = False
        print(f"\n❌ Test {i} FAILED")
        print(f"   Input:    {input_text}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
    else:
        print(f"✓ Test {i}: {input_text}")

print("\n" + "="*50)
if all_passed:
    print("🎉 All WWII/WWI tests PASSED!")
    print("\n✅ Fixed pronunciations:")
    print("   WWII → World War Two")
    print("   WWI → World War One") 
    print("   World War II → World War Two")
    print("   World War I → World War One")
else:
    print("⚠️  Some WWII/WWI tests failed")