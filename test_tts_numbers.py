#!/usr/bin/env python3
"""Test script to verify TTS number handling."""

from src.services.tts import _clean_text

# Test cases that should be handled correctly
test_cases = [
    # World War issues
    ("World War 2 was devastating", "World War Two was devastating"),
    ("World War 1 ended in 1918", "World War One ended in nineteen eighteen"),
    ("World War II was global", "World War Two was global"),
    ("World War I started in 1914", "World War One started in nineteen fourteen"),
    ("WWII began in 1939", "World War Two began in nineteen thirty nine"),
    ("WWI was called the Great War", "World War One was called the Great War"),
    
    # Military designations
    ("The F-16 fighter jet", "The F sixteen fighter jet"),
    ("MiG-21 was used extensively", "M i G twenty one was used extensively"),
    ("B-52 bomber aircraft", "B fifty two bomber aircraft"),
    ("The F-35A variant", "The F thirty five A variant"),
    
    # Years in context
    ("Built in 1945", "Built in nineteen forty five"),
    ("Since 2001", "Since two thousand one"),
    ("During 1960", "During nineteen sixty"),
    ("The 1980s were interesting", "The nineteen eighties were interesting"),
    
    # List numbers with pauses
    ("Number 10. The worst jet", "Number ten,, The worst jet"),
    ("Number 5. This aircraft", "Number five,, This aircraft"),
    
    # Regular numbers
    ("It had 250 units", "It had two hundred and fifty units"),
    ("Cost 1500 dollars", "Cost one thousand five hundred dollars"),
    
    # Ordinals
    ("The 1st place winner", "The first place winner"),
    ("The 21st century", "The twenty first century"),
    
    # Abbreviations
    ("USA and USSR", "U S A and U S S R"),
    ("NATO vs Warsaw Pact", "N A T O versus Warsaw Pact"),
    ("RAF and USAF", "R A F and U S A F"),
    ("Top speed 500 mph", "Top speed five hundred miles per hour"),
    
    # More edge cases
    ("Type 99 tank", "Type ninety nine tank"),
    ("A-10 Warthog", "A ten Warthog"),
    ("Su-27 fighter", "S u twenty seven fighter"),
    ("Mach 2 speed", "Mach two speed"),
    ("Mark II variant", "Mark II variant"),
    ("Version 3.5", "Version three point five"),
    ("From 1939 to 1945", "From nineteen thirty nine to nineteen forty five"),
    ("Early 1940s era", "Early nineteen forties era"),
    ("Mid 2010s design", "Mid twenty tens design"),
    ("Costs $50,000", "Costs fifty thousand"),
    ("Weight 5,000 lbs", "Weight five thousand lbs"),
    ("Range 1,500 miles", "Range one thousand five hundred miles"),
    ("100% failure rate", "one hundred percent failure rate"),
    ("0-60 in 4 seconds", "zero to sixty in four seconds"),
    ("Top 3 worst", "Top three worst"),
    ("Number 1 failure", "Number one,, failure"),
]

print("Testing TTS Number Handling\n" + "="*60)
all_passed = True

for i, (input_text, expected) in enumerate(test_cases, 1):
    result = _clean_text(input_text)
    passed = result == expected
    
    if not passed:
        all_passed = False
        print(f"\n❌ Test {i} FAILED")
        print(f"   Input:    {input_text}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
    else:
        print(f"✓ Test {i}: {input_text[:50]}...")

print("\n" + "="*60)
if all_passed:
    print("🎉 All tests PASSED!")
else:
    print("⚠️  Some tests FAILED - check output above")
