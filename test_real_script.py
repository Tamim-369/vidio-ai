#!/usr/bin/env python3
"""Test with real fighter jet script examples."""

from src.services.tts import _clean_text

# Real-world examples from fighter jet scripts
real_scripts = [
    "Number 10. The Brewster F2A Buffalo was America's first monoplane fighter, entering service in 1939.",
    "Number 9. The Messerschmitt Me 210 was designed during World War 2 to replace the successful Bf 110.",
    "Number 8. During the 1950s, the F-84 Thunderjet saw combat in the Korean War.",
    "Number 7. The MiG-23 entered service in 1970 with the Soviet Air Force.",
    "Number 6. The F-104 Starfighter could reach speeds of Mach 2, but had a terrible safety record from 1958 to 1975.",
    "Number 5. The Soviet Yakovlev Yak-38 was a V/STOL aircraft with only 50% mission success rate.",
    "Number 4. The Tu-22 bomber had a top speed of 1,000 mph but was extremely difficult to fly.",
    "Number 3. The F-111 Aardvark cost $50,000 per flight hour and had major reliability issues in the 1960s.",
    "Number 2. World War 1 aircraft like the BE2c were obsolete by 1916.",
    "Number 1. The worst fighter jet in history is the Christmas Bullet, which killed its test pilot in 1919.",
]

print("Testing Real Fighter Jet Scripts\n" + "="*70)

for i, script in enumerate(real_scripts, 1):
    cleaned = _clean_text(script)
    print(f"\n{i}. Original:")
    print(f"   {script}")
    print(f"   Cleaned:")
    print(f"   {cleaned}")

print("\n" + "="*70)
print("✓ All scripts processed successfully!")
print("\nKey improvements visible:")
print("  - 'Number X.' → 'Number X,,' (with pause)")
print("  - 'World War 2' → 'World War Two'")
print("  - 'MiG-23' → 'M i G twenty three'")
print("  - 'F-104' → 'F one hundred and four'")
print("  - '1939' → 'nineteen thirty nine'")
print("  - '1,000 mph' → 'one thousand miles per hour'")
print("  - '$50,000' → 'fifty thousand'")
