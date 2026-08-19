#!/usr/bin/env python3
"""Test speed consistency across different sentence lengths."""

from src.services.tts import _target_speed

# Test sentences of varying lengths
test_sentences = [
    # Short (5-10 words)
    "Number ten. The worst fighter jet.",
    "This aircraft was terrible.",
    "It crashed frequently during testing.",
    
    # Medium (11-20 words) 
    "The Brewster F2A Buffalo was America's first monoplane fighter aircraft entering service in 1939.",
    "During World War Two, many countries developed experimental aircraft that proved to be failures.",
    "The F-104 Starfighter could reach impressive speeds but had serious safety issues.",
    
    # Long (21+ words)
    "The Messerschmitt Me 210 was designed during World War 2 to replace the successful Bf 110, however it suffered from numerous design flaws and poor handling characteristics that made it dangerous.",
    "In the 1950s, the Soviet Union developed several fighter aircraft including the MiG-19 which was their first supersonic fighter, although it had limited range and weapons capacity compared to American designs of the same era.",
    "The aircraft had a maximum speed of twelve hundred miles per hour, a service ceiling of fifty thousand feet, and a combat radius of only three hundred miles, making it unsuitable for long-range missions during extended combat operations."
]

print("Testing TTS Speed Consistency\n" + "="*60)

print("Target: Consistent speeds around 1.0x (130 WPM base)\n")

speeds = []
for i, sentence in enumerate(test_sentences, 1):
    word_count = len(sentence.split())
    speed = _target_speed(sentence)
    speeds.append(speed)
    
    # Categorize length
    if word_count <= 10:
        category = "SHORT"
    elif word_count <= 20:
        category = "MEDIUM"
    else:
        category = "LONG"
    
    print(f"{i:2d}. {category:6} ({word_count:2d} words): Speed {speed:.2f}x")
    print(f"    {sentence[:70]}{'...' if len(sentence) > 70 else ''}")
    print()

# Calculate consistency metrics
min_speed = min(speeds)
max_speed = max(speeds)
speed_range = max_speed - min_speed
avg_speed = sum(speeds) / len(speeds)

print("="*60)
print(f"📊 Speed Analysis:")
print(f"   Average: {avg_speed:.2f}x")
print(f"   Range: {min_speed:.2f}x to {max_speed:.2f}x")
print(f"   Variation: {speed_range:.2f}x spread")

if speed_range < 0.15:
    print(f"✅ GOOD: Low variation ({speed_range:.2f}x) - consistent speeds")
elif speed_range < 0.25:
    print(f"⚠️  OKAY: Medium variation ({speed_range:.2f}x) - mostly consistent")
else:
    print(f"❌ BAD: High variation ({speed_range:.2f}x) - inconsistent speeds")

print(f"\n🎯 Result: More consistent speeds prevent 'sometimes very slow' issue!")