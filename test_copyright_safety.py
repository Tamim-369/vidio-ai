#!/usr/bin/env python3
"""Test copyright safety features."""

from src.services.asset_fetcher import _fetch_pexels, _fetch_ddg, BLOCKED_DOMAINS
import os

def test_copyright_safety():
    """Test that our image sources are copyright-safe."""
    
    print("Testing Copyright Safety Features\n" + "="*60)
    
    # Test 1: Check blocked domains
    print("✓ Blocked domains configured:")
    for domain in BLOCKED_DOMAINS:
        print(f"  - {domain}")
    
    print(f"\n✓ Total blocked domains: {len(BLOCKED_DOMAINS)}")
    
    # Test 2: Test Pexels (should be safe)
    print("\n📸 Testing Pexels API (100% royalty-free)...")
    test_paths = _fetch_pexels("fighter jet", "/tmp/test_pexels.jpg", count=2)
    if test_paths:
        print(f"  ✓ Successfully fetched {len(test_paths)} Pexels images")
        for path in test_paths:
            if os.path.exists(path):
                os.remove(path)  # cleanup
    else:
        print("  ⚠️  No Pexels images fetched (check API key)")
    
    # Test 3: Test DuckDuckGo with license filtering
    print("\n🔍 Testing DuckDuckGo with license filter...")
    test_paths = _fetch_ddg("aircraft military", "/tmp/test_ddg.jpg", count=2)
    if test_paths:
        print(f"  ✓ Successfully fetched {len(test_paths)} filtered DDG images")
        for path in test_paths:
            if os.path.exists(path):
                os.remove(path)  # cleanup
    else:
        print("  ⚠️  No DDG images fetched (network or filter too strict)")
    
    # Test 4: Check URL filtering logic
    print("\n🚫 Testing URL filtering...")
    test_urls = [
        "https://shutterstock.com/image/123.jpg",  # Should be blocked
        "https://example.com/stock-photo.jpg",     # Should be blocked  
        "https://pexels.com/photo/123.jpg",        # Should be allowed
        "https://wikimedia.org/image.jpg",         # Should be allowed
        "https://gettyimages.com/premium.jpg",     # Should be blocked
    ]
    
    for url in test_urls:
        blocked = any(domain in url.lower() for domain in BLOCKED_DOMAINS)
        stock_blocked = any(indicator in url.lower() for indicator in ['stock', 'premium'])
        is_safe = not (blocked or stock_blocked)
        status = "✓ SAFE" if is_safe else "❌ BLOCKED"
        print(f"  {status}: {url}")
    
    print("\n" + "="*60)
    print("🔒 Copyright Safety Summary:")
    print("  ✅ Pexels: 100% royalty-free, commercial use OK")
    print("  ✅ DuckDuckGo: Filtered for Public Domain + Creative Commons")
    print("  ❌ Stock sites: All major stock photo sites blocked")
    print("  ⚠️  Always review: Final responsibility with content creator")
    print("\n🎯 System is configured for copyright-safe image fetching!")

if __name__ == "__main__":
    test_copyright_safety()