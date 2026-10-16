"""Commons-first candidate sourcing, keyed off the topic entity pool.

WHY THIS REPLACES THE PER-LINE PATH
-----------------------------------
The old path asked Wikimedia Commons a full editorial sentence per narration
line ("Japanese carrier strike force Kido Butai 1942 archive photo") and got
ZERO hits, because Commons indexes archival photo FILENAMES, not prose. Pexels
understood prose, so it filled every slot and won 11/11 slots with modern stock.

Measured on the same archive:
    prose headline query  ->  0/10 hits
    entity query          -> 10/10 hits

So candidates now come from a topic-level entity pool (see entities.py), and
Commons leads. Pexels is off by default for this channel: its corpus is modern
commercial stock with no 1942 archival photography, so no amount of ranking
can reach past "a ship, but the wrong era". Set ASSET_ENABLE_PEXELS=1 to
restore the old provider for A/B comparison.
"""
from __future__ import annotations

import os

# Pexels is the stock provider for this channel. It is curated, royalty-free
# and tagged, so a generic/symbolic beat ("woman scrolling phone", "empty
# courtroom") gets a real, representative photo instead of an unvetted
# Flickr/maxpixel upload. Commons still leads for real archival subjects.
# Set ASSET_ENABLE_PEXELS=0 to disable.
ASSET_ENABLE_PEXELS = os.getenv("ASSET_ENABLE_PEXELS", "1") == "1"

# Generic web-image fallbacks (Openverse, DuckDuckGo) are off by default.
# They carry no Commons extmetadata, so validate_candidate() cannot score them,
# and they answer a warship query with stock photos of anything vaguely
# adjacent -- a Midway video came out with ducks sitting in front of a flag.
# A Battle of Midway render came out containing a photograph of ducks sitting in
# front of a flag because these two providers were unfiltered and unvalidated.
ASSET_ENABLE_GENERIC = os.getenv("ASSET_ENABLE_GENERIC", "0") == "1"

# Per-entity result budget. A short video needs a handful of images; we keep a
# wider pool than we consume so the per-line relevance gate has something to
# choose from instead of being forced into the only candidate it was given.
COMMONS_PER_ENTITY = int(os.getenv("ASSET_COMMONS_PER_ENTITY", "6"))
# Commons likes short terms. A long term is both less likely to match a
# filename and more likely to be a whole phrase that no file uses.
MAX_ENTITY_LEN = 60
