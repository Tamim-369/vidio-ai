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

ASSET_ENABLE_PEXELS = os.getenv("ASSET_ENABLE_PEXELS", "0") == "1"

# Per-entity result budget. A short video needs a handful of images; we keep a
# wider pool than we consume so the per-line relevance gate has something to
# choose from instead of being forced into the only candidate it was given.
COMMONS_PER_ENTITY = int(os.getenv("ASSET_COMMONS_PER_ENTITY", "6"))
# Commons likes short terms. A long term is both less likely to match a
# filename and more likely to be a whole phrase that no file uses.
MAX_ENTITY_LEN = 60
