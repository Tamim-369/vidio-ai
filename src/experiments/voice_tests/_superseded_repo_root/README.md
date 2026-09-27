# Superseded copies (recovered from the repo-root voice_tests/)

These are the original repo-root `voice_tests/` files, recovered verbatim from
git when that directory was consolidated into `src/experiments/voice_tests/`.

They are kept only as a record. **Do not edit or run these** — the live
versions one directory up are the ones to use:

| file               | what changed in the live version |
|--------------------|----------------------------------|
| `test_arnold.py`   | script/image flow -> quote-card flow (`--quotes N` replaces `--topic`) |
| `test_trump.py`    | same |
| `voice_test_utils.py` | dropped `ASSETS_DIR`, `_persist_assets`, `_placeholder_assets` — the per-line image/placeholder machinery that went away with the asset subsystem |

The other four root files (`test_chatterbox.py`, `test_pocket_tts.py`,
`test_qwen_0.6b_base.py`, `_logs/regen.log`) were byte-identical to the live
versions, so they were not duplicated here.
