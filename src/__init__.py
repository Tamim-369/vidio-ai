"""Video AI package.

Environment (.env) is loaded by the CLI entry point (src/cli/main.py) and by
each module that reads its own knobs via ``os.getenv``, so importing a submodule
directly (tests, one-off scripts) still sees .env.
"""