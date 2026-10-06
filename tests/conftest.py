"""Test-session configuration.

PINNING THE EMBEDDING BACKEND IS NOT A CONVENIENCE, IT IS THE POINT.

`expertwins.diversity.embed` prefers a sentence-transformers encoder and falls
back to tf-idf. The two backends do not merely differ in precision -- they
differ in SCALE, which is why `guardrail.ABS_POSITIONS_FLOORS` is keyed on the
backend name. A unit test that asserts a dispersion value is therefore asserting
something about a specific backend, and a suite whose backend depends on what
happens to be installed on the machine is not a regression suite.

So the whole session runs on the calibrated tf-idf backend. That also makes the
suite hermetic: no model weights are read, nothing is downloaded, and the run
takes under a minute instead of reloading a torch graph hundreds of times.

To exercise the neural path deliberately, unset this in your own shell and run
the specific test you care about:

    EXPERTWINS_EMBED_MODEL=all-MiniLM-L6-v2 pytest tests/test_diversity.py
"""

from __future__ import annotations

import os

# Set at import time, not in a fixture: a test module that touches the backend
# while being collected would otherwise get the unpinned one.
os.environ.setdefault("EXPERTWINS_EMBED_MODEL", "__force_tfidf_fallback__")
