"""Canonical per-model network implementations (one folder per model family).

Layout and policy (see META.json in every family):

``<family>/model.py``
    The only implementation the trainers import. It is either
    * **T1** — a thin adapter over an official implementation (e.g. MONAI), or
    * **T2** — a vendored upstream implementation kept verbatim under ``upstream/``,
    * **T3** — an original implementation with an explicit provenance header.
``<family>/META.json``
    Machine-readable provenance: tier, upstream source + version/commit, license,
    interface differences, checkpoint compatibility, parity evidence.
``<family>/parity.py``
    Runnable equivalence check against the official implementation.

Rules
-----
1. One folder per model family; a family may expose several named variants, but the
   network code itself lives here once (no second implementation elsewhere).
2. T1/T2 must never re-implement upstream code: import it. Any deviation is listed
   in ``META.json`` as an interface difference, not as silent drift.
3. Old import paths stay alive as thin shims so trained checkpoints remain loadable
   (trainer classes are resolved by class name, not by module path).
"""
