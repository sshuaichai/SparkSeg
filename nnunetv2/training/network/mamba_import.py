"""Load mamba_ssm with the correct vendored fork for this repo.

The conda ``site-packages`` mamba_ssm is often a BiMamba build that
unconditionally ``assert bimamba_type == "v3"``. That breaks UMamba /
SegMamba uni checkpoints (empty ``AssertionError``) and also leaks across
trainers once cached in ``sys.modules``.

- ``mode="uni"``  → SegMamba vendored fork (BiMamba-v3 / ``bimamba_type``)
- ``mode="bi"``   → Slim-UNETRV2 Tri-SSA fork (``bimamba_type="v3"`` + ``nslices``)
- ``mode="site"`` → plain site-packages ``mamba_ssm`` (uni Mamba; for SegMamba-V2
  tri-axis outer scan — do **not** use SegMamba fork which asserts v3)

New ``mamba-ssm`` / ``causal-conv1d`` wheels may not expose top-level
``selective_scan_cuda`` / ``causal_conv1d_cuda`` the way SegMamba's 1.0.x
fork expects; ``_ensure_cuda_extension_aliases`` bridges that.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Optional, Type

_NET_DIR = os.path.dirname(os.path.abspath(__file__))
# vendored mamba forks now live under network/source_code/<family>/upstream/
_SEGMAMBA_MAMBA = os.path.join(_NET_DIR, "source_code", "segmamba", "upstream", "SegMamba-main", "mamba")
_SLIM_MAMBA = os.path.join(_NET_DIR, "source_code", "slim_unetrv2", "upstream", "environment", "mamba")
_VENDORED_MARKERS = ("/source_code/segmamba", "/source_code/segmamba_v2", "/source_code/slim_unetrv2")

_cached_mode: Optional[str] = None
_cached_cls: Optional[Type[Any]] = None


def _purge_mamba_ssm() -> None:
    for key in list(sys.modules):
        if key == "mamba_ssm" or key.startswith("mamba_ssm."):
            del sys.modules[key]


def _prefer_path(path: str) -> None:
    if not os.path.isdir(path):
        return
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _unprefer_path(path: str) -> None:
    while path in sys.path:
        sys.path.remove(path)


def _strip_vendored_mamba_paths() -> None:
    """Drop any sys.path entry that would shadow site-packages with a vendored fork."""
    for path in (_SEGMAMBA_MAMBA, _SLIM_MAMBA):
        _unprefer_path(path)
    doomed = [
        p
        for p in sys.path
        if isinstance(p, str)
        and any(m in p.replace("\\", "/") for m in _VENDORED_MARKERS)
        and p.endswith("mamba")
    ]
    for p in doomed:
        while p in sys.path:
            sys.path.remove(p)


def _ensure_cuda_extension_aliases() -> None:
    """Make top-level ``*_cuda`` imports work for SegMamba / Slim forks.

    The ``.so`` files need PyTorch's ``libc10.so`` already mapped; import
    ``torch`` first or bare ``import selective_scan_cuda`` fails with
    ``libc10.so: cannot open shared object file``.
    """
    import torch  # noqa: F401  # maps libc10 / libtorch into the process

    if "causal_conv1d_cuda" not in sys.modules:
        try:
            import causal_conv1d_cuda  # noqa: F401
        except ImportError:
            for modname in (
                "causal_conv1d.causal_conv1d_cuda",
                "causal_conv1d.causal_conv1d_interface",
            ):
                try:
                    mod = importlib.import_module(modname)
                except ImportError:
                    continue
                # Prefer a submodule that actually looks like the extension.
                if hasattr(mod, "causal_conv1d_fwd") or modname.endswith(
                    "causal_conv1d_cuda"
                ):
                    sys.modules["causal_conv1d_cuda"] = mod
                    break

    if "selective_scan_cuda" not in sys.modules:
        try:
            import selective_scan_cuda  # noqa: F401
        except ImportError:
            for modname in (
                "mamba_ssm.ops.selective_scan_cuda",
                "selective_scan_cuda",
            ):
                try:
                    sys.modules["selective_scan_cuda"] = importlib.import_module(
                        modname
                    )
                    break
                except ImportError:
                    continue


def _import_mamba() -> Type[Any]:
    _ensure_cuda_extension_aliases()
    try:
        from mamba_ssm import Mamba  # type: ignore

        return Mamba
    except ImportError:
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

        return Mamba


def get_mamba_class(*, mode: str = "uni") -> Type[Any]:
    """Return ``Mamba`` from the requested fork (reload on mode switch)."""
    global _cached_mode, _cached_cls
    if mode not in ("uni", "bi", "site"):
        raise ValueError(f"mode must be 'uni', 'bi', or 'site', got {mode!r}")

    if _cached_mode == mode and _cached_cls is not None:
        return _cached_cls

    if mode == "site":
        _strip_vendored_mamba_paths()
        _purge_mamba_ssm()
        # Prefer real site-packages ahead of cwd / repo paths.
        try:
            import site as _site

            for sp in reversed(_site.getsitepackages()):
                if sp in sys.path:
                    sys.path.remove(sp)
                sys.path.insert(0, sp)
        except Exception:
            pass
        try:
            Mamba = _import_mamba()
        except ImportError as exc:
            raise ImportError(
                "mamba_ssm site-packages required for mode='site' "
                "(SegMamba-V2). Install matching mamba-ssm / causal-conv1d."
            ) from exc
        mod_file = getattr(
            importlib.import_module(Mamba.__module__), "__file__", ""
        ) or ""
        mod_file_norm = os.path.abspath(mod_file).replace("\\", "/")
        if any(s in mod_file_norm for s in _VENDORED_MARKERS):
            raise ImportError(
                f"mode='site' still resolved vendored Mamba at {mod_file_norm}. "
                "Remove vendored mamba dirs from PYTHONPATH/sys.path."
            )
        _cached_mode = mode
        _cached_cls = Mamba
        return Mamba

    path = _SLIM_MAMBA if mode == "bi" else _SEGMAMBA_MAMBA
    vendored_ok = os.path.isdir(path)
    if not vendored_ok:
        import warnings

        warnings.warn(
            f"mamba_import: vendored fork missing at {path}; "
            "using site-packages mamba_ssm (may lack bimamba_type).",
            UserWarning,
            stacklevel=2,
        )

    _prefer_path(path)
    _purge_mamba_ssm()

    try:
        Mamba = _import_mamba()
        # Ensure we actually got the vendored module when the tree exists.
        if vendored_ok:
            mod = importlib.import_module(Mamba.__module__)
            mod_file = getattr(mod, "__file__", "") or ""
            if path not in os.path.abspath(mod_file).replace("\\", "/"):
                raise ImportError(
                    f"expected vendored Mamba under {path}, got {mod_file}"
                )
    except ImportError:
        # Vendored fork needs old top-level CUDA ext; fall back to site-packages.
        _unprefer_path(path)
        _purge_mamba_ssm()
        try:
            Mamba = _import_mamba()
        except ImportError as exc:
            raise ImportError(
                "mamba_ssm is required. Install mamba-ssm / causal-conv1d "
                "(matching torch/cuda/cp), or use the vendored forks under "
                "nnunetv2/training/network/. If wheels installed but SegMamba "
                "still fails, check: import torch; import selective_scan_cuda; "
                "import causal_conv1d_cuda"
            ) from exc
        if vendored_ok:
            import warnings

            warnings.warn(
                f"mamba_import: failed to load vendored fork at {path}; "
                "fell back to site-packages (bimamba_type may be unavailable).",
                UserWarning,
                stacklevel=2,
            )

    _cached_mode = mode
    _cached_cls = Mamba
    return Mamba
