"""Network namespace for SparkSeg and the comparison architectures."""

from __future__ import annotations

from .source_code.SparkSeg import *  # noqa: F401,F403
from .source_code.SparkSeg import __all__ as _sparkseg_all

__all__ = list(_sparkseg_all)
