"""Leaf module for names shared across ``estimation`` (no intra-package imports).

Keeping ``COMPOSITE_NAME`` here lets :mod:`estimation.gate_table` and
:mod:`estimation.composite` both use it without importing each other.
"""

from __future__ import annotations

__all__ = ["COMPOSITE_NAME"]

#: The only learning quantity reported from behaviour (S1-IMP-18).
COMPOSITE_NAME = "precision_weighted_learning_composite"
