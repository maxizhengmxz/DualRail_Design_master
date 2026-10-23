"""Compatibility imports; shared physics now lives in core.electrostatics."""

from core.electrostatics import charge_coupling_ghz, exchange_j01_ghz

__all__ = ["charge_coupling_ghz", "exchange_j01_ghz"]
