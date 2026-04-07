"""strategies/ — Estrategias de trading del sistema."""
from .base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from .irreal        import IrrealStrategy
from .compuesto     import CompuestoStrategy
from .local_reversal import LocalReversalStrategy
from .grid_extremes   import GridExtremesStrategy

__all__ = [
    "BaseStrategy", "Signal", "SignalSide", "HOLD",
    "IrrealStrategy", "CompuestoStrategy",
    "LocalReversalStrategy", "GridExtremesStrategy"
]