"""Stage-4 simulated exchange package (TEST mode only)."""

from btcc.execution.sim.broker import SimulatedBroker
from btcc.execution.sim.exchange import SimFill, SimOrder, SimulatedExchange
from btcc.execution.sim.scenarios import (
    SimBrokerError,
    SimNetworkError,
    SimRejectedError,
    SimScenario,
    SimTimeoutError,
)

__all__ = [
    "SimulatedBroker",
    "SimulatedExchange",
    "SimOrder",
    "SimFill",
    "SimScenario",
    "SimBrokerError",
    "SimTimeoutError",
    "SimNetworkError",
    "SimRejectedError",
]
