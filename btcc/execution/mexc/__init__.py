"""MEXC Spot package.

Stage 3: MexcReadOnlyClient (GET only).
Stage 7: MexcWriteClient + MexcSpotProtectionAdapter (gated writes; protection ASSUMPTION).
"""

from btcc.execution.mexc.client import MexcReadOnlyClient
from btcc.execution.mexc.credentials import (
    MexcCredentials,
    MissingCredentialsError,
    load_mexc_credentials,
    redact_secrets,
)

__all__ = [
    "MexcReadOnlyClient",
    "MexcCredentials",
    "MissingCredentialsError",
    "load_mexc_credentials",
    "redact_secrets",
]
