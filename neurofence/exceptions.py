"""Shared exception hierarchy for NeuroFence.

Every module-specific error should subclass NeuroFenceError so callers
(CLI, API) can catch a single base type without hiding unrelated bugs.
"""


class NeuroFenceError(Exception):
    """Base class for all NeuroFence-raised errors."""


class AcquisitionError(NeuroFenceError):
    """Raised when a model cannot be safely or successfully acquired."""


class PathTraversalError(AcquisitionError):
    """Raised when a resolved path escapes its expected root directory."""


class UnsupportedFormatError(AcquisitionError):
    """Raised when a model file format is not supported or is unsafe to load."""


class IntegrityDiscrepancyError(AcquisitionError):
    """Raised for fatal manifest/hash mismatches during verification.

    Note: most integrity discrepancies are *reported as findings*, not raised
    as exceptions -- a hash mismatch is evidence, not proof of tampering, and
    the scan should continue and let evidence fusion weigh it. This exception
    is reserved for cases where verification cannot proceed at all (e.g. the
    referenced file no longer exists).
    """


class ConfigError(NeuroFenceError):
    """Raised for invalid or missing configuration."""
