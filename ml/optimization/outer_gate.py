from __future__ import annotations

from dataclasses import dataclass, replace

from ml.validation.split import ValidatedFoldBundle


class OuterValidationNotFrozenError(RuntimeError):
    """Raised when Outer Validation is requested before P6 configuration is frozen."""


class OuterValidationNotOpenError(RuntimeError):
    """Raised when Outer Validation is requested before the P6 gate is opened."""


@dataclass(frozen=True)
class OuterValidationGate:
    """Explicit dependency boundary between inner P6 optimization and Outer Validation."""

    _bundle: ValidatedFoldBundle
    _frozen: bool = False
    _opened: bool = False

    def freeze(self) -> "OuterValidationGate":
        """Freeze the selected P6 configuration without exposing Outer Validation."""
        if self._opened:
            raise RuntimeError("Cannot freeze configuration after Outer Validation is open.")
        return replace(self, _frozen=True)

    def open(self) -> "OuterValidationGate":
        """Open Outer Validation only after the configuration freeze point."""
        if not self._frozen:
            raise OuterValidationNotFrozenError("Outer Validation cannot open before P6 configuration is frozen.")
        return replace(self, _opened=True)

    def get_bundle(self) -> ValidatedFoldBundle:
        """Return Outer Validation data only after the gate has been opened."""
        if not self._opened:
            raise OuterValidationNotOpenError("Outer Validation is unavailable until the P6 gate is opened.")
        return self._bundle
