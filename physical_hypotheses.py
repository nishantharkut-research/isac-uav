"""Explicit interpretations of the paper's reference-ambiguous power label."""

from dataclasses import dataclass
import math
from types import MappingProxyType


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def dbm_to_watts(value: float) -> float:
    """Convert dBm to watts."""
    return 10.0 ** ((_finite(value, "dBm value") - 30.0) / 10.0)


def dbw_to_watts(value: float) -> float:
    """Convert dBW to watts."""
    return 10.0 ** (_finite(value, "dBW value") / 10.0)


def watts_to_dbm(value: float) -> float:
    """Convert strictly positive watts to dBm."""
    value = _finite(value, "power in watts")
    if value <= 0.0:
        raise ValueError("power in watts must be finite and positive")
    return 10.0 * math.log10(value) + 30.0


def watts_to_dbw(value: float) -> float:
    """Convert strictly positive watts to dBW."""
    value = _finite(value, "power in watts")
    if value <= 0.0:
        raise ValueError("power in watts must be finite and positive")
    return 10.0 * math.log10(value)


@dataclass(frozen=True)
class PowerHypothesis:
    identifier: str
    display_unit: str
    reference_power_watts: float
    role: str
    rationale: str

    def __post_init__(self):
        if self.display_unit not in {"dBm", "dBW"}:
            raise ValueError("Power hypothesis display unit must be dBm or dBW")
        if self.role not in {"primary", "sensitivity"}:
            raise ValueError("Power hypothesis role must be primary or sensitivity")
        if self.reference_power_watts <= 0.0:
            raise ValueError("Power hypothesis reference must be positive")

    def to_watts(self, paper_label_db: float) -> float:
        paper_label_db = _finite(paper_label_db, "paper power label")
        return self.reference_power_watts * 10.0 ** (paper_label_db / 10.0)

    def as_dict(self, paper_label_db: float) -> dict:
        power_watts = self.to_watts(paper_label_db)
        return {
            "id": self.identifier,
            "paper_label_db": float(paper_label_db),
            "display_unit": self.display_unit,
            "reference_power_watts": self.reference_power_watts,
            "linear_power_watts": power_watts,
            "power_dbm": watts_to_dbm(power_watts),
            "power_dbw": watts_to_dbw(power_watts),
            "role": self.role,
            "rationale": self.rationale,
        }


POWER_HYPOTHESES = MappingProxyType({
    "paper_db_as_dbm": PowerHypothesis(
        identifier="paper_db_as_dbm",
        display_unit="dBm",
        reference_power_watts=1e-3,
        role="primary",
        rationale=(
            "Primary physically grounded interpretation of an RF transmit-power "
            "axis whose paper label omits the reference."
        ),
    ),
    "paper_db_as_dbw": PowerHypothesis(
        identifier="paper_db_as_dbw",
        display_unit="dBW",
        reference_power_watts=1.0,
        role="sensitivity",
        rationale=(
            "Sensitivity interpretation retained because the paper prints dB "
            "rather than dBm or dBW."
        ),
    ),
})


def get_power_hypothesis(identifier: str) -> PowerHypothesis:
    try:
        return POWER_HYPOTHESES[str(identifier)]
    except KeyError as error:
        raise ValueError(f"Unknown power hypothesis: {identifier}") from error


def apply_power_hypothesis(config, identifier: str, paper_label_db: float):
    """Apply one declared hypothesis and all derived display fields."""
    hypothesis = get_power_hypothesis(identifier)
    metadata = hypothesis.as_dict(paper_label_db)
    config.power_hypothesis_id = hypothesis.identifier
    config.PT_dB = metadata["paper_label_db"]
    config.PT_reference = (
        "1 mW" if hypothesis.reference_power_watts == 1e-3 else "1 W"
    )
    config.PT_display_unit = hypothesis.display_unit
    config.PT = metadata["linear_power_watts"]
    config.PT_dBm = metadata["power_dbm"]
    config.PT_dBW = metadata["power_dbw"]
    config.power_hypothesis_metadata = metadata
    return metadata
