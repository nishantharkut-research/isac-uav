"""Machine-readable provenance for reported and omitted paper parameters."""
from dataclasses import dataclass
from typing import Any, Optional, Tuple


EVIDENCE_CLASSES = {
    "journal_reported",
    "figure_inferred",
    "cited_reference",
    "assumption",
}


@dataclass(frozen=True)
class EvidenceEntry:
    name: str
    evidence_class: str
    value: Any
    bounds: Optional[Tuple[float, float]]
    unit: str
    source: str
    applicability: str
    confidence: str
    notes: str = ""

    def __post_init__(self):
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"Unknown evidence class: {self.evidence_class}")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError(f"Unknown confidence: {self.confidence}")
        if self.bounds is not None and self.bounds[0] > self.bounds[1]:
            raise ValueError(f"Invalid evidence bounds for {self.name}")


def _entry(name, evidence_class, value, bounds, unit, source, applicability,
           confidence, notes=""):
    return EvidenceEntry(
        name=name,
        evidence_class=evidence_class,
        value=value,
        bounds=bounds,
        unit=unit,
        source=source,
        applicability=applicability,
        confidence=confidence,
        notes=notes,
    )


JOURNAL = (
    "Local journal PDF, Saikia et al., IEEE Transactions on Communications "
    "73(9), 2025, DOI 10.1109/TCOMM.2025.3548797"
)


PARAMETER_EVIDENCE = {
    "M": _entry("M", "journal_reported", 4, (4, 4), "antennas",
                f"{JOURNAL}, Table III, page 10", "baseline system", "high"),
    "N": _entry("N", "journal_reported", 2, (2, 2), "antennas/node",
                f"{JOURNAL}, Table III, page 10", "baseline system", "high"),
    "U": _entry("U", "journal_reported", 3, (3, 3), "UEs",
                f"{JOURNAL}, Table III, page 10", "baseline system", "high"),
    "K": _entry("K", "journal_reported", 3, (3, 3), "targets",
                f"{JOURNAL}, Table III, page 10", "baseline system", "high"),
    "L": _entry("L", "journal_reported", 40, (40, 40), "elements",
                f"{JOURNAL}, Table III, page 10", "baseline system", "high"),
    "La": _entry("La", "journal_reported", 20, (20, 20), "active elements",
                 f"{JOURNAL}, simulation text, page 10", "baseline system", "high"),
    "alpha_max_dB": _entry(
        "alpha_max_dB", "journal_reported", 10, (10, 10), "dB amplitude gain",
        f"{JOURNAL}, simulation text, page 10", "active HRIS", "high"),
    "sigma2_u_dBm": _entry(
        "sigma2_u_dBm", "journal_reported", -94, (-94, -94), "dBm",
        f"{JOURNAL}, Table III, page 10", "UE receiver noise", "high"),
    "sigma2_k_dBm": _entry(
        "sigma2_k_dBm", "assumption", -94, (-94, -70), "dBm",
        f"{JOURNAL}, Table III and Eq. (14), pages 5 and 10",
        "target receiver noise in Eq. (14)",
        "low",
        "Primary: the table description assigns -70 dBm to active noise, so the unreported target receiver uses the only reported receiver floor (-94 dBm). The printed-symbol -70 dBm reading is an executable sensitivity branch."),
    "sigma2_a_dBm": _entry(
        "sigma2_a_dBm", "assumption", -70, (-70, -70), "dBm",
        f"{JOURNAL}, Table III description and Eqs. (12), (14)",
        "active-amplifier noise", "low",
        "The Table III description directly supports this mapping. Its printed sigma_k symbol is treated as a documented conflict, not silently overwritten."),
    "sigma2_qa_dBm": _entry(
        "sigma2_qa_dBm", "assumption", -70, (-70, -70), "dBm",
        f"{JOURNAL}, Table III, page 10", "deprecated metadata alias", "low",
        "Compatibility evidence only; equations use separate sigma2_k and sigma2_a."),
    "PT_dB": _entry(
        "PT_dB", "journal_reported", 10, (10, 10),
        "paper-reported dB label (reference unspecified)",
        f"{JOURNAL}, Table III, page 10", "baseline transmit-power label", "high",
        "The reference is selected by an explicit power hypothesis; Fig. 6 uses the same ambiguous label."),
    "ue_region_size_m": _entry(
        "ue_region_size_m", "journal_reported", (100, 100), None, "m x m",
        f"{JOURNAL}, simulation text, page 10", "UE sampling region", "high"),
    "service_area_m": _entry(
        "service_area_m", "figure_inferred", (300, 200), None, "m x m",
        f"{JOURNAL}, Fig. 10 axes, page 12", "UAV/target plotting domain", "medium",
        "The figure supports the extent but does not tabulate domain bounds."),
    "z_UAV": _entry(
        "z_UAV", "figure_inferred", 50.0, (50.0, 50.0), "m",
        f"{JOURNAL}, Fig. 10 z-axis and trajectory plane, page 12",
        "fixed UAV altitude", "medium",
        "Visible in the figure but not tabulated in Tables III-IV."),
    "fc": _entry(
        "fc", "assumption", 3e9, (1e9, 6e9), "Hz",
        "Not reported by the journal", "carrier and wavelength", "low"),
    "d_spacing": _entry(
        "d_spacing", "cited_reference", 0.5, (0.25, 1.0), "wavelengths",
        "Journal reference [40], arXiv:2203.05902, half-wavelength spacing",
        "normalized ULA/UPA element spacing", "medium",
        "Compatible cited model; not a journal-reported simulation value."),
    "omega_0": _entry(
        "omega_0", "assumption", 1e-3, (1e-5, 5e-2), "linear power gain at 1 m",
        "Not reported by the journal", "Eq. (7)-(8) reference gain", "low"),
    "rho_br": _entry(
        "rho_br", "assumption", 10.0, (0.1, 30.0), "linear K-factor",
        "Not reported by the journal", "Eq. (7) BS-HRIS Rician channel", "low"),
    "rho_rt": _entry(
        "rho_rt", "assumption", 5.0, (0.1, 30.0), "linear K-factor",
        "Not reported by the journal", "Eq. (8) HRIS-ground Rician channel", "low"),
    "a_env": _entry(
        "a_env", "cited_reference", 9.61, (4.0, 15.0), "dimensionless",
        "Al-Hourani et al. urban LoS model; compatible convention",
        "LoS probability", "medium", "Paper symbols a and b are ordered differently."),
    "b_env": _entry(
        "b_env", "cited_reference", 0.16, (0.05, 0.5), "per degree",
        "Al-Hourani et al. urban LoS model; compatible convention",
        "LoS probability", "medium", "Paper symbols a and b are ordered differently."),
    "zeta_los": _entry(
        "zeta_los", "assumption", 2.0, (1.8, 3.0), "path-loss exponent",
        "Eq. (5) constants are not reported by the journal",
        "P_LoS=1 endpoint of aerial path loss", "low",
        "Endpoint parameterization: paper zeta_e equals zeta_los-zeta_nlos."),
    "zeta_nlos": _entry(
        "zeta_nlos", "assumption", 3.5, (2.1, 5.5), "path-loss exponent",
        "Eq. (5) constants are not reported by the journal",
        "P_LoS=0 endpoint of aerial path loss", "low",
        "Endpoint parameterization: this value equals the paper's zeta_0."),
    "R_min": _entry(
        "R_min", "assumption", 0.5, (1e-6, 4.0), "bits/s/Hz per target",
        "Eq. (16e) defines the symbol; no numeric value is reported",
        "target-rate constraint", "low",
        "The positive 1e-6 search floor is numerical, not paper evidence."),
    "R_u_min": _entry(
        "R_u_min", "assumption", 0.5, (1e-6, 4.0), "bits/s/Hz per UE",
        "The supplied extension PNG defines the symbol but gives no numeric value",
        "extension UE-rate constraint", "low",
        "The primary extension assumption equals the 0.5 target-rate floor; sensitivity runs may override it explicitly."),
    "Psi_max": _entry(
        "Psi_max", "assumption", 1.0, (1e-20, 1.0), "linear cascade power",
        "Eq. (16c) defines the symbol; no numeric value is reported",
        "passive-cascade constraint", "low"),
    "P_a_max": _entry(
        "P_a_max", "assumption", 10.0, (1e-12, 100.0), "W",
        "Eq. (16d) defines the symbol; no numeric value is reported",
        "aggregate active-HRIS power", "low"),
    "ue_x_min": _entry(
        "ue_x_min", "assumption", 0.0, (0.0, 200.0), "m",
        "UE-region origin is not reported", "100 m UE square origin", "low"),
    "ue_x_max": _entry(
        "ue_x_max", "assumption", 100.0, (100.0, 300.0), "m",
        "Derived from assumed x origin plus reported 100 m width",
        "100 m UE square boundary", "low"),
    "ue_y_min": _entry(
        "ue_y_min", "assumption", 0.0, (0.0, 100.0), "m",
        "UE-region origin is not reported", "100 m UE square origin", "low"),
    "ue_y_max": _entry(
        "ue_y_max", "assumption", 100.0, (100.0, 200.0), "m",
        "Derived from assumed y origin plus reported 100 m width",
        "100 m UE square boundary", "low"),
    "target_speed_mps": _entry(
        "target_speed_mps", "assumption", 1.0, (0.0, 5.0), "m/s",
        "Paper states low mobility but gives no speed", "target motion", "low"),
    "v_max": _entry(
        "v_max", "assumption", 10.0, (0.1, 30.0), "m/s",
        "Eq. (2) defines the speed bound but gives no numeric value",
        "UAV displacement constraint", "low"),
    "random_seed": _entry(
        "random_seed", "assumption", 2025, (0, 2147483647), "integer",
        "Not reported by the journal", "reproducible random streams", "low"),
    "source_scene_seed": _entry(
        "source_scene_seed", "assumption", 2025, (0, 2147483647), "integer",
        "Source and target domains are not numerically defined",
        "transfer-learning source task", "low"),
    "target_scene_seed": _entry(
        "target_scene_seed", "assumption", 2026, (0, 2147483647), "integer",
        "Source and target domains are not numerically defined",
        "transfer-learning target task", "low"),
    "actor_hidden_widths": _entry(
        "actor_hidden_widths", "assumption", (400, 300), None, "neurons",
        "Network architecture is not reported", "DDPG actor", "low"),
    "critic_hidden_widths": _entry(
        "critic_hidden_widths", "assumption", (400, 300), None, "neurons",
        "Network architecture is not reported", "DDPG critic", "low"),
    "rho_nmse": _entry(
        "rho_nmse", "figure_inferred", 0.1, (0.0, 1.0), "normalized MSE",
        f"{JOURNAL}, Fig. 8 axis; baseline point is not specified",
        "imperfect-CSI experiments", "medium"),
    "partial_csi_fraction": _entry(
        "partial_csi_fraction", "assumption", 0.5, (0.05, 0.95), "fraction",
        "ParCSI coefficients are not defined by the journal",
        "agent observation mask", "low"),
    "comm_power_frac": _entry(
        "comm_power_frac", "assumption", 0.8, (0.05, 0.95), "fraction",
        "Communication/sensing initialization is not reported",
        "beamformer search initialization", "low"),
    "tau": _entry(
        "tau", "assumption", 0.01, (1e-4, 0.2), "fraction/update",
        "Table IV prints {0,1}, not a usable scalar",
        "DDPG soft target update", "low"),
    "noise_std": _entry(
        "noise_std", "assumption", 0.3, (0.0, 1.0), "normalized action units",
        "Exploration process is not reported", "DDPG exploration", "low"),
    "noise_decay": _entry(
        "noise_decay", "assumption", 0.999, (0.9, 1.0), "per episode",
        "Exploration schedule is not reported", "DDPG exploration", "low"),
    "upsilon_0": _entry(
        "upsilon_0", "assumption", 0.7, (0.0, 1.0), "fraction",
        "Eq. (30) defines upsilon but gives no schedule", "initial transfer weight", "low"),
    "upsilon_min": _entry(
        "upsilon_min", "assumption", 0.1, (0.0, 1.0), "fraction",
        "Transfer floor is not reported", "transfer schedule", "low"),
    "upsilon_decay": _entry(
        "upsilon_decay", "assumption", 0.001, (0.0, 0.1), "per episode",
        "Transfer schedule is not reported", "transfer schedule", "low"),
}


# Only omitted, ambiguous, or figure-inferred values can be overridden by a
# profile. Journal-reported dimensions and table constants remain code locks.
PROFILE_KEYS = frozenset({
    "fc", "d_spacing", "omega_0", "rho_br", "rho_rt", "a_env", "b_env",
    "zeta_los", "zeta_nlos", "R_min", "R_u_min", "Psi_max", "P_a_max",
    "ue_x_min", "ue_x_max", "ue_y_min", "ue_y_max", "target_speed_mps",
    "v_max", "rho_nmse", "partial_csi_fraction", "comm_power_frac", "tau",
    "noise_std", "noise_decay", "upsilon_0", "upsilon_min",
    "upsilon_decay",
})


# These omitted values have a no-training structural criterion and are selected
# automatically. Remaining profile keys need training/sensitivity evidence and
# must not be called "identified" merely because a default was convenient.
STRUCTURAL_IDENTIFICATION_KEYS = frozenset({
    "omega_0", "rho_br", "rho_rt", "zeta_los", "zeta_nlos",
    "ue_x_min", "ue_x_max", "ue_y_min", "ue_y_max",
    "R_min", "Psi_max", "P_a_max",
})
SENSITIVITY_ONLY_KEYS = PROFILE_KEYS - STRUCTURAL_IDENTIFICATION_KEYS


PROFILE_BOUNDS = {
    key: PARAMETER_EVIDENCE[key].bounds for key in PROFILE_KEYS
}
