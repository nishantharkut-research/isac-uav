"""Declarative paper-figure contracts with no result values.

These structures define which computation owns each curve. They deliberately
contain axes and provenance only; y-values must come from a simulation run.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CurveSpec:
    label: str
    method: str
    csi: Optional[str] = None
    hris_elements: Optional[int] = None
    active_fraction: Optional[float] = None


@dataclass(frozen=True)
class FigureSpec:
    number: int
    x_name: str
    x_values: Tuple[object, ...]
    curves: Tuple[CurveSpec, ...]
    note: str = ""


FIGURE_CONTRACTS = {
    4: FigureSpec(
        4, "Number of Episodes", tuple(range(0, 1500, 100)),
        (
            CurveSpec("DTL-pCSI", "DTL", "perfect", 40),
            CurveSpec("DTL-WFB", "DTL", "perfect", 40),
            CurveSpec("DTL-iCSI", "DTL", "imperfect", 40),
            CurveSpec("DDPG", "DDPG", None, 40),
        ),
        "The published markers span 0..1400 although Table IV reports 1000 episodes; the figure protocol uses 1500 episodes. The paper does not disclose the standalone DDPG comparator CSI, so its configured hypothesis is recorded in run metadata.",
    ),
    5: FigureSpec(
        5, "Number of Iterations", tuple(range(1, 21)),
        tuple(CurveSpec(f"L={value}", "AO_SURROGATE", "imperfect", value)
              for value in (30, 20, 10)),
        "Computed fixed-geometry SciPy AO surrogate; the author WMMSE/CVX/full-trajectory implementation is unavailable, and AO iterations are not treated as physical UAV slots.",
    ),
    6: FigureSpec(
        6, "P_T (dB)", (10, 15, 20, 25, 30),
        (
            CurveSpec("DTL-pCSI", "DTL", "perfect", 20),
            CurveSpec("DDPG", "DDPG", None, 20),
            CurveSpec("AO-pCSI", "AO_SURROGATE", "perfect", 20),
            CurveSpec("DTL-iCSI", "DTL", "imperfect", 20),
            CurveSpec("AO-iCSI", "AO_SURROGATE", "imperfect", 20),
            CurveSpec("DTL-ParCSI", "DTL", "partial", 20),
        ),
    ),
    7: FigureSpec(
        7, "Number of HRIS Elements", (20, 30, 40, 50, 60),
        (
            CurveSpec("HRIS-pCSI", "DTL", "perfect"),
            CurveSpec("HRIS-iCSI", "DTL", "imperfect"),
            CurveSpec("HRIS-ParCSI", "DTL", "partial"),
            CurveSpec("Passive RIS", "PASSIVE_DTL", "imperfect"),
            CurveSpec("Random Phase", "RANDOM_PHASE", "imperfect"),
            CurveSpec("No-RIS", "NO_RIS", "imperfect"),
        ),
        "ParCSI uses the same SCEE as iCSI with a fixed documented observation mask; the paper does not define its missing coefficients.",
    ),
    8: FigureSpec(
        8, "SCEE", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        (
            CurveSpec("HRIS, L=40", "DTL", hris_elements=40),
            CurveSpec("HRIS, L=20", "DTL", hris_elements=20),
            CurveSpec("Passive RIS, L=40", "PASSIVE_DTL", hris_elements=40),
            CurveSpec("Passive RIS, L=20", "PASSIVE_DTL", hris_elements=20),
        ),
    ),
    9: FigureSpec(
        9, "Number of HRIS Elements", (40, 30),
        tuple(CurveSpec(f"nu={nu}", "DTL", "imperfect", active_fraction=nu)
              for nu in (0.1, 0.3, 0.5)),
    ),
    10: FigureSpec(
        10, "Trajectory sample", (0,),
        (
            CurveSpec("HRIS case", "TRAJECTORY", "imperfect", 40),
            CurveSpec("Random phase", "TRAJECTORY", "imperfect", 40),
        ),
    ),
    11: FigureSpec(
        11, "Azimuth (degrees)", (-80, 0),
        (CurveSpec("iCSI beampattern", "BEAMPATTERN", "imperfect", 40),),
        "Published angular domain: azimuth -80..0 degrees, elevation 0..80 degrees.",
    ),
    12: FigureSpec(
        12, "Algorithm", ("DTL", "DDPG"),
        (
            CurveSpec("L=40", "MEASURED_RUNTIME", hris_elements=40),
            CurveSpec("L=30", "MEASURED_RUNTIME", hris_elements=30),
        ),
        "Measured local wall time is hardware-dependent; the paper gives no timing protocol.",
    ),
    13: FigureSpec(
        13, "Minimum target SINR", (0, 5, 10, 15, 20),
        (
            CurveSpec("R_u, L=40", "DTL", "imperfect", 40),
            CurveSpec("R_u, L=30", "DTL", "imperfect", 30),
            CurveSpec("R_u, L=20", "DTL", "imperfect", 20),
            CurveSpec("R_k, L=30", "DTL", "imperfect", 30),
            CurveSpec("R_k, L=20", "DTL", "imperfect", 20),
        ),
        "Eq. (14) full target SINR: each delta is enforced as R_k >= log2(1 + delta).",
    ),
    14: FigureSpec(
        14, "Minimum sensing threshold (dB)", (0, 5, 10, 15, 20),
        tuple(
            CurveSpec(f"{method}, L={elements}", method, "imperfect", elements)
            for method in ("DTL", "DDPG") for elements in (30, 20)
        ),
        "Remark 1 sensing-only SINR in dB; communication interference is excluded from this constraint.",
    ),
}
