from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass(frozen=True)
class ResonanceResult:
    f0_hz: float
    q: float
    f1_hz: float
    f2_hz: float
    peak_mag: float
    method: str
    notes: str


def _interp_crossing(freqs: np.ndarray, y: np.ndarray, target: float, i0: int, i1: int) -> float:
    """
    Linear interpolation in log-frequency for crossing y == target
    between indices i0 and i1 (i0 < i1).
    """
    f0, f1 = freqs[i0], freqs[i1]
    y0, y1 = y[i0], y[i1]

    # avoid divide-by-zero; fallback to midpoint
    if y1 == y0:
        return float(np.sqrt(f0 * f1))

    # interpolate in log(f)
    x0, x1 = np.log(f0), np.log(f1)
    t = (target - y0) / (y1 - y0)
    x = x0 + t * (x1 - x0)
    return float(np.exp(x))


def estimate_f0_q_from_impedance(
    freqs_hz: np.ndarray,
    zin: np.ndarray,
    *,
    Re: Optional[float] = None,
    Le: Optional[float] = None,
    search_min_hz: float = 20.0,
    search_max_hz: float = 2000.0,
    prefer_lowest_peak: bool = True,
) -> ResonanceResult:
    """
    Estimate resonance frequency f0 and Q from the impedance curve.

    Recommended: provide Re and Le so we can remove the electrical series part:
        Zm(ω) = Zin(ω) - (Re + j ω Le)
    Then use |Zm| peak and -3 dB bandwidth.

    Q estimate:
        Q ≈ f0 / (f2 - f1)
    where f1,f2 are frequencies where |Zm| = peak/√2 around the peak.

    Notes:
    - Works best for a single dominant low-frequency resonance.
    - Heavy leakage/vents can flatten the peak and reduce accuracy.
    """

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    zin = np.asarray(zin, dtype=np.complex128)

    if freqs_hz.ndim != 1 or zin.ndim != 1 or freqs_hz.size != zin.size:
        raise ValueError("freqs_hz and zin must be 1D arrays of the same length")

    w = 2.0 * np.pi * freqs_hz

    # Motional impedance extraction
    Zcorr = zin.copy()
    notes = []
    if Re is not None:
        Zcorr = Zcorr - complex(Re)
        notes.append("subtracted Re")
    if Le is not None:
        Zcorr = Zcorr - 1j * w * float(Le)
        notes.append("subtracted jωLe")

    mag = np.abs(Zcorr)

    # Limit search range
    mask = (
        (freqs_hz >= float(search_min_hz)) &
        (freqs_hz <= float(search_max_hz))
    )

    if not np.any(mask):
        raise ValueError(
            f"No frequency points inside search range "
            f"[{search_min_hz}, {search_max_hz}] Hz"
        )

    f_s = freqs_hz[mask]
    mag_s = mag[mask]

    # Find peaks (simple approach: argmax; optionally choose lowest among multiple local maxima)
    # Start with global max
    idx_peak = int(np.argmax(mag_s))

    # if peak is at boundary, we cannot estimate bandwidth reliably
    if idx_peak <= 2 or idx_peak >= len(mag_s) - 3:
        return ResonanceResult(
            f0_hz=float(f_s[idx_peak]),
            q=float("nan"),
            f1_hz=float("nan"),
            f2_hz=float("nan"),
            peak_mag=float(mag_s[idx_peak]),
            method="motional|Z| peak + -3dB bandwidth",
            notes="Peak at/near search boundary. Narrow/shift search band or increase search_max_hz.",
        )

    if prefer_lowest_peak:
        # Find local maxima above some threshold and pick the lowest-frequency "dominant" one
        # threshold: 30% of global peak (tunable)
        thr = 0.30 * float(np.max(mag_s))
        candidates = []
        for i in range(1, len(mag_s) - 1):
            if mag_s[i] >= thr and mag_s[i] >= mag_s[i - 1] and mag_s[i] >= mag_s[i + 1]:
                candidates.append(i)
        if candidates:
            idx_peak = int(candidates[0])  # lowest-frequency candidate

    f0 = float(f_s[idx_peak])
    peak = float(mag_s[idx_peak])

    # -3 dB target (peak / sqrt(2))
    target = peak / np.sqrt(2.0)

    # Find left crossing
    i_left = None
    for i in range(idx_peak, 0, -1):
        if mag_s[i - 1] <= target <= mag_s[i] or mag_s[i] <= target <= mag_s[i - 1]:
            i_left = (i - 1, i)
            break

    # Find right crossing
    i_right = None
    for i in range(idx_peak, len(mag_s) - 1):
        if mag_s[i] >= target >= mag_s[i + 1] or mag_s[i] <= target <= mag_s[i + 1]:
            i_right = (i, i + 1)
            break

    if i_left is None or i_right is None:
        return ResonanceResult(
            f0_hz=f0,
            q=float("nan"),
            f1_hz=float("nan"),
            f2_hz=float("nan"),
            peak_mag=peak,
            method="motional|Z| peak + -3dB bandwidth",
            notes="Could not find -3dB crossings (peak too flat or search range too small). " + ", ".join(notes),
        )

    f1 = _interp_crossing(f_s, mag_s, target, i_left[0], i_left[1])
    f2 = _interp_crossing(f_s, mag_s, target, i_right[0], i_right[1])

    bw = f2 - f1
    q = f0 / bw if bw > 0 else float("nan")

    return ResonanceResult(
        f0_hz=f0,
        q=float(q),
        f1_hz=float(f1),
        f2_hz=float(f2),
        peak_mag=peak,
        method="motional|Z| peak + -3dB bandwidth",
        notes=", ".join(notes) if notes else "no correction applied",
    )
