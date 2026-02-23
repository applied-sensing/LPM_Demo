from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


def _prepare_path(p: Path) -> Path:
    p = Path(p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def bode_mag_phase(
    freqs_hz: np.ndarray,
    H: np.ndarray,
    *,
    title: str,
    mag_ylabel: str,
    out_mag: Path,
    out_phase: Path,
    mag_db: bool = True,
    phase_deg: bool = True,
    unwrap: bool = True,
) -> None:
    freqs_hz = np.asarray(freqs_hz)
    H = np.asarray(H)

    mag = np.abs(H)
    if mag_db:
        mag_plot = 20.0 * np.log10(mag + 1e-30)
        mag_unit = "dB"
    else:
        mag_plot = mag
        mag_unit = ""

    fig1 = plt.figure()
    plt.semilogx(freqs_hz, mag_plot)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(f"{mag_ylabel} [{mag_unit}]".rstrip())
    plt.title(f"{title} — Magnitude")
    plt.grid(True, which="both")
    fig1.tight_layout()
    out_mag = _prepare_path(out_mag)
    fig1.savefig(str(out_mag), dpi=200)
    plt.close(fig1)

    ph = np.angle(H)
    if unwrap:
        ph = np.unwrap(ph)
    if phase_deg:
        ph = np.degrees(ph)
        ph_unit = "deg"
    else:
        ph_unit = "rad"

    fig2 = plt.figure()
    plt.semilogx(freqs_hz, ph)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(f"Phase [{ph_unit}]")
    plt.title(f"{title} — Phase")
    plt.grid(True, which="both")
    fig2.tight_layout()
    out_phase = _prepare_path(out_phase)
    fig2.savefig(str(out_phase), dpi=200)
    plt.close(fig2)


def bode_2panel(
    freqs_hz: np.ndarray,
    H: np.ndarray,
    *,
    title: str,
    mag_ylabel: str,
    out_png: Path,
    mag_db: bool = True,
    phase_deg: bool = True,
    unwrap: bool = True,
) -> None:
    freqs_hz = np.asarray(freqs_hz)
    H = np.asarray(H)

    mag = np.abs(H)
    if mag_db:
        mag_plot = 20.0 * np.log10(mag + 1e-30)
        mag_unit = "dB"
    else:
        mag_plot = mag
        mag_unit = ""

    ph = np.angle(H)
    if unwrap:
        ph = np.unwrap(ph)
    if phase_deg:
        ph = np.degrees(ph)
        ph_unit = "deg"
    else:
        ph_unit = "rad"

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.0))

    ax1.semilogx(freqs_hz, mag_plot)
    ax1.set_ylabel(f"{mag_ylabel} [{mag_unit}]".rstrip())
    ax1.set_title(title)
    ax1.grid(True, which="both")

    ax2.semilogx(freqs_hz, ph)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel(f"Phase [{ph_unit}]")
    ax2.grid(True, which="both")

    fig.tight_layout()
    out_png = _prepare_path(out_png)
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
