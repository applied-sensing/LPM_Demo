from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt

from .plotting import bode_mag_phase
from .plotting import bode_2panel

from .config import load_config, build_network_from_config
from .post import (
    get_node_effort,
    get_element_flow,
    zin_from_v_and_i,
    spl_from_pressure,
)

from .metrics import estimate_f0_q_from_impedance

def save_png(fig, out_dir: Path, filename: str):
    out_path = (out_dir / filename).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)



def save_figure(path: Path, fig: plt.Figure) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="lpm",
        description="Lumped-parameter modeling (headphones): V -> p"
    )
    ap.add_argument("config", help="Path to JSON/YAML config")
    ap.add_argument("--out", default="out", help="Output directory for PNGs")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    net, params, freqs = build_network_from_config(cfg)
    result = net.build_and_solve(freqs=freqs, params=params)

    pnode = cfg.get("outputs", {}).get("p_front_node", "a_front")
    vin_elem = cfg.get("outputs", {}).get("input_current_element", "Vin")
    vin_amp = float(cfg.get("outputs", {}).get("Vin_for_Zin", 1.0))

    p_front = get_node_effort(result, pnode)
    i_in = get_element_flow(result, vin_elem, port=0)
    zin = zin_from_v_and_i(vin_amp, i_in)
    spl = spl_from_pressure(p_front)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Bode transfer functions
    # -------------------------------------------------

    Vin = vin_amp

    H_pV = p_front / (Vin + 1e-30)   # Pa/V
    H_Zin = zin                      # Ohm

    res = estimate_f0_q_from_impedance(
        freqs,
        H_Zin,
        Re=params.get("Re"),
        Le=params.get("Le"),
        search_min_hz=20.0,
        search_max_hz=800.0,
    )

    print(f"Impedance resonance: f0 = {res.f0_hz:.2f} Hz, Q = {res.q:.2f}  (f1={res.f1_hz:.2f} Hz, f2={res.f2_hz:.2f} Hz)")
    (out_dir / "impedance_resonance.txt").write_text(
    f"f0_hz: {res.f0_hz}\nQ: {res.q}\nf1_hz: {res.f1_hz}\nf2_hz: {res.f2_hz}\npeak_mag: {res.peak_mag}\nmethod: {res.method}\nnotes: {res.notes}\n",
    encoding="utf-8"
    )


    bode_mag_phase(
        freqs, H_pV,
        title="Front pressure transfer",
        mag_ylabel="|p_front/V| (Pa/V)",
        out_mag=out_dir / "bode_p_over_v_mag.png",
        out_phase=out_dir / "bode_p_over_v_phase.png",
        mag_db=True,
    )

    bode_mag_phase(
        freqs, H_Zin,
        title="Input impedance",
        mag_ylabel="|Zin| (Ohm)",
        out_mag=out_dir / "bode_zin_mag.png",
        out_phase=out_dir / "bode_zin_phase.png",
        mag_db=False,
    )

    bode_2panel(
        freqs, H_pV,
        title="Front pressure transfer (p_front / V)",
        mag_ylabel="|p_front/V| (Pa/V)",
        out_png=out_dir / "bode_p_over_v.png",
        mag_db=True,
    )

    bode_2panel(
        freqs, H_Zin,
        title="Input impedance (Zin)",
        mag_ylabel="|Zin| (Ohm)",
        out_png=out_dir / "bode_zin.png",
        mag_db=False,  # keep ohms linear
    )


    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Pressure transfer
    # -------------------------------------------------
    fig1 = plt.figure()
    plt.semilogx(freqs, 20*np.log10(np.abs(p_front) + 1e-30))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|p_front| [dB re 1 Pa/V]")
    plt.title("Front pressure transfer")
    plt.grid(True, which="both")
    save_png(fig1, out_dir, "p_front_transfer.png")
    

    # -------------------------------------------------
    # Input impedance
    # -------------------------------------------------
    fig2 = plt.figure()
    plt.semilogx(freqs, np.abs(zin))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|Zin| [Ohm]")
    plt.title("Electrical input impedance magnitude")
    plt.grid(True, which="both")
    save_png(fig2, out_dir, "zin_magnitude.png")

    # -------------------------------------------------
    # SPL
    # -------------------------------------------------
    fig3 = plt.figure()
    plt.semilogx(freqs, spl)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPL [dB re 20µPa]")
    plt.title("Relative SPL at front node")
    plt.grid(True, which="both")
    save_png(fig3, out_dir, "spl_relative.png")

    if not args.no_plots:
        plt.show()

    print(f"\nFigures written to: {out_dir.resolve()}")
