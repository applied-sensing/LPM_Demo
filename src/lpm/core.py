from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


# ==============================================================
# Utilities
# ==============================================================

def logspace(fmin: float, fmax: float, n: int) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


def safe_eval(expr: str, params: Dict[str, float]) -> float:
    """
    Evaluate simple numeric expressions from config.
    Allowed: numbers, params, + - * / **, parentheses, pi, e.
    """
    allowed = {
        "__builtins__": {},
        "pi": math.pi,
        "e": math.e,
        **params,
    }
    return float(eval(expr, allowed, {}))


def value_number(v: Any, params: Dict[str, float]) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if v in params:
            return float(params[v])
        return safe_eval(v, params)
    raise TypeError(f"Unsupported value type: {type(v)}")


# ==============================================================
# Data structures
# ==============================================================

@dataclass(frozen=True)
class Port:
    n1: str
    n2: str


@dataclass
class Branch:
    name: str
    kind: str
    ports: List[Port]
    value: Any
    meta: Dict[str, Any]


# ==============================================================
# Network Solver (Tableau formulation)
# ==============================================================

class LPMNetwork:
    """
    Effort/flow tableau solver.

    Unknown vector x = [node_efforts (excluding ground), branch_flows]

    Domains share mathematical structure:
        effort  = V (elec), F (mech), p (ac)
        flow    = I (elec), v (mech), U (ac)

    Each element stamps:
        - KCL equations
        - Constitutive equations
    """

    def __init__(self, ground: str = "0"):
        self.ground = ground
        self.branches: List[Branch] = []
        self._nodes: Dict[str, int] = {}

    # ----------------------------------------------------------

    def add_branch(self, br: Branch) -> None:
        self.branches.append(br)

    # ----------------------------------------------------------

    def _register_node(self, node: str) -> None:
        if node == self.ground:
            return
        if node not in self._nodes:
            self._nodes[node] = len(self._nodes)

    # ----------------------------------------------------------

    def finalize(self) -> None:
        for br in self.branches:
            for p in br.ports:
                self._register_node(p.n1)
                self._register_node(p.n2)

    # ----------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def branch_flow_count(self) -> int:
        return sum(len(br.ports) for br in self.branches)

    # ----------------------------------------------------------

    def node_index(self, node: str) -> Optional[int]:
        return None if node == self.ground else self._nodes[node]

    # ==========================================================
    # SOLVER
    # ==========================================================

    def build_and_solve(
        self,
        freqs: np.ndarray,
        params: Dict[str, float],
        source_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        self.finalize()

        Nn = self.node_count
        Nf = self.branch_flow_count
        Nx = Nn + Nf

        node_efforts = np.zeros((len(freqs), Nn), dtype=np.complex128)
        branch_flows = np.zeros((len(freqs), Nf), dtype=np.complex128)

        # Precompute flow index mapping
        branch_flow_start: List[int] = []
        acc = 0
        for br in self.branches:
            branch_flow_start.append(acc)
            acc += len(br.ports)

        # ------------------------------------------------------

        for k, f in enumerate(freqs):
            w = 2.0 * np.pi * f

            A = np.zeros((Nx, Nx), dtype=np.complex128)
            b = np.zeros((Nx,), dtype=np.complex128)

            # -----------------------------------------
            # 1) KCL at nodes
            # -----------------------------------------
            global_flow = 0
            for br in self.branches:
                for p in br.ports:
                    fi = Nn + global_flow

                    n1i = self.node_index(p.n1)
                    n2i = self.node_index(p.n2)

                    if n1i is not None:
                        A[n1i, fi] += 1.0
                    if n2i is not None:
                        A[n2i, fi] -= 1.0

                    global_flow += 1

            # -----------------------------------------
            # 2) Element equations
            # -----------------------------------------
            row = Nn

            for bi, br in enumerate(self.branches):
                kind = br.kind.lower()
                start = branch_flow_start[bi]

                # -------------------------------------------------
                # 1-port impedance elements
                # -------------------------------------------------
                if kind in (
                    "r", "l", "c",
                    "mass", "damper", "spring",
                    "acompliance", "ainertance", "aresistance",
                    # mobility mechanical variants
                    "mass_mob", "mass_mobility",
                    "damper_mob", "damper_mobility",
                    "spring_mob", "compliance_mob", "cms_mob"
                ):
                    assert len(br.ports) == 1
                    p0 = br.ports[0]
                    fi = Nn + start

                    stamp_effort_drop(A, row, self, p0)

                    Z = impedance_Z(kind, br.value, w, params)
                    A[row, fi] += -Z

                    row += 1

                # -------------------------------------------------
                # Voltage source
                # -------------------------------------------------
                elif kind in ("vsrc", "voltage_source"):
                    assert len(br.ports) == 1
                    p0 = br.ports[0]

                    stamp_effort_drop(A, row, self, p0)

                    Vs = value_number(br.value, params)
                    if source_overrides and br.name in source_overrides:
                        Vs = value_number(source_overrides[br.name], params)

                    b[row] += Vs
                    row += 1

                # -------------------------------------------------
                # Gyrator (Bl)
                # -------------------------------------------------
                elif kind in ("gyrator_bl", "gyrator"):
                    assert len(br.ports) == 2

                    pe = br.ports[0]  # electrical
                    pm = br.ports[1]  # mechanical

                    fe = Nn + (start + 0)
                    fm = Nn + (start + 1)

                    Bl = value_number(br.value, params)

                    # Power-conserving gyrator (moving-coil):
                    #   e_elec = +Bl * f_mech
                    #   e_mech = -Bl * f_elec

                    # Alternative skew-symmetric form

                    # e_elec + Bl * f_mech = 0
                    stamp_effort_drop(A, row, self, pe)
                    A[row, fm] += +Bl
                    row += 1

                    # e_mech - Bl * f_elec = 0
                    stamp_effort_drop(A, row, self, pm)
                    A[row, fe] += -Bl
                    row += 1


                # -------------------------------------------------
                # Transformer (Bl) for mobility analogy
                # -------------------------------------------------
                elif kind in ("xfmr_bl", "transformer_bl"):
                    assert len(br.ports) == 2

                    pe = br.ports[0]  # electrical port (effort=V, flow=I)
                    pm = br.ports[1]  # mechanical port (effort=v, flow=F)  <-- mobility!

                    fe = Nn + (start + 0)   # electrical flow variable (I)
                    fm = Nn + (start + 1)   # mechanical flow variable (F)

                    Bl = value_number(br.value, params)

                    # Mobility transformer:
                    #   V = Bl * v
                    #   F = Bl * I
                    #
                    # Eq1: e_elec - Bl * e_mech = 0
                    stamp_effort_drop(A, row, self, pe)                 # V
                    stamp_effort_drop_scaled(A, row, self, pm, scale=-Bl)  # -Bl*v
                    row += 1

                    # Eq2: f_mech - Bl * f_elec = 0
                    A[row, fm] += 1.0
                    A[row, fe] += -Bl
                    row += 1


                # -------------------------------------------------
                # Transformer S (mechanical <-> acoustic)
                # -------------------------------------------------
                elif kind in ("xfmr_s", "transformer_s"):
                    assert len(br.ports) == 2

                    pm = br.ports[0]  # mechanical
                    pa = br.ports[1]  # acoustic

                    fm = Nn + (start + 0)
                    fa = Nn + (start + 1)

                    S = value_number(br.value, params)

                    # Eq1: e_mech - S * e_ac = 0
                    stamp_effort_drop(A, row, self, pm)
                    stamp_effort_drop_scaled(A, row, self, pa, scale=-S)
                    row += 1

                    # Eq2: f_ac - S * f_mech = 0
                    A[row, fa] += 1.0
                    A[row, fm] += -S
                    row += 1

                else:
                    raise ValueError(f"Unknown element kind: {br.kind}")

            if row != Nx:
                raise RuntimeError("System assembly mismatch.")

            x = np.linalg.solve(A, b)

            node_efforts[k, :] = x[:Nn]
            branch_flows[k, :] = x[Nn:]

        return {
            "freqs": freqs,
            "node_efforts": node_efforts,
            "branch_flows": branch_flows,
            "node_index": dict(self._nodes),
            "branch_map": self._branch_flow_map(),
        }

    # ----------------------------------------------------------

    def _branch_flow_map(self) -> Dict[str, List[int]]:
        m: Dict[str, List[int]] = {}
        idx = 0
        for br in self.branches:
            m[br.name] = list(range(idx, idx + len(br.ports)))
            idx += len(br.ports)
        return m


# ==============================================================
# Element impedance models
# ==============================================================

def impedance_Z(kind: str, value: Any, w: float, params: Dict[str, float]) -> complex:
    kind = kind.lower()

    # --- Mobility analogy mechanical elements (effort=v, flow=F) ---
    if kind in ("mass_mob", "mass_mobility"):
        M = value_number(value, params)
        return 1.0 / (1j * w * M + 1e-30)

    if kind in ("damper_mob", "damper_mobility"):
        R = value_number(value, params)
        return 1.0 / (R + 1e-30)

    if kind in ("spring_mob", "compliance_mob", "cms_mob"):
        Cms = value_number(value, params)
        return 1j * w * Cms

    # --- Existing default impedance-style elements ---
    if kind in ("r", "damper", "aresistance"):
        R = value_number(value, params)
        return complex(R)

    if kind in ("l", "mass", "ainertance"):
        L = value_number(value, params)
        return 1j * w * L

    if kind in ("c", "spring", "acompliance"):
        C = value_number(value, params)
        return 1.0 / (1j * w * C + 1e-30)

    raise ValueError(f"Unknown impedance kind: {kind}")




# ==============================================================
# Matrix stamping helpers
# ==============================================================

def stamp_effort_drop(A: np.ndarray, row: int, net: LPMNetwork, p: Port) -> None:
    n1i = net.node_index(p.n1)
    n2i = net.node_index(p.n2)

    if n1i is not None:
        A[row, n1i] += 1.0
    if n2i is not None:
        A[row, n2i] -= 1.0


def stamp_effort_drop_scaled(A: np.ndarray, row: int, net: LPMNetwork, p: Port, scale: complex) -> None:
    n1i = net.node_index(p.n1)
    n2i = net.node_index(p.n2)

    if n1i is not None:
        A[row, n1i] += scale
    if n2i is not None:
        A[row, n2i] -= scale
