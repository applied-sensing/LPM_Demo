from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

def get_node_effort(result: Dict[str, Any], node: str) -> np.ndarray:
    nodes = result["node_index"]
    if node not in nodes:
        raise KeyError(f"Node '{node}' not found. Known nodes: {list(nodes.keys())}")
    return result["node_efforts"][:, nodes[node]]

def get_element_flow(result: Dict[str, Any], element: str, port: int = 0) -> np.ndarray:
    bm = result["branch_map"]
    if element not in bm:
        raise KeyError(f"Element '{element}' not found. Known: {list(bm.keys())}")
    return result["branch_flows"][:, bm[element][port]]

def zin_from_v_and_i(vin: float, i_in: np.ndarray) -> np.ndarray:
    return vin / (i_in + 1e-30)

def spl_from_pressure(p: np.ndarray, pref: float = 20e-6) -> np.ndarray:
    return 20.0 * np.log10(np.abs(p) / pref + 1e-30)
