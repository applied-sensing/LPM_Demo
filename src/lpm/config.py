from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from .core import LPMNetwork, Branch, Port, logspace

def load_config(path: str) -> Dict[str, Any]:
    if path.lower().endswith(".json"):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if path.lower().endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError("YAML config requested but PyYAML not installed. pip install pyyaml") from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ValueError("Config must be .json or .yaml/.yml")

def build_network_from_config(cfg: Dict[str, Any]) -> Tuple[LPMNetwork, Dict[str, float], np.ndarray]:
    params = {k: float(v) for k, v in cfg.get("params", {}).items()}

    fcfg = cfg.get("freq", {})
    fmin = float(fcfg.get("fmin", 20.0))
    fmax = float(fcfg.get("fmax", 20000.0))
    n = int(fcfg.get("n", 2000))
    spacing = str(fcfg.get("spacing", "log")).lower()
    freqs = logspace(fmin, fmax, n) if spacing == "log" else np.linspace(fmin, fmax, n)

    net = LPMNetwork(ground=str(cfg.get("ground", "0")))

    for e in cfg.get("elements", []):
        name = str(e["name"])
        kind = str(e["type"])
        meta = {k: v for k, v in e.items() if k not in ("name", "type", "ports", "value", "n1", "n2")}

        ports = []
        if "ports" in e:
            for p in e["ports"]:
                ports.append(Port(str(p["n1"]), str(p["n2"])))
        else:
            ports.append(Port(str(e["n1"]), str(e["n2"])))

        value = e.get("value", 0.0)
        net.add_branch(Branch(name=name, kind=kind, ports=ports, value=value, meta=meta))

    return net, params, freqs
