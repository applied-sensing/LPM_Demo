"""
src/lpm/gui.py  —  Streamlit GUI for the LPM headphone modeler.

Run with:
    streamlit run src/lpm/gui.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="LPM Headphone Modeler",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── local imports (works when launched from repo root) ────────────────────────
try:
    from lpm.config import build_network_from_config
    from lpm.post import get_node_effort, get_element_flow, zin_from_v_and_i, spl_from_pressure
    from lpm.metrics import estimate_f0_q_from_impedance
    from lpm.core import logspace
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lpm.config import build_network_from_config
    from lpm.post import get_node_effort, get_element_flow, zin_from_v_and_i, spl_from_pressure
    from lpm.metrics import estimate_f0_q_from_impedance
    from lpm.core import logspace

# ── element catalog ───────────────────────────────────────────────────────────
CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "Electrical": [
        {"type": "vsrc",    "label": "Voltage Source",       "ports": 1, "default": "1.0",       "unit": "V"},
        {"type": "R",       "label": "Resistor",             "ports": 1, "default": "8.0",       "unit": "Ω"},
        {"type": "L",       "label": "Inductor",             "ports": 1, "default": "80e-6",     "unit": "H"},
        {"type": "C",       "label": "Capacitor",            "ports": 1, "default": "1e-6",      "unit": "F"},
    ],
    "Transducer": [
        {"type": "xfmr_Bl", "label": "Transformer Bl (mob)", "ports": 2, "default": "3.4",       "unit": "T·m"},
        {"type": "gyrator", "label": "Gyrator Bl (imp)",     "ports": 2, "default": "3.4",       "unit": "T·m"},
        {"type": "xfmr_S",  "label": "Transformer Sd",       "ports": 2, "default": "2.8e-4",    "unit": "m²"},
    ],
    "Mechanical (mobility)": [
        {"type": "mass_mob",   "label": "Mass",       "ports": 1, "default": "0.18e-3", "unit": "kg"},
        {"type": "damper_mob", "label": "Damper",     "ports": 1, "default": "0.6",     "unit": "N·s/m"},
        {"type": "spring_mob", "label": "Compliance", "ports": 1, "default": "0.35e-3", "unit": "m/N"},
    ],
    "Acoustic": [
        {"type": "acompliance", "label": "Compliance", "ports": 1, "default": "3e-12", "unit": "m³/Pa"},
        {"type": "ainertance",  "label": "Inertance",  "ports": 1, "default": "1e6",   "unit": "kg/m⁴"},
        {"type": "aresistance", "label": "Resistance", "ports": 1, "default": "1e6",   "unit": "Pa·s/m³"},
    ],
}

TWO_PORT_TYPES = {"xfmr_Bl", "xfmr_bl", "transformer_bl", "xfmr_S", "transformer_s", "gyrator", "gyrator_bl"}

ALL_TYPES = {e["type"]: e for cat in CATALOG.values() for e in cat}

# ── default headphone_min network ─────────────────────────────────────────────
DEFAULT_ELEMENTS: List[Dict[str, Any]] = [
    {"name": "Vin",      "type": "vsrc",       "n1": "e_in",   "n2": "0",       "value": "1.0",                   "port2n1": "",        "port2n2": ""},
    {"name": "Re",       "type": "R",          "n1": "e_in",   "n2": "e_coil",  "value": "Re",                    "port2n1": "",        "port2n2": ""},
    {"name": "Le",       "type": "L",          "n1": "e_coil", "n2": "e_mech",  "value": "Le",                    "port2n1": "",        "port2n2": ""},
    {"name": "Bl",       "type": "xfmr_Bl",    "n1": "e_mech", "n2": "0",       "value": "Bl",                    "port2n1": "m_drv",   "port2n2": "0"},
    {"name": "Mms",      "type": "mass_mob",   "n1": "m_drv",  "n2": "0",       "value": "Mms",                   "port2n1": "",        "port2n2": ""},
    {"name": "Rms",      "type": "damper_mob", "n1": "m_drv",  "n2": "0",       "value": "Rms",                   "port2n1": "",        "port2n2": ""},
    {"name": "Cms",      "type": "spring_mob", "n1": "m_drv",  "n2": "0",       "value": "Cms",                   "port2n1": "",        "port2n2": ""},
    {"name": "Sd_front", "type": "xfmr_S",     "n1": "m_drv",  "n2": "0",       "value": "Sd",                    "port2n1": "a_front", "port2n2": "0"},
    {"name": "Sd_back",  "type": "xfmr_S",     "n1": "m_drv",  "n2": "0",       "value": "Sd",                    "port2n1": "a_back",  "port2n2": "0"},
    {"name": "C_front",  "type": "acompliance","n1": "a_front", "n2": "0",       "value": "V_front/(rho*c**2)",    "port2n1": "",        "port2n2": ""},
    {"name": "C_back",   "type": "acompliance","n1": "a_back",  "n2": "0",       "value": "V_back/(rho*c**2)",     "port2n1": "",        "port2n2": ""},
    {"name": "R_leak",   "type": "aresistance","n1": "a_front", "n2": "0",       "value": "R_leak_front",          "port2n1": "",        "port2n2": ""},
    {"name": "R_vent",   "type": "aresistance","n1": "a_back",  "n2": "0",       "value": "R_vent_back",           "port2n1": "",        "port2n2": ""},
    {"name": "M_vent",   "type": "ainertance", "n1": "a_back",  "n2": "0",       "value": "M_vent_back",           "port2n1": "",        "port2n2": ""},
]

DEFAULT_PARAMS: Dict[str, float] = {
    "Re": 2.1, "Le": 80e-6, "Bl": 3.4,
    "Mms": 0.18e-3, "Rms": 0.6, "Cms": 0.35e-3,
    "Sd": 2.8e-4,
    "rho": 1.18, "c": 343.0,
    "V_front": 1.2e-6, "V_back": 6.0e-6,
    "R_leak_front": 8.0e6, "R_vent_back": 2.0e6, "M_vent_back": 2.0e6,
}

# ── session state init ────────────────────────────────────────────────────────
def _init_state():
    if "elements" not in st.session_state:
        st.session_state.elements = [dict(e) for e in DEFAULT_ELEMENTS]
    if "params" not in st.session_state:
        st.session_state.params = dict(DEFAULT_PARAMS)
    if "fmin" not in st.session_state:
        st.session_state.fmin = 20.0
    if "fmax" not in st.session_state:
        st.session_state.fmax = 20000.0
    if "n_pts" not in st.session_state:
        st.session_state.n_pts = 500
    if "p_node" not in st.session_state:
        st.session_state.p_node = "a_front"
    if "vin_elem" not in st.session_state:
        st.session_state.vin_elem = "Vin"
    if "result_cache" not in st.session_state:
        st.session_state.result_cache = None

_init_state()

# ── helpers ───────────────────────────────────────────────────────────────────
def elements_to_config_list(elements: List[Dict]) -> List[Dict]:
    out = []
    for el in elements:
        is2p = el["type"] in TWO_PORT_TYPES
        if is2p:
            entry = {
                "name": el["name"],
                "type": el["type"],
                "value": el["value"],
                "ports": [
                    {"n1": el["n1"], "n2": el["n2"]},
                    {"n1": el["port2n1"], "n2": el["port2n2"]},
                ],
            }
        else:
            entry = {
                "name": el["name"],
                "type": el["type"],
                "n1": el["n1"],
                "n2": el["n2"],
                "value": el["value"],
            }
        out.append(entry)
    return out


def build_cfg() -> Dict[str, Any]:
    spacing = "log" if st.session_state.fmin > 0 else "linear"
    return {
        "ground": "0",
        "freq": {
            "fmin": st.session_state.fmin,
            "fmax": st.session_state.fmax,
            "n": st.session_state.n_pts,
            "spacing": spacing,
        },
        "params": st.session_state.params,
        "elements": elements_to_config_list(st.session_state.elements),
        "outputs": {
            "p_front_node": st.session_state.p_node,
            "input_current_element": st.session_state.vin_elem,
            "Vin_for_Zin": 1.0,
        },
    }


def run_solver():
    cfg = build_cfg()
    try:
        net, params, freqs = build_network_from_config(cfg)
        result = net.build_and_solve(freqs=freqs, params=params)

        p_node = cfg["outputs"]["p_front_node"]
        vin_name = cfg["outputs"]["input_current_element"]

        p_front = get_node_effort(result, p_node)
        i_in = get_element_flow(result, vin_name, port=0)
        zin = zin_from_v_and_i(1.0, i_in)
        spl = spl_from_pressure(p_front)

        res = estimate_f0_q_from_impedance(
            freqs, zin,
            Re=params.get("Re"),
            Le=params.get("Le"),
            search_min_hz=20.0,
            search_max_hz=min(800.0, st.session_state.fmax),
        )

        st.session_state.result_cache = {
            "freqs": freqs,
            "spl": spl,
            "zin_mag": np.abs(zin),
            "p_front_db": 20 * np.log10(np.abs(p_front) + 1e-30),
            "zin_phase_deg": np.degrees(np.unwrap(np.angle(zin))),
            "p_phase_deg": np.degrees(np.unwrap(np.angle(p_front))),
            "resonance": res,
            "error": None,
        }
    except Exception as exc:
        st.session_state.result_cache = {"error": str(exc)}


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 11px; color: #64748b; letter-spacing: 1px; }
    .metric-value { font-size: 20px; font-weight: 700; color: #38bdf8; font-family: monospace; }
    .metric-unit  { font-size: 12px; color: #475569; }
    h1 { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎧 LPM Modeler")
    st.caption("Lumped-Parameter Headphone Model — V → p")

    # ── Frequency ──
    with st.expander("⚙ Frequency Range", expanded=True):
        c1, c2 = st.columns(2)
        st.session_state.fmin = c1.number_input("f_min [Hz]", value=float(st.session_state.fmin), min_value=1.0, step=1.0)
        st.session_state.fmax = c2.number_input("f_max [Hz]", value=float(st.session_state.fmax), min_value=100.0, step=100.0)
        st.session_state.n_pts = st.slider("N frequency points", 50, 2000, st.session_state.n_pts, step=50)

    # ── Output nodes ──
    with st.expander("📡 Output Nodes", expanded=False):
        all_nodes = sorted({
            n
            for el in st.session_state.elements
            for n in [el["n1"], el["n2"], el.get("port2n1",""), el.get("port2n2","")]
            if n
        })
        st.session_state.p_node = st.selectbox(
            "Pressure node (p_front)", all_nodes,
            index=all_nodes.index(st.session_state.p_node) if st.session_state.p_node in all_nodes else 0,
        )
        elem_names = [e["name"] for e in st.session_state.elements]
        st.session_state.vin_elem = st.selectbox(
            "Input current element (Vin)", elem_names,
            index=elem_names.index(st.session_state.vin_elem) if st.session_state.vin_elem in elem_names else 0,
        )

    # ── Parameters ──
    with st.expander("🔢 Parameters", expanded=False):
        updated_params = {}
        for k, v in st.session_state.params.items():
            updated_params[k] = st.number_input(
                k, value=float(v), format="%.6g", key=f"param_{k}"
            )
        st.session_state.params = updated_params

        st.markdown("**Add parameter**")
        pc1, pc2 = st.columns([2, 1])
        new_key = pc1.text_input("Name", key="new_param_key", label_visibility="collapsed", placeholder="name")
        new_val = pc2.text_input("Value", key="new_param_val", label_visibility="collapsed", placeholder="0.0")
        if st.button("＋ Add param", use_container_width=True):
            if new_key:
                st.session_state.params[new_key] = float(new_val) if new_val else 0.0
                st.rerun()

    # ── Load / Save config ──
    with st.expander("💾 Import / Export", expanded=False):
        uploaded = st.file_uploader("Load JSON config", type=["json"])
        if uploaded:
            try:
                cfg_raw = json.load(uploaded)
                st.session_state.params = {k: float(v) for k, v in cfg_raw.get("params", {}).items()}
                raw_els = cfg_raw.get("elements", [])
                new_els = []
                for e in raw_els:
                    if "ports" in e:
                        ports = e["ports"]
                        new_els.append({
                            "name": e["name"], "type": e["type"], "value": str(e.get("value", 0)),
                            "n1": ports[0]["n1"], "n2": ports[0]["n2"],
                            "port2n1": ports[1]["n1"] if len(ports) > 1 else "",
                            "port2n2": ports[1]["n2"] if len(ports) > 1 else "",
                        })
                    else:
                        new_els.append({
                            "name": e["name"], "type": e["type"], "value": str(e.get("value", 0)),
                            "n1": e.get("n1",""), "n2": e.get("n2","0"),
                            "port2n1": "", "port2n2": "",
                        })
                st.session_state.elements = new_els
                freq_cfg = cfg_raw.get("freq", {})
                st.session_state.fmin = float(freq_cfg.get("fmin", 20))
                st.session_state.fmax = float(freq_cfg.get("fmax", 20000))
                st.session_state.n_pts = int(freq_cfg.get("n", 500))
                st.success("Config loaded!")
                st.rerun()
            except Exception as ex:
                st.error(f"Load failed: {ex}")

        if st.button("⬇ Export current config as JSON", use_container_width=True):
            cfg_json = json.dumps(build_cfg(), indent=2, default=str)
            st.download_button("Download config.json", cfg_json, "lpm_config.json", "application/json")

    st.divider()
    run_clicked = st.button("▶ RUN SOLVER", type="primary", use_container_width=True)
    if run_clicked:
        run_solver()

    if st.button("↩ Reset to default", use_container_width=True):
        for key in ["elements","params","result_cache"]:
            if key in st.session_state:
                del st.session_state[key]
        _init_state()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🎧 LPM Headphone Modeler")

tab_network, tab_plots, tab_metrics = st.tabs(["🔌 Network", "📈 Plots", "📊 Metrics"])

# ── NETWORK TAB ───────────────────────────────────────────────────────────────
with tab_network:
    st.markdown("### Element Network (Zweitore)")

    # Add element form
    with st.expander("➕ Add Element", expanded=False):
        ac1, ac2, ac3 = st.columns([1.5, 2, 1])
        add_cat  = ac1.selectbox("Category", list(CATALOG.keys()), key="add_cat")
        cat_types = CATALOG[add_cat]
        add_label = ac2.selectbox("Type", [e["label"] for e in cat_types], key="add_label")
        add_meta  = next(e for e in cat_types if e["label"] == add_label)

        if ac3.button("Add", use_container_width=True):
            is2p = add_meta["ports"] == 2
            st.session_state.elements.append({
                "name":    add_meta["type"] + f"_{len(st.session_state.elements)+1}",
                "type":    add_meta["type"],
                "n1":      "",
                "n2":      "0",
                "value":   add_meta["default"],
                "port2n1": "" if not is2p else "",
                "port2n2": "" if not is2p else "0",
            })
            st.rerun()

    # Element table
    header = st.columns([2, 2, 1.5, 1.5, 2, 1.5, 1.5, 0.8])
    for col, h in zip(header, ["Name","Type","n1","n2","Value","Port2 n1","Port2 n2",""]):
        col.markdown(f"**{h}**")
    st.divider()

    to_delete = None
    for i, el in enumerate(st.session_state.elements):
        is2p = el["type"] in TWO_PORT_TYPES
        cols = st.columns([2, 2, 1.5, 1.5, 2, 1.5, 1.5, 0.8])

        el["name"]  = cols[0].text_input("n", value=el["name"],  key=f"el_name_{i}",  label_visibility="collapsed")
        el["type"]  = cols[1].text_input("t", value=el["type"],  key=f"el_type_{i}",  label_visibility="collapsed")
        el["n1"]    = cols[2].text_input("n1",value=el["n1"],    key=f"el_n1_{i}",    label_visibility="collapsed")
        el["n2"]    = cols[3].text_input("n2",value=el["n2"],    key=f"el_n2_{i}",    label_visibility="collapsed")
        el["value"] = cols[4].text_input("v", value=el["value"], key=f"el_val_{i}",   label_visibility="collapsed")

        if is2p:
            el["port2n1"] = cols[5].text_input("p2n1",value=el.get("port2n1",""), key=f"el_p2n1_{i}", label_visibility="collapsed")
            el["port2n2"] = cols[6].text_input("p2n2",value=el.get("port2n2","0"),key=f"el_p2n2_{i}", label_visibility="collapsed")
        else:
            cols[5].markdown("—")
            cols[6].markdown("—")

        if cols[7].button("🗑", key=f"del_{i}"):
            to_delete = i

    if to_delete is not None:
        st.session_state.elements.pop(to_delete)
        st.rerun()

    st.caption(f"{len(st.session_state.elements)} elements · nodes: {', '.join(sorted(all_nodes))}")

# ── PLOTS TAB ─────────────────────────────────────────────────────────────────
with tab_plots:
    cache = st.session_state.result_cache

    if cache is None:
        st.info("Press **▶ RUN SOLVER** in the sidebar to compute results.")
    elif cache.get("error"):
        st.error(f"Solver error: {cache['error']}")
    else:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            USE_PLOTLY = True
        except ImportError:
            USE_PLOTLY = False

        freqs = cache["freqs"]

        if USE_PLOTLY:
            # SPL
            fig_spl = make_subplots(rows=1, cols=1)
            fig_spl.add_trace(go.Scatter(x=freqs, y=cache["spl"], mode="lines", name="SPL", line=dict(color="#38bdf8", width=1.5)))
            fig_spl.update_xaxes(type="log", title_text="Frequency [Hz]", gridcolor="#1e293b")
            fig_spl.update_yaxes(title_text="SPL [dB re 20 µPa / 1V]", gridcolor="#1e293b")
            fig_spl.update_layout(title="SPL — Front Pressure", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                                  font=dict(color="#e2e8f0", family="monospace"), height=320, margin=dict(t=40,b=40,l=60,r=20))
            st.plotly_chart(fig_spl, use_container_width=True)

            # Zin magnitude
            fig_z = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                  subplot_titles=["Magnitude [Ω]", "Phase [deg]"])
            fig_z.add_trace(go.Scatter(x=freqs, y=cache["zin_mag"], mode="lines", name="|Zin|", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
            fig_z.add_trace(go.Scatter(x=freqs, y=cache["zin_phase_deg"], mode="lines", name="∠Zin", line=dict(color="#fb923c", width=1.5)), row=2, col=1)
            fig_z.update_xaxes(type="log", title_text="Frequency [Hz]", gridcolor="#1e293b", row=2, col=1)
            fig_z.update_xaxes(type="log", gridcolor="#1e293b", row=1, col=1)
            fig_z.update_yaxes(gridcolor="#1e293b")
            fig_z.update_layout(title="Electrical Input Impedance", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                                 font=dict(color="#e2e8f0", family="monospace"), height=420, margin=dict(t=60,b=40,l=60,r=20))
            st.plotly_chart(fig_z, use_container_width=True)

            # Pressure transfer
            fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                  subplot_titles=["Magnitude [dB]", "Phase [deg]"])
            fig_p.add_trace(go.Scatter(x=freqs, y=cache["p_front_db"], mode="lines", name="|p/V|", line=dict(color="#34d399", width=1.5)), row=1, col=1)
            fig_p.add_trace(go.Scatter(x=freqs, y=cache["p_phase_deg"], mode="lines", name="∠p", line=dict(color="#6ee7b7", width=1.5)), row=2, col=1)
            fig_p.update_xaxes(type="log", title_text="Frequency [Hz]", gridcolor="#1e293b", row=2, col=1)
            fig_p.update_xaxes(type="log", gridcolor="#1e293b", row=1, col=1)
            fig_p.update_yaxes(gridcolor="#1e293b")
            fig_p.update_layout(title="Pressure Transfer |p_front / V|", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                                 font=dict(color="#e2e8f0", family="monospace"), height=420, margin=dict(t=60,b=40,l=60,r=20))
            st.plotly_chart(fig_p, use_container_width=True)

        else:
            # Fallback: matplotlib
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor="#0f172a")
            for ax, y, label, color in [
                (axes[0], cache["spl"],        "SPL [dB re 20µPa]",    "#38bdf8"),
                (axes[1], cache["zin_mag"],    "|Zin| [Ω]",            "#f59e0b"),
                (axes[2], cache["p_front_db"], "|p/V| [dB]",           "#34d399"),
            ]:
                ax.semilogx(freqs, y, color=color, linewidth=1.2)
                ax.set_ylabel(label, color="#94a3b8", fontsize=9)
                ax.set_facecolor("#0d1117")
                ax.grid(True, which="both", color="#1e293b", linewidth=0.5)
                ax.tick_params(colors="#475569")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e293b")
            axes[2].set_xlabel("Frequency [Hz]", color="#94a3b8", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Install plotly for interactive plots: `pip install plotly`")

# ── METRICS TAB ───────────────────────────────────────────────────────────────
with tab_metrics:
    cache = st.session_state.result_cache

    if cache is None:
        st.info("Run the solver first.")
    elif cache.get("error"):
        st.error(cache["error"])
    else:
        res = cache["resonance"]
        st.markdown("### Impedance Resonance Metrics")
        st.caption(f"Method: {res.method}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("f₀  [Hz]",      f"{res.f0_hz:.1f}")
        m2.metric("Q",             f"{res.q:.3f}"     if not np.isnan(res.q)    else "—")
        m3.metric("f₁ −3dB [Hz]",  f"{res.f1_hz:.1f}" if not np.isnan(res.f1_hz) else "—")
        m4.metric("f₂ −3dB [Hz]",  f"{res.f2_hz:.1f}" if not np.isnan(res.f2_hz) else "—")

        st.divider()
        st.markdown("### Raw Results Table")

        freqs = cache["freqs"]
        table_data = {
            "Frequency [Hz]": np.round(freqs, 2),
            "SPL [dB]":       np.round(cache["spl"], 2),
            "|Zin| [Ω]":      np.round(cache["zin_mag"], 4),
            "∠Zin [deg]":     np.round(cache["zin_phase_deg"], 2),
            "|p/V| [dB]":     np.round(cache["p_front_db"], 2),
        }
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, height=400)

        csv = df.to_csv(index=False)
        st.download_button("⬇ Download CSV", csv, "lpm_results.csv", "text/csv")

        if res.notes:
            st.caption(f"Notes: {res.notes}")