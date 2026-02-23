from lpm.config import load_config, build_network_from_config

def test_smoke():
    cfg = load_config("configs/headphone_min.json")
    net, params, freqs = build_network_from_config(cfg)
    res = net.build_and_solve(freqs=freqs[:20], params=params)
    assert "node_efforts" in res
    assert res["node_efforts"].shape[0] == 20
