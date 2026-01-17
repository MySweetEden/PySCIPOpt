import json

from pyscipopt import Model


def test_tracefile_basic(tmp_path):
    """Basic tracefile functionality test."""
    trace_path = tmp_path / "trace.jsonl"

    m = Model()
    m.hideOutput()
    x = m.addVar("x", vtype="I", lb=0, ub=10)
    m.setObjective(x, "maximize")
    m.setTracefile(str(trace_path))
    m.optimize()

    assert trace_path.exists()

    with open(trace_path) as f:
        lines = f.readlines()

    assert len(lines) >= 1

    events = [json.loads(line) for line in lines]
    types = [e["type"] for e in events]

    assert "solve_finish" in types

    required_fields = {
        "type",
        "time",
        "primalbound",
        "dualbound",
        "gap",
        "nodes",
        "nsol",
    }
    for e in events:
        assert required_fields <= set(e.keys())


def test_tracefile_none(tmp_path):
    """Test disabling tracefile with None."""
    trace_path = tmp_path / "trace.jsonl"

    m = Model()
    m.hideOutput()
    x = m.addVar("x", vtype="I", lb=0, ub=10)
    m.setObjective(x, "maximize")
    m.setTracefile(str(trace_path))
    m.setTracefile(None)  # Disable
    m.optimize()

    assert not trace_path.exists()
