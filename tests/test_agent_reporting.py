from src.agent_eval.reporting import AgentRunRecord, compare_to_baseline, summarize_runs


def test_summarize_and_compare_agent_runs() -> None:
    records = [
        AgentRunRecord("t1", "baseline", True, 10, 1, 20.0),
        AgentRunRecord("t2", "baseline", False, 12, 2, 30.0),
        AgentRunRecord("t1", "rys_24_35", True, 8, 0, 25.0, extra_layers=11, overhead_fraction=0.17),
        AgentRunRecord("t2", "rys_24_35", True, 9, 1, 28.0, extra_layers=11, overhead_fraction=0.17),
    ]

    summary = summarize_runs(records)
    baseline = next(row for row in summary if row["condition"] == "baseline")
    rys = next(row for row in summary if row["condition"] == "rys_24_35")

    assert baseline["success_rate"] == 0.5
    assert rys["success_rate"] == 1.0

    deltas = compare_to_baseline(summary, "baseline")
    rys_delta = next(row for row in deltas if row["condition"] == "rys_24_35")
    assert rys_delta["delta_success_rate"] == 0.5
