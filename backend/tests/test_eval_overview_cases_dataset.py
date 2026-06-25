from pathlib import Path

from scripts.eval_rag import load_eval_cases


def test_overview_cases_dataset_loads_with_expected_shape() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    dataset_path = backend_root / "evals" / "rag_eval_overview_cases.jsonl"

    cases = load_eval_cases(dataset_path)

    assert len(cases) == 5
    assert len({case.id for case in cases}) == len(cases)
    assert sum(case.answer_mode == "grounded" for case in cases) == 5
    assert {case.id for case in cases} == {"ovc001", "ovc002", "ovc003", "ovc004", "ovc005"}
    assert {case.category for case in cases} >= {
        "overview/leave",
        "overview/discipline",
        "overview/dormitory",
        "overview/boundary",
    }
