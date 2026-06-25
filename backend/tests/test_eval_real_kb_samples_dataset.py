from pathlib import Path

from scripts.eval_rag import load_eval_cases


def test_real_kb_samples_dataset_loads_with_expected_shape() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    dataset_path = backend_root / "evals" / "rag_eval_real_kb_samples.jsonl"

    cases = load_eval_cases(dataset_path)

    assert len(cases) == 16
    assert len({case.id for case in cases}) == len(cases)
    assert sum(case.answer_mode == "grounded" for case in cases) == 14
    assert sum(case.answer_mode == "no_answer" for case in cases) == 2
    assert {case.id for case in cases} >= {"rk001", "rk002", "rk005", "rk014", "rk016"}
    assert {case.category for case in cases} >= {
        "real_kb/plain_pdf",
        "real_kb/table_pdf",
        "real_kb/mixed_ocr",
        "real_kb/control",
    }
