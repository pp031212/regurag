from pathlib import Path

from app.evals.split_profile_ab import run_sparse_provider_ab_compare, run_split_profile_ab_compare


def test_strict_split_profile_ab_compare_shows_contender_improvement() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    report = run_split_profile_ab_compare(
        dataset_path=backend_root / "evals" / "rag_eval_split_profile_strict.jsonl",
        fixture_dir=backend_root / "evals" / "fixtures" / "split_profile_strict",
        baseline_profile="",
        contender_profile="rules_cn",
        child_chunk_size=18,
        parent_chunk_size=32,
    )

    metrics = report["comparison"]["metrics"]
    assert metrics["retrieval_hit"]["delta"] > 0
    assert metrics["final_context_hit"]["delta"] > 0
    assert metrics["citation_hit"]["delta"] > 0
    assert metrics["answer_hit"]["delta"] > 0
    assert "sab002" in metrics["retrieval_hit"]["improved_case_ids"]
    assert "sab002" in metrics["final_context_hit"]["improved_case_ids"]
    assert "sab002" in metrics["citation_hit"]["improved_case_ids"]
    assert "sab004" in metrics["retrieval_hit"]["improved_case_ids"]
    assert "sab004" in metrics["final_context_hit"]["improved_case_ids"]
    assert "sab004" in metrics["citation_hit"]["improved_case_ids"]


def test_sparse_provider_ab_compare_reports_all_variants() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    report = run_sparse_provider_ab_compare(
        dataset_path=backend_root / "evals" / "rag_eval_split_profile_strict.jsonl",
        fixture_dir=backend_root / "evals" / "fixtures" / "split_profile_strict",
        profile="rules_cn",
    )

    sqlite_summary = report["reports"]["sqlite_fts"]["summary"]
    bm25_summary = report["reports"]["bm25"]["summary"]
    scan_summary = report["reports"]["scan"]["summary"]
    none_summary = report["reports"]["none"]["summary"]
    sqlite_vs_none = report["comparisons"]["sqlite_fts_vs_none"]["metrics"]
    sqlite_vs_scan = report["comparisons"]["sqlite_fts_vs_scan"]["metrics"]

    assert bm25_summary["retrieval_hit_rate"] >= none_summary["retrieval_hit_rate"]
    assert sqlite_summary["retrieval_hit_rate"] > none_summary["retrieval_hit_rate"]
    assert sqlite_summary["retrieval_hit_rate"] > scan_summary["retrieval_hit_rate"]
    assert "sab005" in sqlite_vs_none["retrieval_hit"]["regressed_case_ids"]
    assert "sab007" in sqlite_vs_none["retrieval_hit"]["regressed_case_ids"]
    assert "sab006" in sqlite_vs_scan["retrieval_hit"]["regressed_case_ids"]
    assert "sqlite_fts_vs_bm25" in report["comparisons"]
    assert "sqlite_fts_vs_scan" in report["comparisons"]
    assert "sqlite_fts_vs_none" in report["comparisons"]
