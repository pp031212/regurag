"""回归检查 PDF 结构化解析。

读取 PDF fixture 和期望结果，验证正文清洗、表格合并、图片 OCR 等解析行为是否保持稳定。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.document_processing.pdf.service import PDFStructuringService


@dataclass(slots=True)
class PDFRegressionCase:
    id: str
    pdf_path: Path
    output_dir: Path
    enable_ocr: bool
    expected_total_pages: int
    expected_page_text_contains: dict[int, list[str]]
    expected_page_text_not_contains: dict[int, list[str]]
    expected_page_table_text_contains: dict[int, list[str]]
    expected_page_table_text_not_contains: dict[int, list[str]]
    expected_image_counts: dict[int, int]
    expected_table_counts: dict[int, int]


@dataclass(slots=True)
class PDFRegressionResult:
    id: str
    passed: bool
    output_dir: str
    structured_json_path: str
    readable_txt_path: str
    total_pages: int
    checks: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDF structuring regression cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "evals" / "pdf_regression_cases.json",
        help="Path to the PDF regression case definition JSON file.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Only run the specified regression case id.",
    )
    parser.add_argument(
        "--refresh-output",
        action="store_true",
        help="Delete the case output directory before preprocessing.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "evals" / "results" / "pdf_regression_report.json",
        help="Path to save the regression report JSON.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[PDFRegressionCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Regression case file must contain a JSON array.")

    cases: list[PDFRegressionCase] = []
    for item in payload:
        case = PDFRegressionCase(
            id=str(item["id"]),
            pdf_path=_resolve_path(path, str(item["pdf_path"])),
            output_dir=_resolve_path(path, str(item["output_dir"])),
            enable_ocr=bool(item.get("enable_ocr", True)),
            expected_total_pages=int(item["expected_total_pages"]),
            expected_page_text_contains=_normalize_page_mapping(item.get("expected_page_text_contains", {})),
            expected_page_text_not_contains=_normalize_page_mapping(item.get("expected_page_text_not_contains", {})),
            expected_page_table_text_contains=_normalize_page_mapping(item.get("expected_page_table_text_contains", {})),
            expected_page_table_text_not_contains=_normalize_page_mapping(item.get("expected_page_table_text_not_contains", {})),
            expected_image_counts={int(key): int(value) for key, value in dict(item.get("expected_image_counts", {})).items()},
            expected_table_counts={int(key): int(value) for key, value in dict(item.get("expected_table_counts", {})).items()},
        )
        cases.append(case)
    return cases


def _resolve_path(case_file: Path, raw_path: str) -> Path:
    if raw_path.startswith("project://"):
        return (ROOT.parent / raw_path.removeprefix("project://")).resolve()
    if raw_path.startswith("backend://"):
        return (ROOT / raw_path.removeprefix("backend://")).resolve()
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (case_file.parent / path).resolve()


def _normalize_page_mapping(raw_mapping: object) -> dict[int, list[str]]:
    mapping = dict(raw_mapping) if isinstance(raw_mapping, dict) else {}
    return {int(key): [str(item) for item in value] for key, value in mapping.items()}


def run_case(case: PDFRegressionCase, refresh_output: bool) -> PDFRegressionResult:
    if refresh_output and case.output_dir.exists():
        shutil.rmtree(case.output_dir, ignore_errors=True)

    service = PDFStructuringService(enable_ocr=case.enable_ocr)
    artifacts = service.preprocess(case.pdf_path, case.output_dir)
    payload = artifacts.payload

    checks: list[dict[str, object]] = []
    checks.append(
        _build_check(
            name="total_pages",
            passed=payload.get("total_pages") == case.expected_total_pages,
            details={
                "expected": case.expected_total_pages,
                "actual": payload.get("total_pages"),
            },
        )
    )

    pages = payload.get("pages", [])
    for page_number, expected_snippets in case.expected_page_text_contains.items():
        page_text = _get_page_text(pages, page_number)
        for snippet in expected_snippets:
            checks.append(
                _build_check(
                    name=f"page_{page_number}_contains",
                    passed=snippet in page_text,
                    details={"snippet": snippet},
                )
            )

    for page_number, forbidden_snippets in case.expected_page_text_not_contains.items():
        page_text = _get_page_text(pages, page_number)
        for snippet in forbidden_snippets:
            checks.append(
                _build_check(
                    name=f"page_{page_number}_not_contains",
                    passed=snippet not in page_text,
                    details={"snippet": snippet},
                )
            )

    for page_number, expected_snippets in case.expected_page_table_text_contains.items():
        table_text = _get_page_table_text(pages, page_number)
        for snippet in expected_snippets:
            checks.append(
                _build_check(
                    name=f"page_{page_number}_table_contains",
                    passed=snippet in table_text,
                    details={"snippet": snippet},
                )
            )

    for page_number, forbidden_snippets in case.expected_page_table_text_not_contains.items():
        table_text = _get_page_table_text(pages, page_number)
        for snippet in forbidden_snippets:
            checks.append(
                _build_check(
                    name=f"page_{page_number}_table_not_contains",
                    passed=snippet not in table_text,
                    details={"snippet": snippet},
                )
            )

    for page_number, expected_count in case.expected_image_counts.items():
        actual_count = len(_get_page_field(pages, page_number, "images"))
        checks.append(
            _build_check(
                name=f"page_{page_number}_image_count",
                passed=actual_count == expected_count,
                details={"expected": expected_count, "actual": actual_count},
            )
        )

    for page_number, expected_count in case.expected_table_counts.items():
        actual_count = len(_get_page_field(pages, page_number, "tables"))
        checks.append(
            _build_check(
                name=f"page_{page_number}_table_count",
                passed=actual_count == expected_count,
                details={"expected": expected_count, "actual": actual_count},
            )
        )

    return PDFRegressionResult(
        id=case.id,
        passed=all(check["passed"] for check in checks),
        output_dir=str(case.output_dir),
        structured_json_path=str(artifacts.structured_json_path),
        readable_txt_path=str(artifacts.readable_txt_path),
        total_pages=int(payload.get("total_pages", 0)),
        checks=checks,
    )


def _build_check(name: str, passed: bool, details: dict[str, object]) -> dict[str, object]:
    return {"name": name, "passed": passed, "details": details}


def _get_page_text(pages: list[object], page_number: int) -> str:
    page = _get_page(pages, page_number)
    return str(page.get("text") or "") if page else ""


def _get_page_field(pages: list[object], page_number: int, field: str) -> list[object]:
    page = _get_page(pages, page_number)
    value = page.get(field) if page else []
    return value if isinstance(value, list) else []


def _get_page_table_text(pages: list[object], page_number: int) -> str:
    tables = _get_page_field(pages, page_number, "tables")
    texts: list[str] = []
    for table in tables:
        if isinstance(table, dict):
            table_text = table.get("text")
            if isinstance(table_text, str):
                texts.append(table_text)
    return "\n".join(texts)


def _get_page(pages: list[object], page_number: int) -> dict[str, object] | None:
    for page in pages:
        if isinstance(page, dict) and int(page.get("page_number", 0)) == page_number:
            return page
    return None


def save_report(results: list[PDFRegressionResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_cases": len(results),
        "passed_cases": sum(1 for result in results if result.passed),
        "failed_cases": sum(1 for result in results if not result.passed),
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases.resolve())
    if args.case_id:
        cases = [case for case in cases if case.id == args.case_id]
        if not cases:
            raise ValueError(f"Regression case not found: {args.case_id}")

    results = [run_case(case, refresh_output=args.refresh_output) for case in cases]
    save_report(results, args.report_path.resolve())

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.id}")
        print(f"  structured_json: {result.structured_json_path}")
        print(f"  readable_txt: {result.readable_txt_path}")
        for check in result.checks:
            if not check["passed"]:
                print(f"  - failed: {check['name']} -> {check['details']}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
