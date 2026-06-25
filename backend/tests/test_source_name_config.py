from app.services.source_name_config import resolve_source_name


def test_resolve_source_name_uses_configured_official_name() -> None:
    assert resolve_source_name("劳动合同法_整理版.docx") == "《中华人民共和国劳动合同法》"


def test_resolve_source_name_falls_back_to_document_stem() -> None:
    assert resolve_source_name("学生管理制度.docx") == "《学生管理制度》"
