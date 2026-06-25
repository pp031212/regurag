from app.document_processing.pdf.postprocess import is_likely_header_row


def test_is_likely_header_row_prefers_label_row_over_descriptive_row() -> None:
    header_row = ["项目", "要求", "说明"]
    data_row = ["请假", "需提前提交审批", "病假需要补充证明材料"]

    assert is_likely_header_row(header_row, data_row) is True


def test_is_likely_header_row_rejects_descriptive_row_without_keyword_dependency() -> None:
    descriptive_row = ["决策树", "通过特征划分不断构造树形结构完成分类或回归"]
    next_row = ["随机森林", "通过多棵树投票提升稳定性和泛化能力"]

    assert is_likely_header_row(descriptive_row, next_row) is False


def test_is_likely_header_row_rejects_numeric_data_row() -> None:
    numeric_row = ["2024-01", "95", "88.5"]
    next_row = ["2024-02", "96", "89.1"]

    assert is_likely_header_row(numeric_row, next_row) is False
