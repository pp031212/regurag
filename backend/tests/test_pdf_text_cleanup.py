from app.document_processing.pdf.utils import (
    mark_toc_pages_for_skip,
    merge_cross_page_text_blocks,
    normalize_paragraph_text,
    remove_repeated_page_noise_blocks,
    table_rows_to_text,
)


def test_normalize_paragraph_text_removes_fake_line_breaks_and_intra_word_spaces() -> None:
    raw_text = (
        "进入教学区必须佩戴胸卡或园区进门卡，未按 要求佩戴者扣2 分/次。\n"
        "应保持教室卫生以 及课桌椅整洁。\n"
        "如 厕后及时冲水。"
    )

    normalized = normalize_paragraph_text(raw_text)

    assert "未按要求佩戴者" in normalized
    assert "卫生以及课桌椅整洁" in normalized
    assert "如厕后及时冲水" in normalized


def test_normalize_paragraph_text_removes_spacing_around_punctuation_and_units() -> None:
    raw_text = (
        "同学之间要友 好相处，扣20 分/次 。\n"
        "如有违反， 扣2 分/\n"
        "次。希望大家在学习的每一天都严格要求自己， 在学习技术的同时成长。"
    )

    normalized = normalize_paragraph_text(raw_text)

    assert "友好相处" in normalized
    assert "扣20分/次。" in normalized
    assert "如有违反，扣2分/次。" in normalized
    assert "在学习技术的同时成长" in normalized


def test_table_rows_to_text_adds_header_bound_records() -> None:
    rows = [
        ["行为", "处理方式", "扣分"],
        ["迟到20分钟以内", "记录一次", "扣3分/次"],
        ["旷课半天", "按旷课处理", "扣10分/次"],
    ]

    text = table_rows_to_text(rows)

    assert "行为 | 处理方式 | 扣分" in text
    assert "旷课半天 | 按旷课处理 | 扣10分/次" in text
    assert "[记录 2]" in text
    assert "行为: 旷课半天" in text
    assert "处理方式: 按旷课处理" in text
    assert "扣分: 扣10分/次" in text


def test_table_rows_to_text_header_scoring_allows_year_label() -> None:
    rows = [
        ["2024年度考核项", "标准", "备注"],
        ["迟到", "扣3分/次", "纳入月度考核"],
        ["旷课", "扣20分/次", "需班主任确认"],
    ]

    text = table_rows_to_text(rows)

    assert "[记录 1]" in text
    assert "2024年度考核项: 迟到" in text
    assert "标准: 扣3分/次" in text


def test_table_rows_to_text_does_not_bind_data_row_as_header() -> None:
    rows = [
        ["迟到", "扣3分/次", "2024年执行"],
        ["旷课", "扣20分/次", "2024年执行"],
        ["早退", "扣3分/次", "2024年执行"],
    ]

    text = table_rows_to_text(rows)

    assert "迟到 | 扣3分/次 | 2024年执行" in text
    assert "[记录 1]" not in text
    assert "迟到: 旷课" not in text


def test_mark_toc_pages_for_skip_skips_obvious_toc_page() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "目录\n"
                    "第一章 总则 ........ 1\n"
                    "第二章 考勤管理 ........ 3\n"
                    "第三章 奖惩规则 ........ 8\n"
                    "附录 表单模板 ........ 12"
                ),
                "text_blocks": [
                    {
                        "text": "目录",
                        "bbox": [72, 70, 120, 90],
                        "page_height": 800,
                        "avg_font_size": 16,
                    }
                ],
                "tables": [],
                "images": [],
            },
            {
                "page_number": 2,
                "text": "第一章 总则\n本制度适用于全体学员。",
                "text_blocks": [],
                "tables": [],
                "images": [],
            },
        ]
    }

    cleaned = mark_toc_pages_for_skip(structured_data)

    toc_page = cleaned["pages"][0]
    assert toc_page["is_toc_page"] is True
    assert toc_page["skip_reason"] == "toc_page"
    assert toc_page["text"] == ""
    assert toc_page["text_blocks"] == []
    assert cleaned["toc_page_skips"][0]["page_number"] == 1
    assert cleaned["pages"][1]["text"].startswith("第一章 总则")


def test_mark_toc_pages_for_skip_keeps_normal_page_that_mentions_toc() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": 1,
                "text": "目录管理要求\n各部门应当定期维护制度目录，确保文件编号准确。",
                "text_blocks": [
                    {
                        "text": "目录管理要求",
                        "bbox": [72, 70, 220, 90],
                        "page_height": 800,
                        "avg_font_size": 14,
                    }
                ],
                "tables": [],
                "images": [],
            }
        ]
    }

    cleaned = mark_toc_pages_for_skip(structured_data)

    assert cleaned["pages"][0].get("is_toc_page") is None
    assert cleaned["pages"][0]["text"].startswith("目录管理要求")
    assert "toc_page_skips" not in cleaned


def test_remove_repeated_page_noise_blocks_removes_stable_short_template_text() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": page_number,
                "page_height": 800,
                "text": f"某公司内部制度 V1.0\n第{page_number}页正文内容。",
                "text_blocks": [
                    {
                        "text": "某公司内部制度 V1.0",
                        "bbox": [72, 70, 260, 88],
                        "page_height": 800,
                        "avg_font_size": 9,
                    },
                    {
                        "text": f"第{page_number}页正文内容。",
                        "bbox": [72, 150, 360, 180],
                        "page_height": 800,
                        "avg_font_size": 12,
                    },
                ],
                "tables": [],
                "images": [],
            }
            for page_number in range(1, 5)
        ]
    }

    cleaned = remove_repeated_page_noise_blocks(structured_data)

    assert all("某公司内部制度 V1.0" not in page["text"] for page in cleaned["pages"])
    assert cleaned["repeated_text_noise_removals"][0]["text"] == "某公司内部制度 V1.0"
    assert cleaned["repeated_text_noise_removals"][0]["pages"] == [1, 2, 3, 4]
    assert cleaned["pages"][0]["removed_repeated_noise_blocks"][0]["reason"] == "repeated_stable_layout"


def test_remove_repeated_page_noise_blocks_keeps_repeated_document_units() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": page_number,
                "page_height": 800,
                "text": f"第一条 适用范围\n第{page_number}页正文内容。",
                "text_blocks": [
                    {
                        "text": "第一条 适用范围",
                        "bbox": [72, 70, 260, 88],
                        "page_height": 800,
                        "avg_font_size": 12,
                    },
                    {
                        "text": f"第{page_number}页正文内容。",
                        "bbox": [72, 150, 360, 180],
                        "page_height": 800,
                        "avg_font_size": 12,
                    },
                ],
                "tables": [],
                "images": [],
            }
            for page_number in range(1, 5)
        ]
    }

    cleaned = remove_repeated_page_noise_blocks(structured_data)

    assert all("第一条适用范围" in page["text"].replace(" ", "") for page in cleaned["pages"])
    assert "repeated_text_noise_removals" not in cleaned


def test_remove_repeated_page_noise_blocks_ignores_short_two_page_document() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": page_number,
                "page_height": 800,
                "text": "某公司内部制度 V1.0\n正文内容。",
                "text_blocks": [
                    {
                        "text": "某公司内部制度 V1.0",
                        "bbox": [72, 70, 260, 88],
                        "page_height": 800,
                        "avg_font_size": 9,
                    }
                ],
                "tables": [],
                "images": [],
            }
            for page_number in range(1, 3)
        ]
    }

    cleaned = remove_repeated_page_noise_blocks(structured_data)

    assert all("某公司内部制度 V1.0" in page["text"] for page in cleaned["pages"])
    assert "repeated_text_noise_removals" not in cleaned


def test_merge_cross_page_text_blocks_merges_continued_sentence() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": 1,
                "text": "在课堂上捣乱，不听班主任老师劝解并影响大家学习",
                "tables": [],
                "images": [],
            },
            {
                "page_number": 2,
                "text": "的，扣20 分/次。\n以上规则如遇特殊情况，由班主任老师保留最终解释权。",
                "tables": [],
                "images": [],
            },
        ]
    }

    merged = merge_cross_page_text_blocks(structured_data)

    assert merged["pages"][0]["text"].endswith("在课堂上捣乱，不听班主任老师劝解并影响大家学习的，扣20 分/次。")
    assert merged["pages"][1]["text"].startswith("以上规则如遇特殊情况")
    assert merged["pages"][0]["cross_page_text_merges"][0]["source_page"] == 2
    assert merged["pages"][0]["cross_page_text_merges"][0]["score"] >= 2
    assert "next_starts_with_continuation_word" in merged["pages"][0]["cross_page_text_merges"][0]["reasons"]


def test_merge_cross_page_text_blocks_records_layout_reasons_when_available() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": 1,
                "page_height": 800,
                "text": "迟到超过二十分钟但未超过一小时的，按",
                "text_blocks": [
                    {
                        "text": "迟到超过二十分钟但未超过一小时的，按",
                        "bbox": [72, 720, 420, 748],
                        "page_height": 800,
                        "avg_font_size": 11,
                    }
                ],
                "tables": [],
                "images": [],
            },
            {
                "page_number": 2,
                "page_height": 800,
                "text": "旷课半天处理，扣10分。",
                "text_blocks": [
                    {
                        "text": "旷课半天处理，扣10分。",
                        "bbox": [74, 88, 360, 116],
                        "page_height": 800,
                        "avg_font_size": 11.5,
                    }
                ],
                "tables": [],
                "images": [],
            },
        ]
    }

    merged = merge_cross_page_text_blocks(structured_data)
    merge_debug = merged["pages"][0]["cross_page_text_merges"][0]

    assert merged["pages"][0]["text"].endswith("按旷课半天处理，扣10分。")
    assert set(merge_debug["reasons"]) >= {
        "previous_block_near_page_bottom",
        "next_block_near_page_top",
        "indent_similar",
        "font_size_similar",
    }


def test_merge_cross_page_text_blocks_merges_plain_continuation_when_previous_has_no_terminal() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": 1,
                "text": "迟到超过二十分钟但未超过一小时的，按",
                "tables": [],
                "images": [],
            },
            {
                "page_number": 2,
                "text": "旷课半天处理，扣10分。\n二、宿舍纪律",
                "tables": [],
                "images": [],
            },
        ]
    }

    merged = merge_cross_page_text_blocks(structured_data)

    assert merged["pages"][0]["text"].endswith("迟到超过二十分钟但未超过一小时的，按旷课半天处理，扣10分。")
    assert merged["pages"][1]["text"] == "二、宿舍纪律"


def test_merge_cross_page_text_blocks_merges_rule_result_after_terminal_punctuation() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": 1,
                "text": "迟到超过二十分钟按旷课处理。",
                "tables": [],
                "images": [],
            },
            {
                "page_number": 2,
                "text": "扣10分/次，并由班主任记录。\n二、宿舍纪律",
                "tables": [],
                "images": [],
            },
        ]
    }

    merged = merge_cross_page_text_blocks(structured_data)

    assert merged["pages"][0]["text"].endswith("迟到超过二十分钟按旷课处理。扣10分/次，并由班主任记录。")
    assert merged["pages"][1]["text"] == "二、宿舍纪律"


def test_merge_cross_page_text_blocks_does_not_merge_non_continuation_paragraph() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": 1,
                "text": "第一段到这里结束",
                "tables": [],
                "images": [],
            },
            {
                "page_number": 2,
                "text": "以上规则如遇特殊情况，由班主任老师保留最终解释权。",
                "tables": [],
                "images": [],
            },
        ]
    }

    merged = merge_cross_page_text_blocks(structured_data)

    assert merged["pages"][0]["text"] == "第一段到这里结束"
    assert merged["pages"][1]["text"] == "以上规则如遇特殊情况，由班主任老师保留最终解释权。"


def test_merge_cross_page_text_blocks_does_not_merge_new_document_unit() -> None:
    structured_data = {
        "pages": [
            {
                "page_number": 1,
                "text": "迟到超过二十分钟按旷课处理。",
                "tables": [],
                "images": [],
            },
            {
                "page_number": 2,
                "text": "二、宿舍纪律\n熄灯后喧哗扣5分。",
                "tables": [],
                "images": [],
            },
        ]
    }

    merged = merge_cross_page_text_blocks(structured_data)

    assert merged["pages"][0]["text"] == "迟到超过二十分钟按旷课处理。"
    assert merged["pages"][1]["text"].startswith("二、宿舍纪律")
