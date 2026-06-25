"""OCR 引擎封装。

OCR 是可选能力：依赖缺失或识别失败时返回空字符串，让文档解析主流程继续处理可抽取文本。
"""

from functools import lru_cache
from dataclasses import dataclass
import re

from .image_ocr_postprocess import normalize_image_ocr_lines


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    avg_confidence: float | None = None
    min_confidence: float | None = None
    low_confidence_line_count: int = 0
    quality: str = "unknown"
    quality_reasons: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def get_ocr_engine():
    """懒加载并缓存 RapidOCR，避免每张图片重复初始化模型。"""
    try:
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()
    except Exception:
        # OCR 依赖在轻量部署中可能不安装，失败时安静降级。
        return None


def run_ocr_on_image(image_path: str) -> str:
    """对单张图片执行 OCR，失败时返回空文本。"""
    return run_ocr_on_image_result(image_path).text


def run_ocr_on_image_result(image_path: str) -> OCRResult:
    """对单张图片执行 OCR，并输出轻量质量评估。"""
    ocr_engine = get_ocr_engine()
    if ocr_engine is None:
        return OCRResult(text="", quality="unavailable", quality_reasons=("ocr_engine_unavailable",))

    try:
        result, _ = ocr_engine(image_path)
        if not result:
            return OCRResult(text="", quality="empty", quality_reasons=("empty_ocr_result",))

        texts = [item[1] for item in result if len(item) >= 2]
        confidences = [
            float(item[2])
            for item in result
            if len(item) >= 3 and isinstance(item[2], (int, float))
        ]
        normalized_text = normalize_image_ocr_text("\n".join(texts).strip())
        return build_ocr_result(normalized_text, confidences)
    except Exception:
        # 单张图片异常不应中断整份文档入库。
        return OCRResult(text="", quality="failed", quality_reasons=("ocr_exception",))


def normalize_image_ocr_text(text: str) -> str:
    """复用图片 OCR 行清洗逻辑，输出统一换行文本。"""
    return "\n".join(normalize_image_ocr_lines(text)).strip()


def build_ocr_result(text: str, confidences: list[float] | None = None) -> OCRResult:
    """基于 OCR 置信度和文本形态做轻量质量标记。"""
    confidence_values = confidences or []
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
    min_confidence = min(confidence_values) if confidence_values else None
    low_confidence_line_count = sum(1 for score in confidence_values if score < 0.55)

    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty_text")
    if avg_confidence is not None and avg_confidence < 0.72:
        reasons.append("low_avg_confidence")
    if min_confidence is not None and min_confidence < 0.45:
        reasons.append("very_low_min_confidence")
    if low_confidence_line_count >= 2:
        reasons.append("many_low_confidence_lines")
    if _has_suspicious_ocr_shape(text):
        reasons.append("suspicious_text_shape")

    quality = "low" if reasons else "normal"
    if not confidence_values and text.strip():
        quality = "unknown"
        reasons.append("missing_confidence")

    return OCRResult(
        text=text,
        avg_confidence=avg_confidence,
        min_confidence=min_confidence,
        low_confidence_line_count=low_confidence_line_count,
        quality=quality,
        quality_reasons=tuple(reasons),
    )


def _has_suspicious_ocr_shape(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 8:
        return False
    non_content_chars = sum(1 for char in compact if not re.match(r"[\u4e00-\u9fffA-Za-z0-9，。；：、！？（）()/.-]", char))
    return non_content_chars / len(compact) >= 0.18
