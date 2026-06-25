from functools import lru_cache

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


@lru_cache
def _load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


class KnowledgeBaseRouteEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, text: str, *, is_query: bool = False) -> list[float]:
        tokenizer, model = _load_model(self.model_name)
        encoded_text = f"为这个句子生成表示以用于知识库路由：{text}" if is_query else text
        inputs = tokenizer(
            encoded_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        normalized = functional.normalize(embeddings, p=2, dim=1)
        return normalized.squeeze().tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_tensor = torch.tensor(left, dtype=torch.float32)
    right_tensor = torch.tensor(right, dtype=torch.float32)
    return float(functional.cosine_similarity(left_tensor, right_tensor, dim=0).item())
