import numpy as np



def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)



def mmr_rerank(
    query_embedding: list[float],
    doc_embeddings: list[list[float]],
    top_k: int = 5,
    lambda_mult: float = 0.5,
) -> list[int]:
    if not doc_embeddings:
        return []

    doc_embeddings_array = np.array(doc_embeddings, dtype=float)
    query_embedding_array = np.array(query_embedding, dtype=float)

    query_doc_similarities = np.array(
        [_cosine_similarity(query_embedding_array, embedding) for embedding in doc_embeddings_array],
        dtype=float,
    )

    unselected = list(range(len(doc_embeddings_array)))
    selected: list[int] = []

    best_idx = int(np.argmax(query_doc_similarities))
    selected.append(best_idx)
    unselected.remove(best_idx)

    while len(selected) < top_k and unselected:
        best_score = float("-inf")
        best_idx_to_add = -1

        for index in unselected:
            sim_to_query = query_doc_similarities[index]
            max_sim_to_selected = max(
                _cosine_similarity(doc_embeddings_array[index], doc_embeddings_array[selected_index])
                for selected_index in selected
            )
            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx_to_add = index

        selected.append(best_idx_to_add)
        unselected.remove(best_idx_to_add)

    return selected
