from pathlib import Path

from app.evals.intent_dataset import load_intent_dataset, summarize_intent_dataset


def test_intent_dataset_seed_loads_with_expected_shape() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    dataset_path = backend_root / "evals" / "intent_dataset_seed.jsonl"

    examples = load_intent_dataset(dataset_path)
    summary = summarize_intent_dataset(examples)

    assert len(examples) == 24
    assert len({example.id for example in examples}) == len(examples)
    assert summary["labels"] == {
        "business_query": 6,
        "follow_up_query": 6,
        "off_topic": 6,
        "meaningless_input": 6,
    }
    assert summary["splits"] == {
        "train": 20,
        "validation": 4,
    }
    assert summary["follow_up_with_history"] == 6
