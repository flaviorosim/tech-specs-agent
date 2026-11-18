import pytest

from tech_specs.evaluation import score_ranking, summarize_scores


def test_correct_hit_at_second_position():
    result = score_ranking(["noise", "gold", "other"], ["gold"], k=5)
    assert result["hit_at_k"] == 1
    assert result["recall_at_k"] == 1.0
    assert result["first_relevant_rank"] == 2
    assert result["reciprocal_rank_at_k"] == 0.5


def test_partial_recall_for_multiple_relevant_chunks():
    result = score_ranking(["one", "gold_a", "other"], ["gold_a", "gold_b"], k=3)
    assert result["hit_at_k"] == 1
    assert result["recall_at_k"] == 0.5
    assert result["reciprocal_rank_at_k"] == 0.5


def test_missing_relevant_chunks_and_k_cutoff():
    result = score_ranking(["noise", "gold"], ["gold"], k=1)
    assert result["hit_at_k"] == 0
    assert result["recall_at_k"] == 0.0
    assert result["reciprocal_rank_at_k"] == 0.0
    assert result["first_relevant_rank"] is None


def test_macro_average_and_parameter_validation():
    a = score_ranking(["gold"], ["gold"], k=2)
    b = score_ranking([], ["other"], k=2)
    summary = summarize_scores([a, b])
    assert summary["hit_rate_at_k"] == 0.5
    assert summary["mean_recall_at_k"] == 0.5
    assert summary["mrr_at_k"] == 0.5

    with pytest.raises(ValueError):
        score_ranking(["anything"], ["gold"], k=0)
    with pytest.raises(ValueError):
        score_ranking(["anything"], [], k=1)
