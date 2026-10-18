import pytest

from ragsource.evaluation import evaluate_ranking, mean_metrics


def test_ranking_metrics_with_multiple_relevant_chunks():
    result = evaluate_ranking(["a", "b", "c", "d"], {"b", "d"}, 3)
    assert result.precision_at_k == pytest.approx(1 / 3)
    assert result.recall_at_k == pytest.approx(1 / 2)
    assert result.reciprocal_rank_at_k == pytest.approx(1 / 2)


def test_first_hit_at_rank_one():
    result = evaluate_ranking(["a", "b"], {"a"}, 2)
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.reciprocal_rank_at_k == pytest.approx(1.0)


def test_empty_ground_truth_has_zero_recall():
    result = evaluate_ranking(["a"], set(), 5)
    assert result.precision_at_k == 0
    assert result.recall_at_k == 0
    assert result.reciprocal_rank_at_k == 0


def test_no_retrieved_chunks():
    result = evaluate_ranking([], {"missing"}, 2)
    assert result.precision_at_k == 0
    assert result.recall_at_k == 0
    assert result.reciprocal_rank_at_k == 0


@pytest.mark.parametrize("ranked,relevant,k", [
    (["a", "a"], {"a"}, 2),
    ([""], {"a"}, 1),
    (["a"], {"a"}, 0),
])
def test_invalid_rankings(ranked, relevant, k):
    with pytest.raises(ValueError):
        evaluate_ranking(ranked, relevant, k)


def test_mean_metrics():
    first = evaluate_ranking(["a", "b"], {"a"}, 2)
    second = evaluate_ranking(["b", "a"], {"a"}, 2)
    mean = mean_metrics([first, second])
    assert mean.precision_at_k == pytest.approx(0.5)
    assert mean.recall_at_k == pytest.approx(1)
    assert mean.reciprocal_rank_at_k == pytest.approx(0.75)


def test_mean_requires_queries():
    with pytest.raises(ValueError):
        mean_metrics([])
