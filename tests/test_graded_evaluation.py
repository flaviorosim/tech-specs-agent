
import csv

import pytest

from tech_specs.graded_evaluation import (
    discounted_gain,
    load_judgments,
    score_graded,
    summarize_graded,
)


def labels():
    return {
        "a": {"grade": 2, "verify_pdf": False},
        "b": {"grade": 1, "verify_pdf": False},
        "c": {"grade": 0, "verify_pdf": False},
    }


def test_ideal_ranking_has_ndcg_one():

    result = score_graded(
        ["a", "b", "c"],
        labels(),
        k=3,
    )

    assert result["pooled_ndcg_at_k"] == pytest.approx(1)
    assert result["hit_grade2_at_k"] == 1
    assert result["mrr_grade2_at_k"] == 1.0


def test_worse_ranking_and_partial_credit():

    result = score_graded(
        ["b", "c", "a"],
        labels(),
        k=3,
    )

    assert 0 < result["pooled_ndcg_at_k"] < 1
    assert result["first_grade2_rank"] == 3
    assert result["mrr_grade2_at_k"] == pytest.approx(1 / 3)

    assert discounted_gain([0, 0]) == 0


def test_unjudged_and_duplicate_hits_fail():

    with pytest.raises(ValueError, match="Unjudged"):
        score_graded(
            ["not_in_pool"],
            labels(),
        )

    with pytest.raises(ValueError, match="Duplicate"):
        score_graded(
            ["a", "a"],
            labels(),
        )

    with pytest.raises(ValueError, match="k must"):
        score_graded(
            ["a"],
            labels(),
            k=0,
        )


def test_macro_average():

    first = score_graded(
        ["a", "b"],
        labels(),
        k=2,
    )

    second = score_graded(
        ["c", "a"],
        labels(),
        k=2,
    )

    summary = summarize_graded([first, second])

    assert summary["questions"] == 2
    assert summary["mrr_grade2_at_k"] == pytest.approx(0.75)


def test_csv_bom_and_pending_review(tmp_path):

    path = tmp_path / "judgments.csv"

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "question_id",
            "query",
            "expected_document",
            "chunk_id",
            "relevance",
            "verificar_pdf",
        ])

        writer.writerow([
            "q1", "TIM2", "stm32l432kc",
            "c1", "2", "SIM",
        ])

        writer.writerow([
            "q1", "TIM2", "stm32l432kc",
            "c2", "1", "NÃO",
        ])

    judgments, questions = load_judgments(path)

    assert questions["q1"]["document"] == "stm32l432kc"

    assert judgments["q1"]["c1"]["grade"] == 2
    assert judgments["q1"]["c1"]["verify_pdf"] is True

    result = score_graded(
        ["c1", "c2"],
        judgments["q1"],
        k=2,
    )

    assert result["pdf_checks_in_top_k"] == 1
