import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_item_found_by_both_retrievers_wins() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "keyword": ["c", "d"]}, k=60)
    # "c" is 3rd and 1st: 1/63 + 1/61 beats "a" at 1st in one list only (1/61).
    assert fused[0].id == "c"
    assert fused[0].ranks == {"dense": 3, "keyword": 1}
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)
    # "b" and "d" are both 2nd in one list (1/62 each): a tie, kept in input order.
    assert [item.id for item in fused] == ["c", "a", "b", "d"]


def test_ties_break_by_best_single_rank() -> None:
    # "x" and "y" both score 1/61: equal, so order is stable by best rank then insertion.
    fused = reciprocal_rank_fusion({"dense": ["x"], "keyword": ["y"]}, k=60)
    assert {item.id for item in fused} == {"x", "y"}
    assert fused[0].score == fused[1].score


def test_single_list_keeps_its_order() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"]})
    assert [item.id for item in fused] == ["a", "b", "c"]
    assert fused[1].ranks == {"dense": 2}


def test_empty_input() -> None:
    assert reciprocal_rank_fusion({"dense": [], "keyword": []}) == []


def test_smaller_k_rewards_top_ranks_more() -> None:
    lists = {"dense": ["a", "b"], "keyword": ["b", "c"]}
    steep = reciprocal_rank_fusion(lists, k=1)
    flat = reciprocal_rank_fusion(lists, k=1000)
    assert steep[0].score / steep[-1].score > flat[0].score / flat[-1].score
