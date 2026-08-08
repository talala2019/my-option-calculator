import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import cluster_levels


def test_empty_candidates_returns_empty():
    assert cluster_levels([], 100.0) == []


def test_zero_price_returns_empty_instead_of_dividing_by_zero():
    candidates = [{"Price": 10.0, "Type": "MA5"}]
    assert cluster_levels(candidates, 0.0) == []


def test_single_candidate_becomes_one_cluster():
    candidates = [{"Price": 110.0, "Type": "MA20"}]
    result = cluster_levels(candidates, 100.0)
    assert len(result) == 1
    assert result[0]["Price"] == 110.0
    assert result[0]["Pct"] == 10.0
    assert result[0]["Count"] == 1
    assert result[0]["Types"] == ["MA20"]


def test_nearby_candidates_merge_into_one_cluster():
    # 110.0 and 110.5 are within 1.5% of each other -> should merge.
    candidates = [
        {"Price": 110.0, "Type": "MA20"},
        {"Price": 110.5, "Type": "波段壓力"},
    ]
    result = cluster_levels(candidates, 100.0)
    assert len(result) == 1
    assert result[0]["Count"] == 2
    assert result[0]["Types"] == ["MA20", "波段壓力"]
    assert result[0]["Price"] == 110.25  # mean of the two


def test_distant_candidates_stay_separate():
    # 110.0 and 130.0 are far apart (>1.5%) -> two distinct clusters.
    candidates = [
        {"Price": 110.0, "Type": "MA20"},
        {"Price": 130.0, "Type": "MA60"},
    ]
    result = cluster_levels(candidates, 100.0)
    assert len(result) == 2


def test_duplicate_type_in_same_cluster_not_repeated():
    candidates = [
        {"Price": 110.0, "Type": "整數心理關卡"},
        {"Price": 110.1, "Type": "整數心理關卡"},
    ]
    result = cluster_levels(candidates, 100.0)
    assert len(result) == 1
    assert result[0]["Types"] == ["整數心理關卡"]  # not listed twice
