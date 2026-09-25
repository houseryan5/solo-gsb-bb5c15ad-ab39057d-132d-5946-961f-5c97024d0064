"""纸质谱练习页签规划核心的单元测试。"""

from __future__ import annotations

import random
from bisect import bisect_right
from itertools import combinations

import pytest

import app.practice_tabs as practice_tabs
from app.practice_tabs import (
    MAX_PRACTICE_ITEMS,
    MAX_TARGET_PAGES,
    PracticeTabError,
    place_practice_tabs,
)


def _merge_targets(page_starts, practice):
    """独立实现的同页合并：返回 (页码列表, 权重列表)（页码升序）。"""
    merged = {}
    for measure, weight in practice:
        page = bisect_right(page_starts, measure)
        merged[page] = merged.get(page, 0) + weight
    positions = sorted(merged)
    return positions, [merged[p] for p in positions]


def _brute_force(positions, weights, k):
    """穷举所有 C(M,K) 页签组合：最小总代价 + 字典序最小的升序页签列表。"""
    best_cost = None
    best_tabs = None
    for combo in combinations(range(len(positions)), k):
        tabs = [positions[c] for c in combo]
        cost = sum(
            w * min(abs(p - t) for t in tabs)
            for p, w in zip(positions, weights)
        )
        if (
            best_cost is None
            or cost < best_cost
            or (cost == best_cost and tabs < best_tabs)
        ):
            best_cost, best_tabs = cost, tabs
    return best_cost, best_tabs


def _expected_targets(positions, weights, tabs):
    """按"最近页签、距离并列取较小页码"独立计算每个目标页的报告。"""
    out = []
    for p, w in zip(positions, weights):
        d = min(abs(p - t) for t in tabs)
        nearest = min(t for t in tabs if abs(p - t) == d)
        out.append((p, w, nearest, d))
    return out


def _check(page_starts, practice, k):
    """实现结果与暴力穷举对照（代价、页签列表、每页最近页签报告）。"""
    measure_count = max(page_starts[-1] + 1, max(m for m, _ in practice) + 1)
    result = place_practice_tabs(measure_count, page_starts, practice, k)
    positions, weights = _merge_targets(page_starts, practice)
    best_cost, best_tabs = _brute_force(positions, weights, k)

    assert result.tabs == best_tabs
    assert result.total_cost == best_cost
    assert len(result.tabs) == k
    assert len(set(result.tabs)) == k  # 页签页码互不相同
    assert result.tabs == sorted(result.tabs)  # 升序
    assert set(result.tabs) <= set(positions)  # 页签只贴在目标页

    got = [(t.page_index, t.weight, t.nearest_tab, t.distance) for t in result.targets]
    assert got == _expected_targets(positions, weights, best_tabs)
    # 报告距离的加权和与总代价一致
    assert sum(w * d for _, w, _, d in got) == result.total_cost
    return result


# ---------------------------------------------------------------- 确定性用例


def test_single_target_single_tab_zero_cost():
    result = place_practice_tabs(3, [0], [(1, 7)], 1)
    assert result.tabs == [1]
    assert result.total_cost == 0
    assert [(t.page_index, t.weight, t.nearest_tab, t.distance) for t in result.targets] == [
        (1, 7, 1, 0)
    ]


def test_same_page_weights_merged():
    # 页 1 = 小节 0..1，页 2 = 小节 2..3：小节 0、1 的权重合并到页 1
    result = place_practice_tabs(4, [0, 2], [(0, 3), (1, 2), (2, 4), (3, 1)], 1)
    assert [(t.page_index, t.weight) for t in result.targets] == [(1, 5), (2, 5)]
    # 两页权重相同、代价并列（各为 5），字典序取页签 [1]
    assert result.tabs == [1]
    assert result.total_cost == 5
    assert [(t.nearest_tab, t.distance) for t in result.targets] == [(1, 0), (1, 1)]


def test_duplicate_measure_merged_into_same_page():
    result = place_practice_tabs(4, [0, 2], [(0, 2), (0, 3), (1, 1)], 1)
    assert [(t.page_index, t.weight) for t in result.targets] == [(1, 6)]
    assert result.tabs == [1]
    assert result.total_cost == 0


def test_weighted_median_single_tab():
    # 权重集中在页 2：页签取页 2 而非位置中点
    result = place_practice_tabs(10, [0, 1, 2], [(0, 1), (1, 10), (2, 1)], 1)
    assert result.tabs == [2]
    assert result.total_cost == 2
    # 权重集中在页 1：页签取页 1
    result = place_practice_tabs(10, [0, 1, 2], [(0, 10), (1, 1), (2, 1)], 1)
    assert result.tabs == [1]
    assert result.total_cost == 3


def test_tie_prefers_lexicographically_smallest_tabs():
    # 页 1..4 权重全 1、K=2：{1,3} {1,4} {2,3} {2,4} 同为最优代价 2，取 [1,3]
    result = place_practice_tabs(4, [0, 1, 2, 3], [(0, 1), (1, 1), (2, 1), (3, 1)], 2)
    assert result.tabs == [1, 3]
    assert result.total_cost == 2
    # 页 2 到页签 1、3 距离并列（均为 1），取较小页码 1
    assert [(t.page_index, t.nearest_tab, t.distance) for t in result.targets] == [
        (1, 1, 0),
        (2, 1, 1),
        (3, 3, 0),
        (4, 3, 1),
    ]


def test_k_equals_target_count_every_page_tabbed():
    result = place_practice_tabs(4, [0, 1, 2, 3], [(0, 2), (1, 1), (3, 5)], 3)
    assert result.tabs == [1, 2, 4]
    assert result.total_cost == 0
    assert all(t.distance == 0 and t.nearest_tab == t.page_index for t in result.targets)


def test_gap_pages_distance_by_page_number():
    # 目标页不相邻（页 1、5、9），代价按页码距离计
    result = place_practice_tabs(9, list(range(9)), [(0, 1), (4, 1), (8, 1)], 1)
    assert result.tabs == [5]
    assert result.total_cost == 8
    assert [t.distance for t in result.targets] == [4, 0, 4]


def test_huge_weights_stay_exact_integers():
    big = 10**18
    result = place_practice_tabs(2, [0, 1], [(0, big), (1, 1)], 1)
    assert result.tabs == [1]
    assert result.total_cost == 1
    assert isinstance(result.total_cost, int)


# ---------------------------------------------------------------- 随机对照


@pytest.mark.parametrize("seed", range(80))
def test_random_cases_match_brute_force(seed):
    rng = random.Random(seed)
    measure_count = rng.randint(1, 12)
    # 随机分页：从 1..n-1 选若干断页位置
    cuts = sorted(rng.sample(range(1, measure_count), rng.randint(0, measure_count - 1)))
    page_starts = [0] + cuts
    item_count = rng.randint(1, 8)
    practice = [
        (rng.randrange(measure_count), rng.randint(1, 5))
        for _ in range(item_count)
    ]
    positions, _ = _merge_targets(page_starts, practice)
    k = rng.randint(1, len(positions))
    _check(page_starts, practice, k)


# ---------------------------------------------------------------- 非法输入


@pytest.mark.parametrize(
    "practice,fragment",
    [
        ([], "不能为空"),
        ([(0, 1)] * (MAX_PRACTICE_ITEMS + 1), "超过上限"),
        ([(-1, 1)], "越界"),
        ([(4, 1)], "越界"),
        ([(1.5, 1)], "整数"),
        ([(True, 1)], "整数"),
        ([("1", 1)], "整数"),
        ([(None, 1)], "整数"),
        ([(1, 0)], "正整数"),
        ([(1, -2)], "正整数"),
        ([(1, 1.5)], "正整数"),
        ([(1, True)], "正整数"),
        ([(1, "3")], "正整数"),
        ([(1, None)], "正整数"),
        ([1], "（小节序号, 权重）对"),
        ([(1, 2, 3)], "（小节序号, 权重）对"),
    ],
)
def test_invalid_practice_raises(practice, fragment):
    with pytest.raises(PracticeTabError) as exc:
        place_practice_tabs(4, [0, 1, 2, 3], practice, 1)
    assert fragment in str(exc.value)


@pytest.mark.parametrize(
    "tab_count,fragment",
    [
        (0, "1 到目标页数"),
        (-1, "1 到目标页数"),
        (3, "1 到目标页数 2"),  # 目标页数为 2
        (1.5, "整数"),
        (True, "整数"),
        ("1", "整数"),
        (None, "整数"),
    ],
)
def test_invalid_tab_count_raises(tab_count, fragment):
    with pytest.raises(PracticeTabError) as exc:
        place_practice_tabs(4, [0, 2], [(0, 1), (2, 1)], tab_count)
    assert fragment in str(exc.value)


def test_tab_count_checked_against_merged_target_pages():
    # 4 个小节落在 2 个目标页上：K=3 非法
    with pytest.raises(PracticeTabError) as exc:
        place_practice_tabs(4, [0, 2], [(0, 1), (1, 1), (2, 1), (3, 1)], 3)
    assert "目标页数 2" in str(exc.value)


def test_target_page_limit_raises(monkeypatch):
    monkeypatch.setattr(practice_tabs, "MAX_TARGET_PAGES", 2)
    with pytest.raises(PracticeTabError) as exc:
        place_practice_tabs(4, [0, 1, 2, 3], [(0, 1), (1, 1), (2, 1)], 1)
    assert "目标页数" in str(exc.value)
    assert "超过上限" in str(exc.value)


def test_target_page_limit_boundary_accepted():
    practice = [(i, 1) for i in range(MAX_TARGET_PAGES)]
    result = place_practice_tabs(
        MAX_TARGET_PAGES, list(range(MAX_TARGET_PAGES)), practice, 1
    )
    assert len(result.targets) == MAX_TARGET_PAGES


# ---------------------------------------------------------------- 规模性能


def test_large_scale_performance():
    rng = random.Random(20260925)
    page_count = MAX_TARGET_PAGES
    practice = [(i, rng.randint(1, 100)) for i in range(page_count)]
    result = place_practice_tabs(page_count, list(range(page_count)), practice, 100)
    assert len(result.tabs) == 100
    assert sum(t.weight * t.distance for t in result.targets) == result.total_cost
