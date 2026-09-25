"""纸质谱练习页签规划核心。

背景
====
已保存的翻页方案给出每个小节的页码。练习者在纸质谱上为**常练小节**贴
索引页签：练习时从最近的页签翻找目标页。请求给出若干常练小节及各自的
**正整数查找权重**与页签数 K；同一页上的多个常练小节（含重复列出的同
一小节）按方案合并为一个**目标页**，权重相加。

规则
====
- 页签只能贴在有练习目标的页上，恰好 K 个、页码互不相同；
- K 必须在 1 到目标页数之间；
- 每个目标页从**最近页签**开始查找，代价 = 页码距离 × 该页权重；
- 全局最小化总代价；同代价时按**升序页签列表的字典序**取最小；
- 输出每个目标页的最近页签，距离并列时取**较小页码**的页签。

算法（一维 K-median，中心限目标页集合）
========================================
目标页按页码升序 p[0..M-1]、合并权重 w[0..M-1]；权重与页码都是整数，
所有代价均为整数。直线上的 K-median 最优解可把目标页划分为 K 个**连续
段**、每段一个页签（位于段内加权中位数处，必为目标页）。后缀 DP：

    f[t][s] = 用 t 个页签服务目标页 s..M-1 的最小代价
            = min_e cost(s, e) + f[t-1][e+1]

其中 cost(s, e) 为段 [s..e] 内最优单页签代价（段内加权中位数）。段代价
满足四边形不等式，最优 e 随 s 单调不减，用**分治优化**以 O(K·M log M)
求值，并记录每个 (t, s) 最小的最优 e；cost 表用双指针（段终点右移时加
权中位数单调右移）O(M²) 预处理，查询 O(1)。

字典序最小重构：固定全局最优时，第 i 段的终点 e 必须是该状态下的最优
段终点、页签必须是段 [s..e] 内的最优（加权中位数）位置；而段的"最左
加权中位数"随 e 单调不减，故每步取最小最优段终点 e*（DP 已记录）、页
签取段 [s..e*] 的最左加权中位数，即得字典序最小的页签列表，O(K log M)。

本模块仅依赖标准库，便于脱离 Web 层直接单测。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import List, Sequence, Tuple

#: 常练小节条数上限
MAX_PRACTICE_ITEMS = 500

#: 同页合并后的目标页数上限
MAX_TARGET_PAGES = 500


class PracticeTabError(ValueError):
    """练习页签规划输入不合法：空目标、无效小节、权重或页签数等。"""


@dataclass(frozen=True)
class TabTarget:
    """一个目标页的查找结果。

    :param page_index: 目标页页码（1 基）
    :param weight: 该页合并后的查找权重
    :param nearest_tab: 最近页签页码（距离并列取较小页码）
    :param distance: 到最近页签的页码距离
    """

    page_index: int
    weight: int
    nearest_tab: int
    distance: int


@dataclass(frozen=True)
class TabPlan:
    """练习页签规划结果。

    :param tabs: 页签页码（1 基，升序，恰好 K 个）
    :param total_cost: 最小总代价（各目标页权重 × 到最近页签页码距离之和）
    :param targets: 每个目标页的最近页签（按页码升序）
    """

    tabs: List[int]
    total_cost: int
    targets: List[TabTarget]


def _validate(
    measure_count: int,
    practice: Sequence[Tuple[int, int]],
    tab_count: int,
) -> List[Tuple[int, int]]:
    """校验规划小节数、常练小节列表与页签数，返回规整后的（小节, 权重）对。"""
    if (
        not isinstance(measure_count, int)
        or isinstance(measure_count, bool)
        or measure_count < 1
    ):
        raise PracticeTabError(f"规划小节数必须为正整数（实际 {measure_count!r}）")
    if isinstance(practice, (str, bytes)) or not isinstance(practice, Sequence):
        raise PracticeTabError("practice_measures 必须是（小节序号, 权重）对的数组")
    if not practice:
        raise PracticeTabError("常练小节列表不能为空：没有练习目标，无法规划页签")
    if len(practice) > MAX_PRACTICE_ITEMS:
        raise PracticeTabError(
            f"常练小节条数（{len(practice)}）超过上限 {MAX_PRACTICE_ITEMS}，无法规划"
        )

    checked: List[Tuple[int, int]] = []
    for idx, item in enumerate(practice):
        try:
            measure, weight = item
        except (TypeError, ValueError) as exc:
            raise PracticeTabError(
                f"第 {idx + 1} 个常练小节必须是（小节序号, 权重）对"
                f"（收到 {item!r}）"
            ) from exc
        if not isinstance(measure, int) or isinstance(measure, bool):
            raise PracticeTabError(
                f"第 {idx + 1} 个常练小节序号必须为整数（收到 {measure!r}），"
                "不接受布尔值或小数"
            )
        if not 0 <= measure < measure_count:
            raise PracticeTabError(
                f"第 {idx + 1} 个常练小节序号 {measure} 越界："
                f"合法范围为 0..{measure_count - 1}"
            )
        if not isinstance(weight, int) or isinstance(weight, bool):
            raise PracticeTabError(
                f"第 {idx + 1} 个常练小节的查找权重必须为正整数"
                f"（收到 {weight!r}），不接受布尔值或小数"
            )
        if weight < 1:
            raise PracticeTabError(
                f"第 {idx + 1} 个常练小节的查找权重必须为正整数（实际 {weight}）"
            )
        checked.append((measure, weight))

    if not isinstance(tab_count, int) or isinstance(tab_count, bool):
        raise PracticeTabError(
            f"页签数必须为整数（收到 {tab_count!r}），不接受布尔值或小数"
        )
    return checked


def _prefix_sums(p: Sequence[int], w: Sequence[int]) -> Tuple[List[int], List[int]]:
    """权重与"权重×页码"的前缀和（长度 M+1，首元素 0）。"""
    pw = [0]
    pwp = [0]
    for pi, wi in zip(p, w):
        pw.append(pw[-1] + wi)
        pwp.append(pwp[-1] + wi * pi)
    return pw, pwp


def _range_cost(
    p: Sequence[int],
    pw: Sequence[int],
    pwp: Sequence[int],
    s: int,
    e: int,
    c: int,
) -> int:
    """段 [s..e] 内所有目标页到页签 c（s ≤ c ≤ e）的加权页码距离和。"""
    pc = p[c]
    left = pc * (pw[c + 1] - pw[s]) - (pwp[c + 1] - pwp[s])
    right = (pwp[e + 1] - pwp[c + 1]) - pc * (pw[e + 1] - pw[c + 1])
    return left + right


def _build_cost_table(p: Sequence[int], w: Sequence[int]) -> List[List[int]]:
    """cost[s][e] = 段 [s..e] 的最优单页签代价（页签取段内加权中位数）。

    段终点 e 右移时加权中位数单调右移，双指针均摊 O(1)，整体 O(M²)。
    """
    m = len(p)
    pw, pwp = _prefix_sums(p, w)
    cost = [[0] * m for _ in range(m)]
    for s in range(m):
        row = cost[s]
        med = s
        right_w = 0  # (med, e] 的权重和
        total = 0
        for e in range(s, m):
            total += w[e]
            if e > med:
                right_w += w[e]
            # 右侧权重超过一半：中位数右移（左侧权重因此仍不超过一半）
            while 2 * right_w > total:
                right_w -= w[med + 1]
                med += 1
            row[e] = _range_cost(p, pw, pwp, s, e, med)
    return cost


def _suffix_dp(
    cost: Sequence[Sequence[int]], m: int, tab_count: int
) -> Tuple[List[List[float]], List[List[int]]]:
    """后缀 DP：values[t][s] = 用 t 个页签服务目标页 s..m-1 的最小代价。

    转移 values[t][s] = min_e cost[s][e] + values[t-1][e+1]（e ∈ [s, m-t]，
    剩余目标页须够分）。最优 e 随 s 单调不减（段代价满足四边形不等式），
    分治优化 O(K·m log m)；opts[t][s] 记录**最小**的最优 e，供字典序重构。
    values[t][s] 仅对 s ≤ m-t 有意义，其余为 inf。
    """
    inf = float("inf")
    values: List[List[float]] = [[inf] * (m + 1)]
    values[0][m] = 0  # 0 个页签服务空段
    opts: List[List[int]] = [[0] * (m + 1)]

    for t in range(1, tab_count + 1):
        prev = values[t - 1]
        cur = [inf] * (m + 1)
        opt = [0] * (m + 1)
        hi = m - t  # s 与 e 的共同上界

        def solve(s_lo: int, s_hi: int, e_lo: int, e_hi: int) -> None:
            if s_lo > s_hi:
                return
            smid = (s_lo + s_hi) // 2
            row = cost[smid]
            best_v = inf
            best_e = max(e_lo, smid)  # 段非空要求 e ≥ s
            for e in range(best_e, e_hi + 1):
                v = row[e] + prev[e + 1]
                if v < best_v:  # 严格小于才更新：并列取最小 e
                    best_v = v
                    best_e = e
            cur[smid] = best_v
            opt[smid] = best_e
            solve(s_lo, smid - 1, e_lo, best_e)
            solve(smid + 1, s_hi, best_e, e_hi)

        solve(0, hi, 0, hi)
        values.append(cur)
        opts.append(opt)
    return values, opts


def _leftmost_median(pw: Sequence[int], s: int, e: int) -> int:
    """段 [s..e] 的最左加权中位数下标（在权重前缀和上二分）。

    即最小 c 使右侧权重 pw[e+1]-pw[c+1] 不超过段总权重的一半
    （⟺ 2·pw[c+1] ≥ pw[s]+pw[e+1]）；该位置必为段内最优页签之一。
    """
    threshold = (pw[s] + pw[e + 1] + 1) // 2
    return bisect_left(pw, threshold, s + 1, e + 2) - 1


def _reconstruct(
    p: Sequence[int],
    w: Sequence[int],
    opts: Sequence[Sequence[int]],
    m: int,
    tab_count: int,
) -> List[int]:
    """重构字典序最小的页签列表（目标页下标）。

    固定全局最优时，每段的终点必须是该状态下的最优段终点、页签必须是段
    内最优（加权中位数）位置；最左加权中位数随段终点单调不减，故每步取
    最小最优段终点 e*（DP 已记录）、页签取段 [s..e*] 的最左加权中位数。
    """
    pw, _ = _prefix_sums(p, w)
    tabs: List[int] = []
    s = 0
    for t in range(tab_count, 0, -1):
        e = opts[t][s]
        tabs.append(_leftmost_median(pw, s, e))
        s = e + 1
    return tabs


def _nearest_tabs(
    p: Sequence[int], w: Sequence[int], tabs: Sequence[int]
) -> List[TabTarget]:
    """每个目标页的最近页签（距离并列取较小页码）。"""
    targets: List[TabTarget] = []
    for pi, wi in zip(p, w):
        pos = bisect_left(tabs, pi)
        best_tab = -1
        best_d = -1
        # 最近页签必是二分插入点左右之一；左侧（页码较小）先查，
        # 右侧仅在严格更近时替换，故距离并列保留较小页码
        for j in (pos - 1, pos):
            if 0 <= j < len(tabs):
                d = abs(pi - tabs[j])
                if best_d < 0 or d < best_d:
                    best_d = d
                    best_tab = tabs[j]
        targets.append(
            TabTarget(page_index=pi, weight=wi, nearest_tab=best_tab, distance=best_d)
        )
    return targets


def place_practice_tabs(
    measure_count: int,
    page_starts: Sequence[int],
    practice: Sequence[Tuple[int, int]],
    tab_count: int,
) -> TabPlan:
    """为已保存方案规划纸质谱练习页签。

    :param measure_count: 规划小节数（正整数）
    :param page_starts: 每页起始小节序号（0 基，升序，首元素为 0）
    :param practice: 常练小节（0 基序号, 正整数查找权重）对，非空；同页
        （含重复列出的同一小节）权重合并
    :param tab_count: 页签数 K，须在 1 到目标页数之间
    :return: 页签页码（升序）、最小总代价与每个目标页的最近页签
    :raises PracticeTabError: 空目标、无效小节、权重或页签数等
    """
    items = _validate(measure_count, practice, tab_count)

    # 小节 -> 1 基页码；同页（含重复小节）权重合并
    starts = list(page_starts)
    merged: dict[int, int] = {}
    for measure, weight in items:
        page = bisect_right(starts, measure)  # 页 i 覆盖 [starts[i], starts[i+1])
        merged[page] = merged.get(page, 0) + weight

    positions = sorted(merged)
    m = len(positions)
    if m > MAX_TARGET_PAGES:
        raise PracticeTabError(
            f"目标页数（{m}）超过上限 {MAX_TARGET_PAGES}，无法规划"
        )
    if not 1 <= tab_count <= m:
        raise PracticeTabError(
            f"页签数必须在 1 到目标页数 {m} 之间（实际 {tab_count}）"
        )
    weights = [merged[page] for page in positions]

    cost = _build_cost_table(positions, weights)
    values, opts = _suffix_dp(cost, m, tab_count)
    tab_indices = _reconstruct(positions, weights, opts, m, tab_count)
    tabs = [positions[c] for c in tab_indices]
    targets = _nearest_tabs(positions, weights, tabs)

    total_cost = values[tab_count][0]
    # 自洽性：按"每个目标页找最近页签"重算的加权和必等于 DP 最优值
    assert sum(t.weight * t.distance for t in targets) == total_cost
    return TabPlan(tabs=tabs, total_cost=total_cost, targets=targets)
