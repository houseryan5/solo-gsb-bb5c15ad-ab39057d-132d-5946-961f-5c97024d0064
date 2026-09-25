"""纸质谱练习页签规划接口的端到端测试（FastAPI TestClient + 临时 SQLite）。"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _create_plan(client, page_height=1) -> str:
    """创建并保存一个 4 小节规划：每小节高度 1、静默充足。

    page_height=1 时每小节一页（页 1..4 = 小节 0..3）；
    page_height=2 时每页两小节（页 1 = 小节 0..1，页 2 = 小节 2..3）。
    """
    resp = client.post(
        "/api/v1/plans",
        json={
            "measure_heights": [1] * 4,
            "silence_seconds": [5] * 4,
            "page_height": page_height,
            "min_turn_time": 1,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _post_tabs(client, plan_id, practice, tab_count):
    return client.post(
        f"/api/v1/plans/{plan_id}/practice-tabs",
        json={"practice_measures": practice, "tab_count": tab_count},
    )


def test_tabs_single_tab_two_pages(client):
    plan_id = _create_plan(client, page_height=2)
    resp = _post_tabs(
        client,
        plan_id,
        [
            {"measure": 0, "weight": 3},
            {"measure": 1, "weight": 2},
            {"measure": 2, "weight": 4},
            {"measure": 3, "weight": 1},
        ],
        1,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == plan_id
    assert body["page_count"] == 2
    assert body["target_page_count"] == 2
    # 页 1 权重 3+2=5，页 2 权重 4+1=5：两页同代价 5，字典序取页签 [1]
    assert body["tabs"] == [1]
    assert body["total_cost"] == 5
    assert body["targets"] == [
        {"page_index": 1, "weight": 5, "nearest_tab": 1, "distance": 0},
        {"page_index": 2, "weight": 5, "nearest_tab": 1, "distance": 1},
    ]
    # 请求原样回显
    assert body["request"]["tab_count"] == 1
    assert body["request"]["practice_measures"][0] == {"measure": 0, "weight": 3}


def test_tabs_lexicographic_tie_and_distance_tie(client):
    plan_id = _create_plan(client, page_height=1)
    resp = _post_tabs(
        client,
        plan_id,
        [{"measure": i, "weight": 1} for i in range(4)],
        2,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # {1,3} {1,4} {2,3} {2,4} 同为最优代价 2，字典序最小为 [1, 3]
    assert body["tabs"] == [1, 3]
    assert body["total_cost"] == 2
    assert body["target_page_count"] == 4
    # 页 2 到页签 1、3 距离并列（均为 1），取较小页码 1
    assert [(t["page_index"], t["nearest_tab"], t["distance"]) for t in body["targets"]] == [
        (1, 1, 0),
        (2, 1, 1),
        (3, 3, 0),
        (4, 3, 1),
    ]


def test_tabs_same_page_and_duplicate_measures_merged(client):
    plan_id = _create_plan(client, page_height=2)
    resp = _post_tabs(
        client,
        plan_id,
        [
            {"measure": 0, "weight": 2},
            {"measure": 1, "weight": 3},
            {"measure": 0, "weight": 1},
        ],
        1,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 小节 0、1 同页（页 1），重复小节 0 权重并入：页 1 权重 2+3+1=6
    assert body["target_page_count"] == 1
    assert body["tabs"] == [1]
    assert body["total_cost"] == 0
    assert body["targets"] == [
        {"page_index": 1, "weight": 6, "nearest_tab": 1, "distance": 0}
    ]


def test_tabs_k_equals_target_page_count_zero_cost(client):
    plan_id = _create_plan(client, page_height=1)
    resp = _post_tabs(
        client,
        plan_id,
        [{"measure": 0, "weight": 2}, {"measure": 3, "weight": 5}],
        2,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tabs"] == [1, 4]
    assert body["total_cost"] == 0
    assert [(t["nearest_tab"], t["distance"]) for t in body["targets"]] == [
        (1, 0),
        (4, 0),
    ]


def test_tabs_plan_not_found_returns_404(client):
    resp = _post_tabs(
        client, "does-not-exist", [{"measure": 0, "weight": 1}], 1
    )
    assert resp.status_code == 404
    assert "规划结果不存在" in resp.json()["detail"]


def test_tabs_empty_practice_returns_422(client):
    plan_id = _create_plan(client)
    resp = _post_tabs(client, plan_id, [], 1)
    assert resp.status_code == 422, resp.text
    assert "不能为空" in resp.json()["detail"]


@pytest.mark.parametrize(
    "measure,fragment",
    [
        (-1, "越界"),
        (4, "越界"),
        (99, "越界"),
        (1.5, "整数"),
        (2.0, "整数"),
        (True, "整数"),
        (False, "整数"),
        ("1", "整数"),
        (None, "整数"),
    ],
)
def test_tabs_invalid_measure_returns_422(client, measure, fragment):
    plan_id = _create_plan(client)
    resp = _post_tabs(client, plan_id, [{"measure": measure, "weight": 1}], 1)
    assert resp.status_code == 422, resp.text
    assert fragment in resp.text


@pytest.mark.parametrize(
    "weight,fragment",
    [
        (0, "正整数"),
        (-2, "正整数"),
        (1.5, "整数"),
        (2.0, "整数"),
        (True, "整数"),
        ("3", "整数"),
        (None, "整数"),
    ],
)
def test_tabs_invalid_weight_returns_422(client, weight, fragment):
    plan_id = _create_plan(client)
    resp = _post_tabs(client, plan_id, [{"measure": 0, "weight": weight}], 1)
    assert resp.status_code == 422, resp.text
    assert fragment in resp.text


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
def test_tabs_invalid_tab_count_returns_422(client, tab_count, fragment):
    plan_id = _create_plan(client, page_height=2)
    resp = _post_tabs(
        client,
        plan_id,
        [{"measure": 0, "weight": 1}, {"measure": 2, "weight": 1}],
        tab_count,
    )
    assert resp.status_code == 422, resp.text
    assert fragment in resp.text


def test_tabs_practice_items_over_limit_returns_422(client):
    plan_id = _create_plan(client)
    resp = _post_tabs(
        client,
        plan_id,
        [{"measure": 0, "weight": 1}] * 501,
        1,
    )
    assert resp.status_code == 422, resp.text
    assert "超过上限" in resp.json()["detail"]
    assert "500" in resp.json()["detail"]


def test_tabs_missing_and_extra_fields_rejected(client):
    plan_id = _create_plan(client)
    missing = client.post(f"/api/v1/plans/{plan_id}/practice-tabs", json={})
    assert missing.status_code == 422
    extra = _post_tabs(client, plan_id, [{"measure": 0, "weight": 1}], 1)
    assert extra.status_code == 200
    extra = client.post(
        f"/api/v1/plans/{plan_id}/practice-tabs",
        json={
            "practice_measures": [{"measure": 0, "weight": 1}],
            "tab_count": 1,
            "notes": "x",
        },
    )
    assert extra.status_code == 422
    extra_item = client.post(
        f"/api/v1/plans/{plan_id}/practice-tabs",
        json={
            "practice_measures": [{"measure": 0, "weight": 1, "tag": "x"}],
            "tab_count": 1,
        },
    )
    assert extra_item.status_code == 422


def test_tabs_does_not_modify_saved_plan(client):
    plan_id = _create_plan(client)
    before = client.get(f"/api/v1/plans/{plan_id}").json()
    resp = _post_tabs(client, plan_id, [{"measure": 1, "weight": 2}], 1)
    assert resp.status_code == 200
    after = client.get(f"/api/v1/plans/{plan_id}").json()
    assert before == after
