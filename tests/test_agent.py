# ============================================================
# test_agent.py —— agent.py 的单元测试：调度循环与兜底路径（不联网）
# ============================================================
import json
from types import SimpleNamespace

import agent
from agent import MAX_ROUNDS, run_agent, save_trace


# ---------------- 测试替身：一个不联网的假客户端 ----------------

class FakeClient:
    """按预设顺序吐出响应，用于替代真实的 OpenAI 客户端。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    # 让 client.chat.completions.create(...) 这条调用链成立
    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def reply(content):
    """构造"模型直接作答、不调用工具"的响应。"""
    message = SimpleNamespace(role="assistant", content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_request(name, arguments="{}", call_id="call_1"):
    """构造"模型请求调用工具"的响应。"""
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# ---------------- 主流程 ----------------

def test_direct_answer_without_tool_calls():
    client = FakeClient([reply("你好")])

    answer, trace = run_agent(client, "你好", "sys")

    assert answer == "你好"
    assert trace == []
    assert len(client.requests) == 1


def test_tool_call_is_executed_and_result_fed_back():
    client = FakeClient([
        tool_request("calculate", '{"expression": "12*12"}'),
        reply("144"),
    ])

    answer, trace = run_agent(client, "12*12", "sys")

    assert answer == "144"
    assert len(trace) == 1
    assert trace[0]["round"] == 1
    assert trace[0]["tool"] == "calculate"
    assert trace[0]["args"] == {"expression": "12*12"}
    assert trace[0]["result"] == "144"


def test_messages_grow_in_expected_order():
    """第二次请求应带上 assistant(工具请求) 与 tool(执行结果) 两条消息。"""
    client = FakeClient([
        tool_request("get_current_time"),
        reply("现在是某个时间"),
    ])

    run_agent(client, "现在几点", "sys")

    messages = client.requests[1]["messages"]
    roles = [m["role"] if isinstance(m, dict) else m.role for m in messages]
    assert roles == ["system", "user", "assistant", "tool"]


def test_history_is_placed_between_system_and_current_question():
    client = FakeClient([reply("ok")])
    history = [
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好，小明"},
    ]

    run_agent(client, "我叫什么", "sys", history)

    messages = client.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "我叫什么"}


# ---------------- 兜底路径 ----------------

def test_unknown_tool_name_does_not_crash():
    client = FakeClient([tool_request("no_such_tool"), reply("ok")])

    _, trace = run_agent(client, "q", "sys")

    assert "不存在名为 no_such_tool 的工具" in trace[0]["result"]


def test_tool_exception_is_captured(monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("内部错误")

    monkeypatch.setitem(agent.TOOLS, "boom", boom)
    client = FakeClient([tool_request("boom"), reply("ok")])

    _, trace = run_agent(client, "q", "sys")

    assert trace[0]["result"].startswith("工具执行出错")
    assert "内部错误" in trace[0]["result"]


def test_invalid_json_arguments_fall_back_to_empty_dict():
    client = FakeClient([tool_request("get_current_time", "{不是合法JSON"), reply("ok")])

    _, trace = run_agent(client, "q", "sys")

    assert trace[0]["args"] == {}


def test_loop_stops_after_max_rounds():
    """模型一直要工具不收敛时，应在 MAX_ROUNDS 轮后主动停止。"""
    responses = [
        tool_request("get_current_time", call_id=f"call_{i}")
        for i in range(MAX_ROUNDS)
    ]
    client = FakeClient(responses)

    answer, trace = run_agent(client, "q", "sys")

    assert "任务步骤过多" in answer
    assert len(trace) == MAX_ROUNDS
    assert len(client.requests) == MAX_ROUNDS


# ---------------- 轨迹落盘 ----------------

def test_save_trace_writes_one_json_line_per_step(tmp_path):
    target = tmp_path / "trace.jsonl"

    save_trace(
        "帮我算一下",
        [
            {"round": 1, "tool": "calculate", "args": {"expression": "1+1"}, "result": "2"},
            {"round": 1, "tool": "get_current_time", "args": {}, "result": "2026-01-01 00:00:00"},
        ],
        path=target,
    )

    lines = target.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    record = json.loads(lines[0])
    assert record["question"] == "帮我算一下"
    assert record["tool"] == "calculate"
    assert record["args"] == {"expression": "1+1"}


def test_save_trace_skips_empty_trace(tmp_path):
    target = tmp_path / "trace.jsonl"

    save_trace("闲聊", [], path=target)

    assert not target.exists()
