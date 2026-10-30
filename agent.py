# ============================================================
# agent.py —— Agent 主循环：多轮 Function Calling 调度
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import config
from tools import TOOLS, TOOL_SCHEMAS

# 单次请求允许的最大工具调用轮数，防止模型不收敛导致调用链无限延长。
MAX_ROUNDS = 5

# 工具调用轨迹落盘路径（JSON Lines，每行一条记录）。
TRACE_PATH = Path("agent_trace.jsonl")


def run_agent(
    client,
    user_query: str,
    system_prompt: str,
    history: list | None = None,
) -> tuple[str, list]:
    """执行一轮 Agent 调度，直至模型给出最终答复。

    Args:
        client: OpenAI 兼容客户端。
        user_query: 本轮用户提问。
        system_prompt: 系统提示词。
        history: 历史对话消息（不含本轮 user_query），用于维持多轮上下文。

    Returns:
        (最终答复, 工具调用轨迹)。轨迹元素为
        {"round": int, "tool": str, "args": dict, "result": str}。
    """
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    trace = []

    for round_no in range(1, MAX_ROUNDS + 1):
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        msg = response.choices[0].message

        # 未请求工具，说明模型已给出最终答复。
        if not msg.tool_calls:
            return msg.content or "", trace

        # 保留 assistant 消息：其后的 tool 消息需通过 tool_call_id 与之对应。
        messages.append(msg)

        for call in msg.tool_calls:
            name = call.function.name

            # arguments 为 JSON 字符串，解析失败时退化为空参数。
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            # 由本地代码执行工具；异常转为文本回传，不中断主循环。
            if name not in TOOLS:
                result = f"错误：不存在名为 {name} 的工具。"
            else:
                try:
                    result = TOOLS[name](**args)
                except Exception as exc:
                    result = f"工具执行出错：{exc}"

            trace.append({
                "round": round_no,
                "tool": name,
                "args": args,
                "result": result,
            })

            # 结果以 tool 角色回填，供下一轮模型读取。
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return "任务步骤过多，已停止。请把问题拆成更小的部分再问。", trace


def save_trace(question: str, trace: list, path: Path = TRACE_PATH) -> None:
    """将一次运行的工具调用轨迹以 JSON Lines 追加写入 path。

    仅负责持久化，不参与调度决策；空轨迹不产生任何写入。
    """
    if not trace:
        return

    stamp = datetime.now(config.TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        for step in trace:
            record = {
                "time": stamp,
                "question": question,
                "round": step["round"],
                "tool": step["tool"],
                "args": step["args"],
                "result": str(step["result"]),
            }
            # ensure_ascii=False 保留中文原文，便于人工阅读日志。
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
