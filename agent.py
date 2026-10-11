# ============================================================
# agent.py —— Agent 主循环：让模型自己决定调不调工具、调哪个
# ============================================================
import json
from datetime import datetime
from pathlib import Path

import config
from tools import TOOLS, TOOL_SCHEMAS

# 护栏：最多允许模型连续调用几轮工具。防止它不收敛、无限要工具。
MAX_ROUNDS = 5

# 轨迹落盘的文件（jsonl = 每行一个独立 JSON，方便追加和逐行读取）
TRACE_PATH = Path("agent_trace.jsonl")


def run_agent(
    client,
    user_query: str,
    system_prompt: str,
    history: list | None = None,
) -> tuple[str, list]:
    """跑一轮 Agent，直到模型给出最终回答。

    history：之前几轮的对话（形如 [{"role": ..., "content": ...}, ...]），
             不含本次的 user_query。传进来模型才能"记得"上文。
    返回 (最终回答, 工具调用轨迹)
    """
    # ① 开局：系统设定 + 历史对话 + 用户这次的问题
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    trace = []

    for round_no in range(1, MAX_ROUNDS + 1):
        # ② 把整个对话 + 工具清单发给模型
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        msg = response.choices[0].message

        # ③ 模型没请求工具 → 说明它给最终回答了，收工
        if not msg.tool_calls:
            return msg.content or "", trace

        # 模型请求了工具 → 先把这条"请求"原样存回对话（不能丢，后面要对 id）
        messages.append(msg)

        # ④ 模型可能一次请求多个工具，逐个处理
        for call in msg.tool_calls:
            name = call.function.name

            # 模型给的参数是一段 JSON 文本，得先解析成字典
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            # ⑤ 真正执行工具——这一步是"你的代码在干活"
            if name not in TOOLS:
                # 兜底一：模型编造了一个不存在的工具名
                result = f"错误：不存在名为 {name} 的工具。"
            else:
                try:
                    result = TOOLS[name](**args)
                except Exception as exc:
                    # 兜底二：工具自己出错，不能让它炸穿主循环
                    result = f"工具执行出错：{exc}"

            trace.append({
                "round": round_no,
                "tool": name,
                "args": args,
                "result": result,
            })

            # ⑥ 把结果作为一条 tool 消息回填，下轮模型就能看到它
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    # 跑满 MAX_ROUNDS 还没收敛 → 主动刹车，把问题退回给用户
    return "任务步骤过多，已停止。请把问题拆成更小的部分再问。", trace


def save_trace(question: str, trace: list, path: Path = TRACE_PATH) -> None:
    """把一次运行的工具调用轨迹追加写入 jsonl 文件（一行一条记录）。

    注意：这个函数只负责"记录"，不参与决策——所以它放在 run_agent 外面，
    由调用方决定要不要存、存到哪。这样 run_agent 自身保持干净、可单独测试。
    """
    if not trace:
        return

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            # ensure_ascii=False：中文原样写入，而不是被转成 \u4e2d\u6587
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
