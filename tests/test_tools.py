# ============================================================
# test_tools.py —— tools.py 的单元测试：工具行为、安全防护、注册表一致性
# ============================================================
from datetime import datetime

import pytest

from tools import (
    TOOLS,
    TOOL_SCHEMAS,
    add_knowledge,
    calculate,
    clear_knowledge,
    get_current_time,
    knowledge_info,
    search_knowledge,
)


# ---------------- calculate：正常计算 ----------------

def test_calculate_basic_arithmetic():
    assert calculate("(1234+5678)*3") == "20736"


def test_calculate_returns_string():
    assert isinstance(calculate("1+1"), str)


def test_calculate_supports_decimals():
    assert calculate("0.5*4") == "2.0"


# ---------------- calculate：安全防护 ----------------

@pytest.mark.parametrize(
    "payload",
    [
        '__import__("os").system("calc")',
        "open('secret.txt','w')",
        "1+1; os.system('ls')",
        "print(1)",
        "__builtins__",
        "eval('1+1')",
    ],
)
def test_calculate_rejects_non_whitelisted_input(payload):
    """白名单之外的字符一律拒绝——这几条都是典型的注入载荷。"""
    assert calculate(payload) == "只支持数字和 + - * / ( ) 运算。"


def test_calculate_survives_syntax_error():
    result = calculate("(1+2")
    assert result.startswith("计算失败")


def test_calculate_survives_division_by_zero():
    result = calculate("1/0")
    assert result.startswith("计算失败")


# ---------------- get_current_time ----------------

def test_get_current_time_matches_expected_format():
    value = get_current_time()
    # 格式不对会抛 ValueError，等价于断言
    datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


# ---------------- 两张表的一致性 ----------------

def test_tool_names_are_consistent_between_tables():
    schema_names = {item["function"]["name"] for item in TOOL_SCHEMAS}
    assert schema_names == set(TOOLS)


def test_every_schema_has_description():
    for item in TOOL_SCHEMAS:
        assert item["function"]["description"].strip()


# ---------------- 服务端知识库 ----------------

def test_knowledge_store_add_and_clear():
    clear_knowledge()
    assert knowledge_info() == {"sources": [], "chunks": 0}

    total = add_knowledge(["片段一", "片段二"], [[0.1], [0.2]], source="demo.md")
    assert total == 2
    assert knowledge_info() == {"sources": ["demo.md"], "chunks": 2}

    clear_knowledge()
    assert knowledge_info() == {"sources": [], "chunks": 0}


def test_search_knowledge_reports_empty_store():
    """知识库为空时应直接给出提示，不加载检索模型。"""
    clear_knowledge()
    assert search_knowledge("任意问题") == "知识库中还没有内容。"
