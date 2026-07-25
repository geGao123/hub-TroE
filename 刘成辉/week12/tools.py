"""
tools.py — 工具调用 (Step 4: 装饰器注册表版)
=============================================

职责:
    用 @tool 装饰器自动注册工具, 加新工具只写一个函数 + 一行装饰器.

学到的概念:
    1. 装饰器: 接函数返函数, 可以在不动函数本体的情况下"包"一层逻辑
    2. 注册表模式: 全局 dict 收集所有工具, 遍历统一暴露
    3. schema 参数化: 从手工 dict 改成调用方传 description + parameters
    4. 这一版还是手写 parameters, 进阶版可以从 Python type hints 自动推
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Callable


# === 工具注册表 ===

# name -> {"func": <callable>, "schema": <dict>}
# 装饰器把每个工具塞这里, 外部靠 all_tool_schemas() / dispatch_tool() 访问
_REGISTRY: dict[str, dict] = {}


def tool(
    name: str | None = None,
    description: str = "",
    parameters: dict | None = None,
):
    """
    装饰器: 把函数注册为 LLM 可调用的工具.

    用法:
        @tool(description="返回当前时间, 格式: YYYY-MM-DD HH:MM:SS")
        def get_current_time() -> str:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        @tool(
            description="获取指定城市天气",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"]
            }
        )
        def get_weather(city: str) -> str:
            return f"{city} 晴"

    参数:
        name:        工具名, 不传就用 fn.__name__
        description: 工具描述, LLM 靠这个判断要不要调
        parameters:  JSON Schema 描述参数, 不传就是无参工具
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        _description = description or (fn.__doc__ or "").strip()
        _parameters = parameters or {"type": "object", "properties": {}, "required": []}
        schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": _description,
                "parameters": _parameters
            }
        }
        _REGISTRY[tool_name] = {"func": fn, "schema": schema}
        return fn
    return decorator


def all_tool_schemas() -> list[dict]:
    """返回所有工具的 schema 列表, 直接传给 LLM."""
    return [r["schema"] for r in _REGISTRY.values()]


def all_tool_names() -> list[str]:
    """返回所有已注册工具名, 调试用."""
    return list(_REGISTRY.keys())


def dispatch_tool(name: str, arguments: str) -> str:
    """
    根据工具名执行对应函数, 返回结果字符串.

    出错时 (未知工具 / 参数解析失败 / 函数执行失败) 返回错误信息字符串,
    这样 LLM 看到错误能决定下一步 (比如改参数重试, 或者告知用户).
    """
    entry = _REGISTRY.get(name)
    if not entry:
        raise ValueError(f"未知工具: {name}")

    fn = entry["func"]

    # arguments 可能是 "{}" 也可能是 "" (取决于 provider), 都按空 dict 处理
    try:
        args_dict = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return f"参数解析失败: {e}"

    try:
        return fn(**args_dict)
    except Exception as e:
        return f"工具执行失败: {e}"


# === 工具定义: 只写函数 + 一行装饰器 ===

@tool(description="返回服务器当前的本地时间, 格式: YYYY-MM-DD HH:MM:SS")
def get_current_time() -> str:
    """返回当前时间字符串."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(
    description="返回指定城市的当前天气信息",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"]
    }
)
def get_weather(city: str) -> str:
    """返回指定城市的模拟天气信息."""
    weather_conditions = ["晴朗", "多云", "阴天", "下雨"]
    condition = random.choice(weather_conditions)
    temperature = random.randint(15, 30)
    return f"{city} 当前天气: {condition}, 温度: {temperature}°C"

@tool(
    description="返回两个整数的和",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "第一个整数"},
            "b": {"type": "integer", "description": "第二个整数"}
        },
        "required": ["a", "b"]
    })
def add(a: int, b: int) -> str:
    """返回两个整数的和."""
    return str(a + b)

if __name__ == "__main__":
    print(f"已注册 {len(_REGISTRY)} 个工具: {all_tool_names()}")
    print(dispatch_tool("get_current_time", "{}"))
    print(dispatch_tool("get_weather", '{"city": "杭州"}'))
    print(dispatch_tool("add", '{"a": 5, "b": 3}'))
