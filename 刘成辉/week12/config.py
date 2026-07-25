"""
config.py — 配置加载
====================

职责:
    启动时从 .env 读出 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL,
    集中校验, 给 main.py / llm.py 用。

学到的概念:
    1. 12-factor: 配置和代码分离 (代码进 git, 配置只放环境)
    2. fail-fast: 启动时就该暴露配置错误, 不要拖到第一次调 LLM 才炸
    3. 单例模式: 一个进程只加载一次, 后面都从同一个对象取
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 在模块顶部就把 .env 加载进 os.environ
# override=False 避免覆盖 shell 里已经设的同名环境变量
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)


@dataclass(frozen=True)
class Config:
    """所有配置字段, frozen=True 防止运行中被意外改写."""
    api_key: str
    base_url: str
    model: str


def load_config() -> Config:
    """
    读取并校验配置, 返回一个 Config 实例.

    TODO (你来实现):
        1. 从 os.environ 读三个变量: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
        2. 如果 LLM_API_KEY 缺失或还是 .env.example 里的占位符, raise ValueError
           错误信息要明确告诉用户怎么修 (例: "请在 .env 里填 LLM_API_KEY")
        3. 如果 LLM_BASE_URL / LLM_MODEL 缺失, 给一个合理默认值而不是 raise
        4. 返回 Config(api_key=..., base_url=..., model=...)

    提示:
        - 校验占位符可以用: api_key.startswith("sk-your-")
        - 想想为什么 base_url 没填就给默认值, 但 api_key 必须 raise
    """
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")

    if api_key is None or api_key == "":
        raise RuntimeError("请在 .env 输入 LLM_API_KEY")
    if api_key.startswith("sk-your-"):
        raise RuntimeError("请在 .env 输入有效的 LLM_API_KEY")

    if base_url is None or base_url == "":
        base_url = "https://api.deepseek.com/v1"
        print("LLM_BASE_URL 未设置, 使用默认值:", base_url)

    if model is None or model == "":
        model = "deepseek-v4-flash"
        print("LLM_MODEL 未设置, 使用默认值:", model)

    return Config(api_key=api_key, base_url=base_url, model=model)



# 模块级单例: 别的文件 `from config import load_config` 即可
if __name__ == "__main__":
    # 快速冒烟测试: python config.py 应该打印出 Config(...)
    cfg = load_config()
    print(cfg)
