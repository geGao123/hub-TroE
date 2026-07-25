"""
context.py — 上下文管理
========================

职责:
    防止 messages 列表无限增长, 在调 LLM 前做长度控制。

学到的概念:
    1. token 不是字符: 中文字符 1 个 ≈ 1.5 token, 英文单词 1 个 ≈ 1.3 token
       精确算要 tiktoken, 教学里粗估够用
    2. 截断 vs 压缩: 工程上最简单的就是丢老消息, 真要保留语义才上 summarization
    3. 上下文窗口: 主流 LLM 4k~128k token 不等, 你不主动管, 模型替你随机丢
"""
from __future__ import annotations

# 最多保留多少条消息 (不算 system)
# 调大: 上下文长, 单轮贵, 但 LLM 记得多
# 调小: 便宜, 但容易"失忆"
MAX_MESSAGES = 20


def estimate_tokens(messages: list[dict]) -> int:
    """
    粗估 messages 总 token 数.

    TODO (你来实现):
        思路: 遍历每条消息, 累加 content 长度
        - 中文为主的文本: len(content) / 1.5
        - 想更简单: len(content) // 2 也行, 数量级对就行
        - 不要算 role / metadata, 只算 content
        - 返回 int

    提示:
        - messages 里 message.get("content") 可能是 None (比如 tool call),
          这种情况 content 不计入, 或者按 0 处理
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if content is not None:
            total += len(content) // 2  # 粗估
    return total

def trim_messages(messages: list[dict], max_messages: int = MAX_MESSAGES) -> list[dict]:
    """
    把 messages 截到 max_messages 条以内.

    策略:
        - 如果第一条是 system, 永远保留 (它是人设, 截掉 agent 就变傻子)
        - 然后只保留最后 max_messages 条 user/assistant/tool 消息
        - 老的 user/assistant 中间直接丢 (这一版不做 summarization)

    TODO (你来实现):
        1. 判断 messages[0] 是不是 system
        2. 拆成 system_part + history_part
        3. history_part 保留最后 max_messages 条
        4. 返回 system_part + history_part
        5. 注意 list 不要原地改 (slice 返回新 list, 避免污染原 list)

    提示:
        - 考虑如果 messages 本来就 <= max_messages, 直接 return 原 list
        - 边界: messages 为空 / 只有 system / 没有 system
    """
    if not messages:
        return []

    if len(messages) <= max_messages:
        return messages

    if messages[0]["role"] == "system":
        system_part = [messages[0]]
        history_part = messages[1:]
    else:
        system_part = []
        history_part = messages

    trimmed_history = history_part[-max_messages:]
    return system_part + trimmed_history


if __name__ == "__main__":
    # 冒烟测试: 构造 25 条消息, 看 trim 之后剩多少
    fake = [{"role": "system", "content": "你是助手"}]
    for i in range(24):
        fake.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"})
    print(f"before: {len(fake)} msgs, ~{estimate_tokens(fake)} tokens")
    trimmed = trim_messages(fake)
    print(f"after:  {len(trimmed)} msgs, ~{estimate_tokens(trimmed)} tokens")
    print(f"system preserved: {trimmed[0]['role'] == 'system'}")
