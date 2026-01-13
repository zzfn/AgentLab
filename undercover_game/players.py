"""
AI 玩家模块

定义 AI 玩家的行为：描述词语和投票。
"""

from __future__ import annotations

import os
import re
import time

from langchain_openai import ChatOpenAI


# 全局共享的 LLM 实例
_global_llm = None


def get_llm() -> ChatOpenAI:
    """获取 LLM 实例 (单例模式)"""
    global _global_llm
    if _global_llm is not None:
        return _global_llm

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    if not api_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY")

    _global_llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model="glm-4-flash",
        streaming=True,
        timeout=30,
    )
    return _global_llm


class AIPlayer:
    """AI 玩家类"""

    def __init__(self, name: str, word: str, is_undercover: bool) -> None:
        self.name = name
        self.word = word
        self.is_undercover = is_undercover
        self.is_alive = True
        self.descriptions: list[str] = []
        self.llm = get_llm()

    def describe(self, round_num: int, all_descriptions: dict[str, list[str]]) -> str:
        """
        生成对自己词语的描述。

        Args:
            round_num: 当前轮数
            all_descriptions: 所有玩家之前的描述

        Returns:
            本轮描述
        """
        # 构建历史描述上下文
        history = ""
        if all_descriptions:
            history = "\n之前各玩家的描述:\n"
            for player_name, descs in all_descriptions.items():
                if descs:
                    history += f"  {player_name}: {'; '.join(descs)}\n"

        prompt = f"""你正在玩"谁是卧底"游戏。
你的词语是: {self.word}

游戏规则:
- 你需要用一句话描述你的词语，但不能直接说出这个词
- 描述要足够准确让同伴认出你，但又不能太明显让卧底发现
- 如果你是卧底，你需要让描述足够模糊，不被发现
{history}
这是第 {round_num} 轮。请给出你的描述（一句话，10-20字左右）:"""

        print(f"{self.name}: ", end="", flush=True)
        start_time = time.time()
        first_token_time = None
        full_response = ""

        for chunk in self.llm.stream(prompt):
            if first_token_time is None and chunk.content:
                first_token_time = time.time()

            content = chunk.content
            print(content, end="", flush=True)
            full_response += content

        end_time = time.time()

        ttft = (first_token_time - start_time) if first_token_time else 0
        gen_time = (
            (end_time - first_token_time)
            if first_token_time
            else (end_time - start_time)
        )
        print(
            f" (TTFT: {ttft:.2f}s, 生成: {gen_time:.2f}s, 总计: {end_time - start_time:.2f}s)"
        )

        description = full_response.strip()
        self.descriptions.append(description)
        return description

    def vote(
        self, alive_players: list[str], all_descriptions: dict[str, list[str]]
    ) -> str:
        """
        投票选出最可疑的玩家。

        Args:
            alive_players: 存活玩家列表（不包含自己）
            all_descriptions: 所有玩家的描述

        Returns:
            被投票玩家的名字
        """
        # 构建描述信息
        desc_info = ""
        for player_name in alive_players:
            if player_name in all_descriptions:
                descs = all_descriptions[player_name]
                desc_info += f"  {player_name}: {'; '.join(descs)}\n"

        prompt = f"""你正在玩"谁是卧底"游戏。
你的词语是: {self.word}
你的名字是: {self.name}

各玩家的描述:
{desc_info}
请分析谁最可疑，选择一个玩家投票淘汰。
可选玩家: {", ".join(alive_players)}

请只回复一个玩家的名字（如: 玩家A）:"""

        print(f"  > {self.name} 正在投票: ", end="", flush=True)
        start_time = time.time()
        first_token_time = None
        full_response = ""

        for chunk in self.llm.stream(prompt):
            if first_token_time is None and chunk.content:
                first_token_time = time.time()

            content = chunk.content
            print(content, end="", flush=True)
            full_response += content

        end_time = time.time()

        ttft = (first_token_time - start_time) if first_token_time else 0
        gen_time = (
            (end_time - first_token_time)
            if first_token_time
            else (end_time - start_time)
        )
        print(
            f" [完成] (TTFT: {ttft:.2f}s, 生成: {gen_time:.2f}s, 总计: {end_time - start_time:.2f}s)",
            flush=True,
        )
        vote = full_response.strip()

        # 尝试从回复中提取玩家名
        for player in alive_players:
            if player in vote:
                return player

        # 如果无法匹配，检查是否包含 "玩家" + 字母/数字
        match = re.search(r"玩家[A-Z0-9]", vote)
        if match and match.group() in alive_players:
            return match.group()

        # 默认返回第一个可选玩家
        return alive_players[0] if alive_players else ""

    def __repr__(self) -> str:
        status = "存活" if self.is_alive else "淘汰"
        role = "卧底" if self.is_undercover else "平民"
        return f"AIPlayer({self.name}, {role}, {status})"
