"""
谁是卧底游戏核心逻辑

使用 LangGraph 实现游戏流程控制。
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .players import AIPlayer
from .words import WORD_PAIRS


class GameState(TypedDict):
    """游戏状态"""

    players: list[AIPlayer]  # 所有玩家
    round_num: int  # 当前轮数
    descriptions: dict[str, list[str]]  # 玩家名 -> 描述列表
    votes: dict[str, str]  # 投票者 -> 被投票者
    eliminated: list[str]  # 被淘汰玩家名单
    winner: str  # 获胜方: "平民" 或 "卧底" 或 ""
    game_log: list[str]  # 游戏日志


class UndercoverGame:
    """谁是卧底游戏"""

    def __init__(self, num_players: int = 3) -> None:
        if num_players < 3:
            raise ValueError("至少需要 3 名玩家")
        self.num_players = num_players
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建游戏状态图"""
        graph = StateGraph(GameState)

        # 添加节点
        graph.add_node("describe", self._describe_phase)
        graph.add_node("vote", self._vote_phase)
        graph.add_node("eliminate", self._eliminate_phase)
        graph.add_node("check_winner", self._check_winner)

        # 添加边
        graph.add_edge(START, "describe")
        graph.add_edge("describe", "vote")
        graph.add_edge("vote", "eliminate")
        graph.add_conditional_edges(
            "eliminate",
            self._should_continue,
            {
                "continue": "describe",
                "end": "check_winner",
            },
        )
        graph.add_edge("check_winner", END)

        return graph

    def _describe_phase(self, state: GameState) -> dict:
        """描述阶段：每个存活玩家描述自己的词语"""
        round_num = state["round_num"] + 1
        descriptions = dict(state["descriptions"])
        game_log = list(state["game_log"])

        msg = f"\n=== 第 {round_num} 轮描述 ==="
        game_log.append(msg)
        print(msg, flush=True)

        alive_players = [p for p in state["players"] if p.is_alive]
        for player in alive_players:
            desc = player.describe(round_num, descriptions)
            if player.name not in descriptions:
                descriptions[player.name] = []
            descriptions[player.name].append(desc)
            log_entry = f"{player.name}: {desc}"
            game_log.append(log_entry)

        return {
            "round_num": round_num,
            "descriptions": descriptions,
            "game_log": game_log,
        }

    def _vote_phase(self, state: GameState) -> dict:
        """投票阶段：每个存活玩家投票"""
        votes: dict[str, str] = {}
        game_log = list(state["game_log"])
        descriptions = state["descriptions"]

        msg = f"\n=== 第 {state['round_num']} 轮投票 ==="
        game_log.append(msg)
        print(msg, flush=True)

        alive_players = [p for p in state["players"] if p.is_alive]
        for player in alive_players:
            # 只能投给其他存活玩家
            other_players = [p.name for p in alive_players if p.name != player.name]
            vote = player.vote(other_players, descriptions)
            votes[player.name] = vote
            log_entry = f"{player.name} 投票给 {vote}"
            game_log.append(log_entry)
            print(log_entry, flush=True)

        return {"votes": votes, "game_log": game_log}

    def _eliminate_phase(self, state: GameState) -> dict:
        """淘汰阶段：票数最高的玩家被淘汰"""
        votes = state["votes"]
        game_log = list(state["game_log"])
        eliminated = list(state["eliminated"])

        # 统计票数
        vote_counts = Counter(votes.values())
        max_votes = max(vote_counts.values())
        most_voted = [name for name, count in vote_counts.items() if count == max_votes]

        # 平票时随机选择一个
        eliminated_name = random.choice(most_voted)
        eliminated.append(eliminated_name)

        # 更新玩家状态
        for player in state["players"]:
            if player.name == eliminated_name:
                player.is_alive = False
                role = "卧底" if player.is_undercover else "平民"
                msg = f"\n{eliminated_name} 被淘汰！身份: {role}"
                game_log.append(msg)
                print(msg, flush=True)
                break

        return {"eliminated": eliminated, "game_log": game_log}

    def _should_continue(self, state: GameState) -> Literal["continue", "end"]:
        """判断游戏是否继续"""
        alive_players = [p for p in state["players"] if p.is_alive]
        undercover_alive = sum(1 for p in alive_players if p.is_undercover)
        civilian_alive = sum(1 for p in alive_players if not p.is_undercover)

        # 卧底被淘汰，平民胜利
        if undercover_alive == 0:
            return "end"

        # 卧底人数 >= 平民人数，卧底胜利
        if undercover_alive >= civilian_alive:
            return "end"

        return "continue"

    def _check_winner(self, state: GameState) -> dict:
        """判定胜负"""
        game_log = list(state["game_log"])
        alive_players = [p for p in state["players"] if p.is_alive]
        undercover_alive = sum(1 for p in alive_players if p.is_undercover)

        if undercover_alive == 0:
            winner = "平民"
            msg = "\n🎉 平民胜利！卧底被找出来了！"
        else:
            winner = "卧底"
            msg = "\n🎭 卧底胜利！成功隐藏到最后！"

        game_log.append(msg)
        print(msg, flush=True)

        # 揭示所有身份
        msg = "\n=== 身份揭晓 ==="
        game_log.append(msg)
        print(msg, flush=True)
        for player in state["players"]:
            role = "卧底" if player.is_undercover else "平民"
            log_entry = f"{player.name}: {role} (词语: {player.word})"
            game_log.append(log_entry)
            print(log_entry, flush=True)

        return {"winner": winner, "game_log": game_log}

    def create_players(self) -> list[AIPlayer]:
        """创建玩家并分配词语"""
        # 随机选择词语对
        civilian_word, undercover_word = random.choice(WORD_PAIRS)

        # 随机选择一个卧底
        undercover_idx = random.randint(0, self.num_players - 1)

        players = []
        for i in range(self.num_players):
            is_undercover = i == undercover_idx
            word = undercover_word if is_undercover else civilian_word
            player = AIPlayer(f"玩家{chr(65 + i)}", word, is_undercover)
            players.append(player)

        return players

    def run(self) -> GameState:
        """运行游戏"""
        players = self.create_players()

        initial_state: GameState = {
            "players": players,
            "round_num": 0,
            "descriptions": {},
            "votes": {},
            "eliminated": [],
            "winner": "",
            "game_log": ["🎮 谁是卧底游戏开始！"],
        }

        app = self.graph.compile()

        final_state = initial_state
        # 使用 stream 模式运行
        for output in app.stream(initial_state):
            # output 是一个字典，键是节点名称，值是该节点的输出
            for node_name, state_update in output.items():
                print(f"\n[节点完成: {node_name}]")
                # 更新状态
                final_state.update(state_update)

        return final_state
