"""
谁是卧底 AI 游戏入口

运行: uv run python undercover_game/main.py
"""

from dotenv import load_dotenv

from undercover_game.game import UndercoverGame


def main() -> None:
    load_dotenv()

    print("=" * 50)
    print("   🎭 谁是卧底 AI 游戏 🎭")
    print("=" * 50)
    print("\n3 个 AI 玩家，1 个卧底，2 个平民")
    print("卧底需要隐藏身份，平民需要找出卧底!\n")

    game = UndercoverGame(num_players=3)

    # 打印初始状态
    print("🎮 谁是卧底游戏开始！", flush=True)

    result = game.run()

    print("\n" + "=" * 50)
    print(f"   游戏结束！获胜方: {result['winner']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
