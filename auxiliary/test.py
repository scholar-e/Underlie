#!/usr/bin/env python3
"""Test the chess environment directly (no HTTP adapter)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["STOCKFISH_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "stockfish", "src", "stockfish"
)
import json
from env import MyEnv

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def test_reset():
    print("\n=== Reset ===")
    env = MyEnv()
    obs = env.reset()
    check("returns dict", isinstance(obs, dict))
    check("has board", "board" in obs)
    check("has fen", "fen" in obs)
    check("has board_visual", "board_visual" in obs)
    check("has legal_moves", "legal_moves" in obs)
    check("has turn", "turn" in obs)
    check("turn is White", obs["turn"] == "White")
    check("WHITE TO MOVE in board", "WHITE TO MOVE" in obs["board"])
    check("fen correct", obs["fen"].startswith("rnbqkbnr/pppppppp/8"))
    env.close()
    return env


def test_legal_moves():
    print("\n=== Legal moves ===")
    env = MyEnv()
    env.reset()
    for move in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
        r = env.step(move)
        check(f"{move:8} legal", r.info.get("legal_move"))
    env.close()


def test_move_formats():
    print("\n=== Move format parsing ===")
    env = MyEnv()
    env.reset()
    formats = [
        ("MOVE: e4", True),
        ("  MOVE: e5  ", True),
        ("I think d4 is best\nMOVE: d4", True),
        ("MOVE: Nc6", True),
        ("MOVE: Nf3", True),
    ]
    for raw, expected in formats:
        r = env.step(raw)
        check(f"  {raw[:30]:30} legal={r.info.get('legal_move')}", r.info.get("legal_move") == expected)
    env.close()


def test_illegal_move():
    print("\n=== Illegal moves ===")
    env = MyEnv()
    env.reset()
    env.step("MOVE: e4")
    env.step("MOVE: e5")
    r = env.step("not_a_move")
    check("rejected", r.info.get("legal_move") == False)
    check("reward -5.0", r.reward == -5.0)
    check("not terminated", r.terminated == False)
    check("error has turn", "not a legal move for" in r.info.get("error", "").lower())
    check("rejected_moves present", "rejected_moves" in r.observation)
    check("has rejected entry", len(r.observation.get("rejected_moves", [])) > 0)
    env.close()


def test_wrong_turn_rejection():
    print("\n=== Wrong-turn rejection ===")
    env = MyEnv()
    env.reset()
    env.step("MOVE: e4")
    env.step("MOVE: e5")
    env.step("MOVE: Nf3")
    env.step("MOVE: Nc6")
    r = env.step("MOVE: d4")  # White's turn, d4 legal
    check("d4 legal for White", r.info.get("legal_move"))
    r = env.step("MOVE: d4")  # Black's turn, d4 illegal
    check("d4 illegal for Black", not r.info.get("legal_move"))
    check("banner shows WHITE MOVED d4", "WHITE MOVED d4" in r.observation.get("board", ""))
    check("banner shows BLACK TO MOVE", "BLACK TO MOVE" in r.observation.get("board", ""))
    check("rejected banner visible", "!!! Last move REJECTED" in r.observation.get("board", ""))
    env.close()


def test_illegal_limit():
    print("\n=== Consecutive illegal limit ===")
    env = MyEnv()
    env.reset()
    for i in range(4):
        r = env.step("bad")
        check(f"illegal {i+1} not terminated", not r.terminated and not r.truncated)
    r = env.step("bad")
    check("5th truncated", r.truncated)
    check("5th reward -10.0", r.reward == -10.0)
    env.close()


def test_scholars_mate():
    print("\n=== Scholar's Mate ===")
    env = MyEnv()
    env.reset()
    moves = ["MOVE: e4", "MOVE: e5", "MOVE: Qh5", "MOVE: Nc6",
             "MOVE: Bc4", "MOVE: Nf6", "MOVE: Qxf7"]
    for i, m in enumerate(moves):
        r = env.step(m)
        check(f"  step {i+1}: {m:12} legal", r.info.get("legal_move"))
    check("game over", r.terminated)
    check("result 1-0", r.info.get("game_result") == "1-0")
    env.close()


def test_step_numbering():
    print("\n=== Unique step numbering ===")
    env = MyEnv()
    env.reset()
    env.step("MOVE: e4")
    env.step("MOVE: e5")
    env.step("bad")
    env.step("bad")
    env.step("MOVE: Nf3")
    env.step("bad")
    steps = [d for d in os.listdir(env._trial_dir) if d.startswith("step_")]
    check(f"6 unique step dirs", len(steps) == 6)
    step_nums = sorted(int(s.split("_")[1]) for s in steps)
    check("step numbers 1-6", step_nums == [1, 2, 3, 4, 5, 6])
    env.close()


def test_board_visual():
    print("\n=== Board visual ===")
    env = MyEnv()
    obs = env.reset()
    bv = obs.get("board_visual", "")
    check("has black pieces", "♜" in bv or "♞" in bv)
    check("has white pieces", "♖" in bv or "♘" in bv)
    check("has borders", "|" in bv)
    check("has coordinates", "a b c d e f g h" in bv)
    env.close()


def test_reward():
    print("\n=== Reward ===")
    env = MyEnv()
    env.reset()
    r = env.step("MOVE: e4")
    check("opening e4 near 0", -0.01 < r.reward < 0.01)
    env.close()


def main():
    print("=" * 50)
    print("        CHESS ENV TEST SUITE")
    print("=" * 50)
    for fn in [test_reset, test_legal_moves, test_move_formats,
               test_illegal_move, test_wrong_turn_rejection,
               test_illegal_limit, test_scholars_mate,
               test_step_numbering, test_board_visual, test_reward]:
        fn()
    total = PASS + FAIL
    print(f"\n{'=' * 50}")
    print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL > 0:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
