#!/usr/bin/env python3
"""Test the Chess Benchmark environment directly (no HTTP adapter needed).

Usage:
    python3 test_env.py
"""

import sys
import traceback
from env import MyEnv

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)


def test_reset():
    print("\n=== Reset ===")
    env = MyEnv()
    obs = env.reset()

    check("observation is a dict", isinstance(obs, dict))
    check("board text present", "board" in obs)
    check("board_visual present", "board_visual" in obs)
    check("fen is starting position", obs.get("fen") == chess.STARTING_FEN)
    check("turn is White", obs.get("turn") == "White")
    check("move_number is 1", obs.get("move_number") == 1)
    check("legal_moves is non-empty", len(obs.get("legal_moves", [])) > 0)
    check("20 legal moves at start", len(obs.get("legal_moves", [])) == 20)
    check("is_check is False", obs.get("is_check") is False)
    check("move_history is empty", obs.get("move_history") == [])
    check("rejected_moves is empty", obs.get("rejected_moves") == [])

    env.close()
    return env


def test_legal_move():
    print("\n=== Legal move ===")
    env = MyEnv()
    env.reset()

    moves = env.board.legal_moves
    first_legal_san = env.board.san(list(moves)[0])

    result = env.step(first_legal_san)

    check("terminated is False", result.terminated is False)
    check("truncated is False", result.truncated is False)
    check("info.legal_move is True", result.info.get("legal_move") is True)
    check("reward is a number", isinstance(result.reward, (int, float)))
    check("turn flipped to Black", result.observation.get("turn") == "Black")
    check("move_history has 1 entry", len(result.observation.get("move_history", [])) == 1)
    check("error not in info", "error" not in result.info)

    env.close()


def test_illegal_move():
    print("\n=== Illegal move ===")
    env = MyEnv()
    env.reset()

    result = env.step("Qe99")

    check("terminated is False", result.terminated is False)
    check("truncated is False", result.truncated is False)
    check("info.legal_move is False", result.info.get("legal_move") is False)
    check("reward is -5.0", result.reward == -5.0)
    check("error message present", "error" in result.info)

    env.close()


def test_consecutive_illegal_terminates():
    print("\n=== Consecutive illegal moves (max 5) ===")
    env = MyEnv()
    env.reset()

    for i in range(4):
        result = env.step("Qe99")
        check(f"illegal #{i+1} not terminated", result.terminated is False, f"at step {i+1}")

    result = env.step("Qe99")
    check("5th illegal terminates", result.truncated is True, "should be truncated after 5 illegals")

    env.close()


def test_uci_format_parsing():
    print("\n=== UCI format move parsing ===")
    env = MyEnv()
    env.reset()

    result = env.step("e2e4")
    check("UCI e2e4 accepted", result.info.get("legal_move") is True, f"got error: {result.info.get('error', 'none')}")

    env.close()


def test_move_with_prefix():
    print("\n=== MOVE: prefix parsing ===")
    env = MyEnv()
    env.reset()

    result = env.step("MOVE: e4")
    check("MOVE: prefix accepted", result.info.get("legal_move") is True, f"got error: {result.info.get('error', 'none')}")

    env.close()


def test_game_over_checkmate():
    print("\n=== Fool's mate (game over) ===")
    env = MyEnv()
    env.reset()

    moves = ["f3", "e5", "g4", "Qh4"]
    for i, m in enumerate(moves):
        result = env.step(m)
        if i < len(moves) - 1:
            check(f"move {m} not terminated", result.terminated is False)

    check("fool's mate terminates", result.terminated is True, f"game result: {result.info.get('game_result', '?')}")
    check("game result is 0-1", result.info.get("game_result") == "0-1")

    env.close()


def test_stockfish_evaluation():
    print("\n=== Stockfish evaluation present ===")
    env = MyEnv()
    env.reset()

    result = env.step("e4")
    check("stockfish_cp_delta in info", "stockfish_cp_delta" in result.info)
    check("delta is a number", isinstance(result.info["stockfish_cp_delta"], (int, float)))

    env.close()


def test_reward_for_good_move():
    print("\n=== Reward for good move ===")
    env = MyEnv()
    env.reset()

    result = env.step("e4")
    check("e4 reward is a number", isinstance(result.reward, (int, float)))

    env.close()


def test_must_avoid_moves():
    print("\n=== must_avoid_moves tracking ===")
    env = MyEnv()
    env.reset()

    result = env.step("Qe99")
    check("must_avoid_moves populated", len(result.observation.get("must_avoid_moves", [])) > 0)
    check("rejected_moves populated", len(result.observation.get("rejected_moves", [])) > 0)
    check("rejected move recorded", result.observation["rejected_moves"][0]["attempt"] == "Qe99")

    env.close()


def test_observation_after_legal_move():
    print("\n=== Observation after legal move ===")
    env = MyEnv()
    env.reset()

    result = env.step("e4")
    obs = result.observation

    check("must_avoid_moves is empty after legal", obs.get("must_avoid_moves") == [])
    check("fen changed from start", obs.get("fen") != chess.STARTING_FEN)
    check("board text contains history", "History:" in obs.get("board", ""))
    check("legal_moves still populated", len(obs.get("legal_moves", [])) > 0)

    env.close()


def test_multiple_turns():
    print("\n=== Multiple turn flow ===")
    env = MyEnv()
    env.reset()

    results = []
    for move in ["e4", "e5", "Nf3", "Nc6"]:
        result = env.step(move)
        results.append(result)
        check(f"move {move} accepted", result.info.get("legal_move") is True)

    final_obs = results[-1].observation
    check("turn is White (after 4 half-moves)", final_obs.get("turn") == "White")
    check("move_number is 3", final_obs.get("move_number") == 3)
    check("move_history has 4 entries", len(final_obs.get("move_history", [])) == 4)

    env.close()


def test_castling_detection():
    print("\n=== Castling move regex ===")
    env = MyEnv()
    env.reset()

    moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"]
    last_result = None
    for m in moves:
        last_result = env.step(m)

    check("castling king-side accepted", last_result.info.get("legal_move") is True)

    env.close()


def test_board_visual_in_obs():
    print("\n=== Board visual observation ===")
    env = MyEnv()
    env.reset()

    obs = env.reset()
    check("board_visual is a string", isinstance(obs.get("board_visual"), str))
    check("board_visual contains ranks", "8" in obs["board_visual"])
    check("board_visual contains files", "a" in obs["board_visual"])

    env.close()


def test_reset_clears_state():
    print("\n=== Reset clears state ===")
    env = MyEnv()
    env.reset()
    env.step("e4")
    env.step("e5")

    obs2 = env.reset()
    check("fen is starting after reset", obs2.get("fen") == chess.STARTING_FEN)
    check("move_history empty after reset", obs2.get("move_history") == [])

    env.close()


def test_draw_by_repetition():
    print("\n=== Draw by repetition ===")
    env = MyEnv()
    env.reset()

    for _ in range(3):
        env.step("Nf3")
        env.step("Nf6")
        env.step("Ng1")
        env.step("Ng8")

    result = env.step("Nf3")
    check("threefold repetition detected", result.terminated is True)

    env.close()


def test_illegal_en_passant_not_crash():
    print("\n=== Illegal en passant (no crash) ===")
    env = MyEnv()
    env.reset()

    result = env.step("e4")
    check("first move ok", result.info.get("legal_move") is True)

    result = env.step("d5")
    check("second move ok", result.info.get("legal_move") is True)

    result = env.step("exd6")
    check("en passant rejection is graceful", not result.info.get("legal_move", True))
    check("no crash", "error" in result.info)

    env.close()


HTTPS_DOC = '''
import urllib.request
import json
url = "http://localhost:8765"
req = urllib.request.Request(f"{url}/health")
with urllib.request.urlopen(req) as resp:
    print(resp.read().decode())
'''

HTTP_DOC = '''
import httpx
resp = httpx.get("http://localhost:8765/health")
print(resp.json())
'''


def test_http_adapter_health():
    """Test the HTTP adapter health endpoint if the adapter is running."""
    print("\n=== HTTP Adapter health check (optional) ===")
    try:
        import httpx
        resp = httpx.get("http://localhost:8765/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            check("adapter healthy", data.get("status") == "ok", f"got: {data}")
        else:
            check("adapter reachable", False, f"status {resp.status_code}")
    except ImportError:
        print("  [SKIP] httpx not available")
    except Exception as e:
        print(f"  [SKIP] adapter not running ({e})")
        print("         Start with: python3 adapter.py")


def main():
    global PASS, FAIL

    tests = [
        test_reset,
        test_legal_move,
        test_illegal_move,
        test_consecutive_illegal_terminates,
        test_uci_format_parsing,
        test_move_with_prefix,
        test_game_over_checkmate,
        test_stockfish_evaluation,
        test_reward_for_good_move,
        test_must_avoid_moves,
        test_observation_after_legal_move,
        test_multiple_turns,
        test_castling_detection,
        test_board_visual_in_obs,
        test_reset_clears_state,
        test_draw_by_repetition,
        test_illegal_en_passant_not_crash,
        test_http_adapter_health,
    ]

    print(f"Running {len(tests)} test suites...")

    for test in tests:
        try:
            test()
        except Exception as e:
            FAIL += 1
            print(f"  [FAIL] {test.__name__} threw exception: {e}")
            traceback.print_exc()

    total = PASS + FAIL
    pc = PASS / total * 100 if total else 0
    print(f"\n{'=' * 40}")
    print(f"Results: {PASS}/{total} passed ({pc:.0f}%), {FAIL}/{total} failed")
    if FAIL > 0:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
