from __future__ import annotations

import chess
import json
import os
import re
from stockfish import Stockfish
from typing import Any
from bench_common.env_sdk.base import BaseEnv, StepResult

# --- Trial persistence ---

AUX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auxiliary")
TRIALS_DIR = os.path.join(AUX_DIR, "trials")
os.makedirs(TRIALS_DIR, exist_ok=True)

_TEST_NUMBER: int | None = None


def _get_test_number() -> int:
    global _TEST_NUMBER
    if _TEST_NUMBER is None:
        existing = []
        if os.path.isdir(TRIALS_DIR):
            for d in os.listdir(TRIALS_DIR):
                parts = d.split("_")
                if len(parts) == 2 and parts[0] == "test" and parts[1].isdigit():
                    existing.append(int(parts[1]))
        _TEST_NUMBER = max(existing) + 1 if existing else 1
    return _TEST_NUMBER

class MyEnv(BaseEnv):
    def __init__(self, stockfish_path: str = ".\stockfish\stockfish-windows-x86-64-avx2.exe") -> None:
        self.board: chess.Board | None = None
        self.engine = Stockfish(path=stockfish_path, depth=10)
        self.max_consecutive_illegal_moves = 5
        self.current_consecutive_illegal_moves = []
        self._trial_dir: str | None = None
        self._current_step: int = 0
        self._best_reward: float = 0.0
        self._game_result: str = ""

    def _get_observation(self) -> dict[str, Any]:
        if self.board is None:
            return {}
            
        # --- 1. Keep all your existing setup / dictionary code ---
        # (If you had something like info setups or tracking, leave it here)

        # --- 2. Generate the text-based board description ---
        white_pieces = []
        black_pieces = []

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                square_name = chess.square_name(square)
                piece_name = chess.piece_name(piece.piece_type)
                description = f"{piece_name} on {square_name}"
            
                if piece.color == chess.WHITE:
                    white_pieces.append(description)
                else:
                    black_pieces.append(description)

        observation_parts = []
        if white_pieces:
            observation_parts.append(f"White has a {', '.join(white_pieces)}.")
        else:
            observation_parts.append("White has no pieces left.")
        
        if black_pieces:
            observation_parts.append(f"Black has a {', '.join(black_pieces)}.")
        else:
            observation_parts.append("Black has no pieces left.")

        # This is our new natural language string
        text_board_string = " ".join(observation_parts)

        legal_moves = [self.board.san(move) for move in self.board.legal_moves]

        # 3. Generate the move history sequentially in standard SAN notation
        # We temporarily step through the game stack to format the historical moves cleanly
        temp_board = chess.Board()
        move_history = []
        for move in self.board.move_stack:
            move_history.append(temp_board.san(move))
            temp_board.push(move)

        return {
            "board": text_board_string,
            "legal_moves": legal_moves,
            "must_avoid_moves": self.current_consecutive_illegal_moves,
            "move_number": self.board.fullmove_number,
            "move_history": move_history,
            "turn": "White" if self.board.turn == chess.WHITE else "Black",
            "is_check": self.board.is_check()
        }

    # --- Trial persistence ---

    @staticmethod
    def _next_trial_in_test(test_dir: str) -> int:
        existing = []
        if os.path.isdir(test_dir):
            for d in os.listdir(test_dir):
                parts = d.split("_")
                if len(parts) == 2 and parts[0] == "trial" and parts[1].isdigit():
                    existing.append(int(parts[1]))
        return max(existing) + 1 if existing else 1

    def _create_trial_dir(self) -> str:
        test_num = _get_test_number()
        test_dir = os.path.join(TRIALS_DIR, f"test_{test_num}")
        os.makedirs(test_dir, exist_ok=True)
        trial_num = self._next_trial_in_test(test_dir)
        name = f"trial_{trial_num}"
        trial_dir = os.path.join(test_dir, name)
        os.makedirs(trial_dir, exist_ok=True)
        return trial_dir

    def _save_step(self, step: int, move: str, reward: float, best_reward: float,
                   legal_move: bool, observation: dict[str, Any], info: dict[str, Any]) -> None:
        if not self._trial_dir:
            return
        step_dir = os.path.join(self._trial_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)

        data = {
            "step": step,
            "move": move,
            "reward": reward,
            "best_reward": best_reward,
            "legal_move": legal_move,
            "observation": observation,
            "info": {str(k): str(v) for k, v in info.items()},
        }
        with open(os.path.join(step_dir, "step_data.json"), "w") as f:
            json.dump(data, f, indent=2)

    def _write_results(self) -> None:
        if not self._trial_dir:
            return
        results_path = os.path.join(self._trial_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_reward": self._best_reward,
                "total_steps": self._current_step,
                "game_result": self._game_result,
                "num_moves": self._current_step,
            }, f, indent=2)

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self.board = chess.Board()
        self.current_consecutive_illegal_moves = []
        self._trial_dir = self._create_trial_dir()
        self._current_step = 0
        self._best_reward = 0.0
        self._game_result = ""
        return self._get_observation()

    def _get_engine_evaluation(self, current_board: chess.Board) -> float:
        """
        Always returns the evaluation from WHITE's perspective, 
        regardless of whose turn it is.
        """
        self.engine.set_fen_position(current_board.fen())
        eval_data = self.engine.get_evaluation()
        
        if eval_data["type"] == "mate":
            mate_moves = eval_data["value"]
            base_mate_score = 10000.0 if mate_moves > 0 else -10000.0
            raw_score = base_mate_score / (abs(mate_moves) + 1)
        else:
            raw_score = float(eval_data["value"])
            
        # FIX: Stockfish returns values from the perspective of the player whose turn it is.
        # If it's White's turn, Stockfish's perspective is already White's perspective.
        # If it's Black's turn, Stockfish's positive score means Black is better, 
        # so we must invert it to get White's perspective.
        if current_board.turn == chess.BLACK:
            return -raw_score
        return raw_score

    def step(self, action: Any) -> StepResult:
        if self.board is None:
            raise RuntimeError("Call reset() before step()")

        raw_input = str(action).strip()
        info = {"requested_move": raw_input, "legal_move": False}
        reward = 0.0
        player_is_white = False
        legal_move_san = ""

        move_match = re.search(r"[BRQNK][a-h][1-8]|[a-h][1-8]|[BRQNK][a-h][a-h][1-8]|O-O|0-0-0|[BRQNK]x[a-h][1-8]|[a-h]x[a-h][1-8]|1\/2-1\/2|1\/-O|O-\/1", raw_input)

        if move_match:
            move_input = move_match.group()
        else:
            move_input = raw_input

        try:
            move = self.board.parse_san(move_input)

            if move in self.board.legal_moves:
                self.current_consecutive_illegal_moves = []

                player_is_white = (self.board.turn == chess.WHITE)

                white_eval_before = self._get_engine_evaluation(self.board)

                self.board.push(move)
                info["legal_move"] = True
                legal_move_san = move_input

                white_eval_after = self._get_engine_evaluation(self.board)

                white_cp_delta = white_eval_after - white_eval_before
                info["stockfish_cp_delta"] = white_cp_delta

                EPSILON_THRESHOLD = -0.20

                if player_is_white:
                    if white_cp_delta >= EPSILON_THRESHOLD:
                        reward = 0.1
                    else:
                        reward = white_cp_delta / 100.0
                else:
                    black_cp_delta = -white_cp_delta
                    if black_cp_delta >= EPSILON_THRESHOLD:
                        reward = 0.1
                    else:
                        reward = black_cp_delta / 100.0

        except ValueError:
            self.current_consecutive_illegal_moves.append(move_input)
            info["error"] = f"Illegal move. Parsed move: {move_input} is not in the list of legal moves."
            reward = -5.0

        terminated = self.board.is_game_over()
        truncated = False

        if terminated:
            result = self.board.result()
            info["game_result"] = result
            self._game_result = result
            if result == "1-0":
                reward += 10.0 if player_is_white else -10.0
            elif result == "0-1":
                reward += 10.0 if not player_is_white else -10.0

        elif self.current_consecutive_illegal_moves.__len__() >= self.max_consecutive_illegal_moves:
            terminated = True
            reward -= 20
            truncated = True
            self._game_result = "illegal-loss"

        self._current_step += 1
        self._best_reward = max(self._best_reward, reward)

        self._save_step(
            self._current_step, legal_move_san, reward,
            self._best_reward, info["legal_move"],
            self._get_observation(), info,
        )
        if terminated:
            self._write_results()

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        if self._trial_dir and self._current_step > 0:
            self._write_results()
