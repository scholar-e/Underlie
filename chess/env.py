from __future__ import annotations

import chess
from stockfish import Stockfish
from typing import Any
from bench_common.env_sdk.base import BaseEnv, StepResult

class ChessEnv(BaseEnv):
    def __init__(self, stockfish_path: str = "path/to/your/stockfish/executable") -> None:
        self.board: chess.Board | None = None
        # Initialize Stockfish. Adjust depth based on how fast you want your benchmarks to run.
        self.engine = Stockfish(path=stockfish_path, depth=10)

    def _get_observation(self) -> dict[str, Any]:
        if self.board is None:
            return {}
            
        board_str = str(self.board).split('\n')
        visual_board = "\n   a b c d e f g h\n  +-----------------+\n"
        for i, row in enumerate(board_str):
            rank = 8 - i
            formatted_row = row.replace('.', '·')
            visual_board += f"{rank} | {formatted_row} | {rank}\n"
        visual_board += "  +-----------------+\n   a b c d e f g h\n"

        legal_moves = [self.board.san(move) for move in self.board.legal_moves]

        return {
            "visual_board": visual_board,
            "legal_moves": legal_moves,
            "turn": "White" if self.board.turn == chess.WHITE else "Black",
            "is_check": self.board.is_check()
        }

    def _get_engine_evaluation(self, perspective_is_white: bool) -> float:
        """
        Gets the current board evaluation from Stockfish.
        Returns a normalized score where positive is good for the active perspective.
        """
        self.engine.set_fen_position(self.board.fen())
        eval_data = self.engine.get_evaluation()
        
        # If it's a forced checkmate sequence
        if eval_data["type"] == "mate":
            mate_moves = eval_data["value"]
            # Assign a massive score for mate, decaying slightly if it takes more moves
            base_mate_score = 10000.0 if mate_moves > 0 else -10000.0
            raw_score = base_mate_score / (abs(mate_moves) + 1)
        else:
            # Centipawn value (100 cp = 1 pawn advantage)
            raw_score = float(eval_data["value"])
            
        # Stockfish gives evaluation relative to White. 
        # Flip it if we want the evaluation from Black's perspective.
        return raw_score if perspective_is_white else -raw_score

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self.board = chess.Board()
        return self._get_observation()

    def step(self, action: Any) -> StepResult:
        if self.board is None:
            raise RuntimeError("Call reset() before step()")
        
        move_input = str(action).strip()
        info = {"requested_move": move_input, "legal_move": False}
        reward = 0.0
        
        # 1. Get the engine evaluation BEFORE the move is made
        # (Relative to the player whose turn it currently is)
        current_turn_is_white = (self.board.turn == chess.WHITE)
        eval_before = self._get_engine_evaluation(perspective_is_white=current_turn_is_white)

        try:
            move = self.board.parse_san(move_input)
            
            if move in self.board.legal_moves:
                # 2. Execute the move
                self.board.push(move)
                info["legal_move"] = True
                
                # 3. Get the evaluation AFTER the move
                # (Still relative to the player who just moved to see if they improved or hurt their position)
                eval_after = self._get_engine_evaluation(perspective_is_white=current_turn_is_white)
                
                # 4. Calculate reward based on Centipawn Shift
                # Scaling by 100 turns a 1-pawn blunder into a -1.0 reward
                reward = (eval_after - eval_before) / 100.0
                info["stockfish_cp_delta"] = eval_after - eval_before
                
            else:
                info["error"] = "Illegal move"
                reward = -5.0  # Heavy penalty for picking an illegal move so the AI learns rules
                
        except ValueError:
            info["error"] = "Invalid notation syntax"
            reward = -5.0  # Heavy penalty for complete syntax gibberish

        # Check final game conditions
        terminated = self.board.is_game_over()
        if terminated:
            result = self.board.result()
            info["game_result"] = result
            if result == "1-0":
                reward += 10.0 if current_turn_is_white else -10.0
            elif result == "0-1":
                reward += 10.0 if not current_turn_is_white else -10.0

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info=info,
        )