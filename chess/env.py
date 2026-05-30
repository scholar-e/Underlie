from __future__ import annotations

import chess
import re
from stockfish import Stockfish
from typing import Any
from bench_common.env_sdk.base import BaseEnv, StepResult

class MyEnv(BaseEnv):
    def __init__(self, stockfish_path: str = ".\stockfish\stockfish-windows-x86-64-avx2.exe") -> None:
        self.board: chess.Board | None = None
        # Initialize Stockfish. Adjust depth based on how fast you want your benchmarks to run.
        self.engine = Stockfish(path=stockfish_path, depth=10)
        self.max_consecutive_illegal_moves = 5
        self.current_consecutive_illegal_moves = []

    def _get_observation(self) -> dict[str, Any]:
        if self.board is None:
            return {}
            
        # board.unicode() automatically uses distinct Unicode symbols 
        # for White (e.g. ♙, ♘) and Black (e.g. ♟, ♞) pieces.
        # invert_color=True matches standard dark terminal themes better.
        unicode_board_str = self.board.unicode(invert_color=False, empty_square='·')
        board_str = unicode_board_str.split('\n')
        
        visual_board = "\n   a b c d e f g h\n  +-----------------+\n"
        for i, row in enumerate(board_str):
            rank = 8 - i
            visual_board += f"{rank} | {row} | {rank}\n"
        visual_board += "  +-----------------+\n   a b c d e f g h\n"

        legal_moves = [self.board.san(move) for move in self.board.legal_moves]

        # 3. Generate the move history sequentially in standard SAN notation
        # We temporarily step through the game stack to format the historical moves cleanly
        temp_board = chess.Board()
        move_history = []
        for move in self.board.move_stack:
            move_history.append(temp_board.san(move))
            temp_board.push(move)

        return {
            "visual_board": visual_board,
            "legal_moves": legal_moves,
            "must_avoid_moves": self.current_consecutive_illegal_moves,
            "move_number": self.board.fullmove_number,
            "move_history": move_history,
            "turn": "White" if self.board.turn == chess.WHITE else "Black",
            "is_check": self.board.is_check()
        }

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self.board = chess.Board()
        self.current_consecutive_illegal_moves = []
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

        # Regex to pull out standard chess notation (e.g., Nf3, e4, O-O, Qh4+) 
        # out of conversational text
        move_match = re.search(r"[BRQNK][a-h][1-8]|[a-h][1-8]|[BRQNK][a-h][a-h][1-8]|O-O|0-0-0|[BRQNK]x[a-h][1-8]|[a-h]x[a-h][1-8]|1\/2-1\/2|1\/-O|O-\/1", raw_input)
        
        if move_match:
            move_input = move_match.group()
        else:
            move_input = raw_input  # Fallback if no pattern matches

        try:
            move = self.board.parse_san(move_input)
            
            if move in self.board.legal_moves:
                self.current_consecutive_illegal_moves = []

                # Track who is making the move BEFORE pushing it
                player_is_white = (self.board.turn == chess.WHITE)
                
                # 1. Get White-absolute score BEFORE the move
                white_eval_before = self._get_engine_evaluation(self.board)
                
                # 2. Apply the move to the real board
                self.board.push(move)
                info["legal_move"] = True
                
                # 3. Get White-absolute score AFTER the move
                white_eval_after = self._get_engine_evaluation(self.board)
                
                # 4. Calculate the delta from White's perspective
                white_cp_delta = white_eval_after - white_eval_before
                info["stockfish_cp_delta"] = white_cp_delta
                
                # 5. Assign reward using an epsilon threshold
                # Let's say a loss of less than 0.20 cp is perfectly acceptable (engine noise/book move)
                EPSILON_THRESHOLD = -0.20 

                if player_is_white:
                    # white_cp_delta is negative if White made things worse
                    if white_cp_delta >= EPSILON_THRESHOLD:
                        reward = 0.1  # Small positive reward for maintaining the position/good move
                    else:
                        reward = white_cp_delta / 100.0  # Negative reward for a real mistake
                else:
                    # Black wants white_cp_delta to be negative (White's score goes down)
                    black_cp_delta = -white_cp_delta
                    if black_cp_delta >= EPSILON_THRESHOLD:
                        reward = 0.1
                    else:
                        reward = black_cp_delta / 100.0
                
        except ValueError:
            self.current_consecutive_illegal_moves.append(move_input)
            info["error"] = f"Illegal move. Parsed move: {move_input} is not in the list of legal moves."
            reward = -5.0

        # Check final game conditions
        terminated = self.board.is_game_over()
        if terminated:
            result = self.board.result()
            info["game_result"] = result
            if result == "1-0":
                reward += 10.0 if player_is_white else -10.0
            elif result == "0-1":
                reward += 10.0 if not player_is_white else -10.0

        truncated = false
        elif self.current_consecutive_illegal_moves.__len__() >= self.max_consecutive_illegal_moves:
            terminated = true
            reward -= 20
            truncated = true
        
        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
