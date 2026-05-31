from __future__ import annotations

import chess
import json
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
from stockfish import Stockfish
from typing import Any
from bench_common.env_sdk.base import BaseEnv, StepResult

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
    MAX_CONSECUTIVE_ILLEGAL = 5
    MOVE_PATTERN = re.compile(
        r"O-O-O|0-0-0|"
        r"O-O|0-0|"
        r"[KQRBN][a-h1-8]??x?[a-h][1-8](?:=[QRBN])?[+#]?|"
        r"[a-h](?:x[a-h])?[1-8](?:=[QRBN])?[+#]?|"
        r"1-0|0-1|1/2-1/2"
    )

    def __init__(self, stockfish_path: str | None = None) -> None:
        self.board: chess.Board | None = None
        self._stockfish_path = stockfish_path or os.environ.get("STOCKFISH_PATH") or None
        self.engine: Stockfish | None = None
        self.current_consecutive_illegal_moves: list[str] = []
        self._all_rejected_moves: list[dict[str, Any]] = []
        self._trial_dir: str | None = None
        self._current_step: int = 0
        self._best_reward: float = 0.0
        self._game_result: str = ""

    def _init_engine(self) -> None:
        if self.engine is not None:
            return
        sf_path = self._stockfish_path
        if not sf_path:
            _chess_dir = os.path.dirname(os.path.abspath(__file__))
            _candidates = [
                "stockfish",
                os.path.join(_chess_dir, "stockfish", "src", "stockfish"),
            ]
            if sys.platform == "win32":
                _candidates.append(
                    os.path.join(_chess_dir, "stockfish", "stockfish-windows-x86-64-avx2.exe")
                )
            for c in _candidates:
                if c == "stockfish":
                    try:
                        subprocess.run(["stockfish", "--version"], capture_output=True, timeout=5)
                        sf_path = "stockfish"
                        break
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                elif os.path.isfile(c) and os.access(c, os.X_OK):
                    sf_path = c
                    break
        if not sf_path:
            _chess_dir = os.path.dirname(os.path.abspath(__file__))
            _src_dir = os.path.join(_chess_dir, "stockfish", "src")
            _makefile = os.path.join(_src_dir, "Makefile")
            if os.path.isfile(_makefile):
                print("Compiling Stockfish from source...", file=sys.stderr)
                try:
                    subprocess.run(
                        ["make", "-j", str(os.cpu_count() or 2), "build"],
                        cwd=_src_dir, check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    )
                    _binary = os.path.join(_src_dir, "stockfish")
                    if os.path.isfile(_binary):
                        sf_path = _binary
                        print(f"Stockfish compiled: {sf_path}", file=sys.stderr)
                except subprocess.CalledProcessError as e:
                    err = e.stderr.decode(errors="replace")[:200] if e.stderr else str(e)
                    print(f"Stockfish compilation failed: {err}", file=sys.stderr)
        if not sf_path:
            sf_path = self._download_stockfish()
        if not sf_path:
            print("Stockfish not available — running without engine evaluation", file=sys.stderr)
            self.engine = None
            return
        try:
            self.engine = Stockfish(path=sf_path, depth=10)
            self._stockfish_path = sf_path
        except Exception as e:
            print(f"Stockfish init failed: {e} — running without engine evaluation", file=sys.stderr)
            self.engine = None

    @staticmethod
    def _download_stockfish() -> str | None:
        _chess_dir = os.path.dirname(os.path.abspath(__file__))
        _dest_dir = os.path.join(_chess_dir, "stockfish", "src")
        os.makedirs(_dest_dir, exist_ok=True)
        _binary = os.path.join(_dest_dir, "stockfish")

        if os.path.isfile(_binary) and os.access(_binary, os.X_OK):
            return _binary

        import platform
        arch = platform.machine()
        if arch not in ("x86_64", "amd64"):
            return None

        # Try GitHub API first to get the latest release
        variants = ["bmi2", "avx2", "sse41-popcnt"]
        download_urls: list[str] = []
        try:
            api = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
            req = urllib.request.Request(api, headers={"User-Agent": "python", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                import json
                release = json.loads(r.read())
                tag = release["tag_name"]
                for asset in release.get("assets", []):
                    name = asset["name"]
                    for v in variants:
                        if f"ubuntu-x86-64-{v}" in name or name == "stockfish-ubuntu-x86-64.tar":
                            download_urls.append(asset["browser_download_url"])
                            break
        except Exception:
            pass

        # Fallback static URLs if API failed
        if not download_urls:
            for v in variants:
                download_urls.append(
                    f"https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64-{v}.tar"
                )
            download_urls.append(
                "https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar"
            )

        import tempfile, shutil
        print("Downloading Stockfish...", file=sys.stderr)
        for url in download_urls:
            short = url.rsplit("/", 1)[-1]
            try:
                print(f"  trying {short}...", file=sys.stderr)
                req = urllib.request.Request(url, headers={"User-Agent": "python"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    with tempfile.TemporaryDirectory() as tmp:
                        tarpath = os.path.join(tmp, "sf.tar")
                        with open(tarpath, "wb") as f:
                            f.write(resp.read())
                        with tarfile.open(tarpath, "r") as tar:
                            tar.extractall(path=tmp)
                        # Find the binary (large executable file with stockfish in name)
                        found = None
                        for root, _dirs, files in os.walk(tmp):
                            for f in files:
                                fp = os.path.join(root, f)
                                if "stockfish" in f and os.path.isfile(fp) and os.path.getsize(fp) > 1_000_000:
                                    found = fp
                                    break
                            if found:
                                break
                        if found:
                            shutil.copy2(found, _binary)
                            os.chmod(_binary, 0o755)
                            print(f"Stockfish downloaded: {_binary}", file=sys.stderr)
                            return _binary
            except Exception as e:
                print(f"  {short} failed: {e}", file=sys.stderr)
                continue

        return None

    @staticmethod
    def _pieces_summary(board: chess.Board, color: chess.Color) -> str:
        pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                sq = chess.square_name(square)
                p = chess.piece_symbol(piece.piece_type)
                pieces.append(f"{p.upper() if color == chess.WHITE else p}{sq}")
        return ", ".join(pieces) if pieces else "none"

    def _get_observation(self) -> dict[str, Any]:
        if self.board is None:
            return {}

        turn_str = "White" if self.board.turn == chess.WHITE else "Black"
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]

        temp_board = chess.Board()
        move_history = []
        for move in self.board.move_stack:
            move_history.append(temp_board.san(move))
            temp_board.push(move)

        last_san = move_history[-1] if move_history else None
        if last_san is None:
            lines = ["=== WHITE TO MOVE ==="]
        else:
            prev_color = "BLACK" if self.board.turn == chess.WHITE else "WHITE"
            lines = [
                f"=== {prev_color} MOVED {last_san} ===",
                f">>> {turn_str.upper()} TO MOVE <<<",
            ]
        move_num = self.board.fullmove_number
        lines.append(f"Move: {move_num}")
        if self.board.is_check():
            lines.append("Check: YES")
        else:
            lines.append("Check: no")
        lines.append("")

        if self._all_rejected_moves:
            last = self._all_rejected_moves[-1]
            lines.append(f"!!! Last move REJECTED: {last['attempt']} - {last['reason']} !!!")
            lines.append("")

        lines.append(f"White: {self._pieces_summary(self.board, chess.WHITE)}")
        lines.append(f"Black: {self._pieces_summary(self.board, chess.BLACK)}")

        if self.board.move_stack:
            temp_board2 = chess.Board()
            history_parts = []
            for i, move in enumerate(self.board.move_stack):
                san = temp_board2.san(move)
                temp_board2.push(move)
                if i % 2 == 0:
                    history_parts.append(f"{i//2 + 1}. {san}")
                else:
                    history_parts[-1] += f" {san}"
            lines.append(f"History: {' '.join(history_parts)}")

        lines.append(f"Legal moves: {', '.join(legal_moves)}")

        if self._all_rejected_moves:
            recent = self._all_rejected_moves[-3:]
            lines.append("Previously rejected (do not repeat):")
            for r in recent:
                lines.append(f"  - {r['attempt']} ({r['reason']})")

        text_board_string = "\n".join(lines)

        visual_lines = []
        board_str = self.board.unicode(invert_color=False, borders=True)
        for line in board_str.splitlines():
            visual_lines.append(line)
        board_visual = "\n".join(visual_lines)

        return {
            "board": text_board_string,
            "board_visual": board_visual,
            "fen": self.board.fen(),
            "legal_moves": legal_moves,
            "must_avoid_moves": self.current_consecutive_illegal_moves,
            "rejected_moves": self._all_rejected_moves[-5:] if self._all_rejected_moves else [],
            "move_number": self.board.fullmove_number,
            "move_history": move_history,
            "turn": turn_str,
            "is_check": self.board.is_check()
        }

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

        fen = observation.get("fen") if isinstance(observation, dict) else None
        data = {
            "step": step,
            "move": move,
            "reward": reward,
            "best_reward": best_reward,
            "legal_move": legal_move,
            "observation": observation,
            "fen": fen,
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

    def _write_diagnostics(self, reason: str, raw_input: str, parse_attempts: dict) -> None:
        if not self._trial_dir:
            return
        diag = {
            "termination_reason": reason,
            "last_raw_input": raw_input,
            "parse_attempts": parse_attempts,
            "board_fen": self.board.fen() if self.board else None,
            "legal_moves": [self.board.san(m) for m in self.board.legal_moves] if self.board else [],
            "consecutive_illegal": list(self.current_consecutive_illegal_moves),
            "step": self._current_step,
            "move_number": self.board.fullmove_number if self.board else None,
            "turn": "White" if self.board and self.board.turn == chess.WHITE else "Black",
        }
        diag_path = os.path.join(self._trial_dir, "diagnostics.json")
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2)

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._init_engine()
        self.board = chess.Board()
        self.current_consecutive_illegal_moves = []
        self._all_rejected_moves = []
        self._trial_dir = self._create_trial_dir()
        self._current_step = 0
        self._best_reward = 0.0
        self._game_result = ""
        return self._get_observation()

    def _get_engine_evaluation(self, current_board: chess.Board) -> float:
        self.engine.set_fen_position(current_board.fen())
        eval_data = self.engine.get_evaluation()

        if eval_data["type"] == "mate":
            mate_moves = eval_data["value"]
            base_mate_score = 10000.0 if mate_moves > 0 else -10000.0
            raw_score = base_mate_score / (abs(mate_moves) + 1)
        else:
            raw_score = float(eval_data["value"])

        if current_board.turn == chess.BLACK:
            return -raw_score
        return raw_score

    def step(self, action: Any) -> StepResult:
        if self.board is None:
            raise RuntimeError("Call reset() before step()")

        self._current_step += 1
        step_num = self._current_step

        raw_input = str(action).strip()
        info: dict[str, Any] = {"requested_move": raw_input}
        reward = 0.0
        player_is_white = False

        move = None
        move_input = raw_input
        parse_diag = {"move_prefix": None, "parse_san_raw": None, "regex_match": None}

        move_prefix = re.search(r"(?:^|\n)\s*MOVE:\s*(\S+)", raw_input, re.IGNORECASE)
        if move_prefix:
            parse_diag["move_prefix"] = move_prefix.group(1)
            try:
                move = self.board.parse_san(move_prefix.group(1))
                move_input = move_prefix.group(1)
            except ValueError:
                pass

        if move is None:
            parse_diag["parse_san_raw"] = raw_input[:100]
            try:
                move = self.board.parse_san(raw_input)
                move_input = raw_input
            except ValueError:
                pass

        if move is None:
            move_match = self.MOVE_PATTERN.search(raw_input)
            if move_match:
                parse_diag["regex_match"] = move_match.group()
                candidate = move_match.group()
                try:
                    move = self.board.parse_san(candidate)
                    move_input = candidate
                except ValueError:
                    pass

        if move is None or move not in self.board.legal_moves:
            self.current_consecutive_illegal_moves.append(move_input)
            turn_str = "White" if self.board.turn == chess.WHITE else "Black"
            err = (
                f"Illegal move. '{move_input}' is not a legal move for {turn_str}. "
                f"Choose one from legal_moves and avoid must_avoid_moves."
            )
            self._all_rejected_moves.append({
                "attempt": move_input[:120],
                "reason": f"Not legal for {turn_str}",
                "turn": turn_str,
                "step": step_num,
            })
            info.update({"legal_move": False, "error": err})

            illegal_count = len(self.current_consecutive_illegal_moves)
            self._save_step(
                step_num, move_input, -5.0,
                self._best_reward, False,
                self._get_observation(), {**info, "illegal_count": illegal_count},
            )

            if illegal_count >= self.MAX_CONSECUTIVE_ILLEGAL:
                reason = (
                    f"Terminated after {illegal_count} consecutive illegal/failed moves. "
                    f"Last input: {raw_input!r}"
                )
                info["termination_reason"] = reason
                info["illegal_count"] = illegal_count
                self._save_step(
                    step_num, move_input, -10.0,
                    self._best_reward, False,
                    self._get_observation(), {**info, "illegal_count": illegal_count},
                )
                self._write_diagnostics(reason, raw_input, parse_diag)
                self._game_result = reason
                if self.board.is_game_over():
                    self._write_results()
                return StepResult(
                    observation=self._get_observation(),
                    reward=-10.0,
                    terminated=self.board.is_game_over(),
                    truncated=True,
                    info=info,
                )

            return StepResult(
                observation=self._get_observation(),
                reward=-5.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        self.current_consecutive_illegal_moves = []
        info["legal_move"] = True
        player_is_white = (self.board.turn == chess.WHITE)

        white_eval_before = self._get_engine_evaluation(self.board)
        self.board.push(move)
        white_eval_after = self._get_engine_evaluation(self.board)
        white_cp_delta = white_eval_after - white_eval_before
        info["stockfish_cp_delta"] = white_cp_delta

        DEAD_ZONE = 50
        if player_is_white:
            delta = white_cp_delta
        else:
            delta = -white_cp_delta

        if abs(delta) <= DEAD_ZONE:
            reward = 0.0
        elif delta > DEAD_ZONE:
            reward = 0.1
        else:
            reward = delta / 100.0

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

        self._best_reward = max(self._best_reward, reward)

        self._save_step(
            step_num, move_input, reward,
            self._best_reward, True,
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
            if self.current_consecutive_illegal_moves:
                self._write_diagnostics(
                    f"Episode closed with {len(self.current_consecutive_illegal_moves)} unresolved illegal moves",
                    self.current_consecutive_illegal_moves[-1] if self.current_consecutive_illegal_moves else "",
                    {},
                )
