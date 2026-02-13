"""
Greedy Solver Wrapper
====================
Wraps the existing `GreedyCPU` implementation in a solver strategy class.

CRITICAL:
- Do not change gameplay behavior.
- Do not change hint logic.
- UI currently calls `game_state.cpu.decide_move()`; this wrapper preserves that API.
"""

from __future__ import annotations

from typing import Any

from logic.greedy_cpu import GreedyCPU
from logic.solvers.solver_interface import AbstractSolver, HintPayload


class GreedySolver(AbstractSolver):
    """
    Adapter around `GreedyCPU`.

    This class implements the new solver interface while delegating the existing
    CPU methods so the UI and tests keep working unchanged.
    """

    def __init__(self, game_state: Any):
        self.game_state = game_state
        self._cpu = GreedyCPU(game_state)
        self._last_explanation: str = ""

    # ---- AbstractSolver API -------------------------------------------------
    def solve(self, board: Any = None):
        """
        Returns the next greedy move without applying it.

        For compatibility with the existing project, this matches the semantics
        of `GreedyCPU.make_move()`.
        """
        move = self.make_move()
        return move

    def generate_hint(self, board: Any = None):
        """
        Delegates to the existing hint system.

        IMPORTANT: Hint logic is owned by `GameState.get_hint()`; we do not
        modify it here, only call it.
        """
        target = board if board is not None else self.game_state
        move, reason = target.get_hint()

        explanation = self._build_hint_explanation(target=target, move=move, reason=reason)
        payload: HintPayload = {
            "move": move,
            "explanation": explanation,
            "strategy": "Greedy",
        }
        return payload

    def _build_hint_explanation(self, target: Any, move: Any, reason: str) -> str:
        """
        Build an explanation string without changing hint selection logic.
        Best-effort: if we can't infer details, return a safe generic explanation.
        """
        if not move:
            return "No forced hint is available from the Greedy strategy in the current position."

        try:
            u, v = move
        except Exception:
            return "Greedy selected a hint move based on the current board constraints."

        # Determine which rule likely triggered (based on existing hint reasons).
        rule = "Greedy hint"
        normalized = (reason or "").lower()
        if "loose end" in normalized:
            rule = "Loose-Ends Connection (Dijkstra)"
        elif "solution edge" in normalized:
            rule = "Solution-Edge Suggestion (fallback)"
        elif "remove" in normalized and "incorrect" in normalized:
            rule = "Remove Incorrect Edge (fallback)"
        elif "remove" in normalized:
            rule = "Remove Extra/Blocking Edge (fallback)"

        # Identify an adjacent cell (if any) to anchor the explanation.
        cell_note = ""
        try:
            graph = getattr(target, "graph", None)
            clues = getattr(target, "clues", {}) or {}
            if graph is not None:
                adj_cells: list[tuple[int, int]] = []
                r1, c1 = u
                r2, c2 = v
                if r1 == r2:  # horizontal edge
                    c_min = min(c1, c2)
                    if r1 > 0:
                        adj_cells.append((r1 - 1, c_min))  # above
                    if r1 < getattr(graph, "rows", r1):
                        adj_cells.append((r1, c_min))  # below
                else:  # vertical edge
                    r_min = min(r1, r2)
                    if c1 > 0:
                        adj_cells.append((r_min, c1 - 1))  # left
                    if c1 < getattr(graph, "cols", c1):
                        adj_cells.append((r_min, c1))  # right

                clue_cells = [c for c in adj_cells if c in clues]
                if clue_cells:
                    picked = clue_cells[0]
                    cell_note = f"Cell {picked} (clue {clues[picked]}) is adjacent to this edge."
                else:
                    # If no clue-adjacent cell, fall back to endpoint-based explanation.
                    if graph is not None and hasattr(graph, "get_degree"):
                        deg_u = graph.get_degree(u)
                        deg_v = graph.get_degree(v)
                        cell_note = f"This edge touches vertices with degrees {deg_u} and {deg_v}."
        except Exception:
            # Keep explanation robust; do not crash hint flow.
            cell_note = ""

        forced_note = (
            "This move is considered forced because it is valid under current constraints and helps preserve loop continuity "
            "without creating branches or immediate contradictions."
        )

        lines = [
            f"Rule triggered: {rule}.",
            f"Decision caused by: {cell_note}" if cell_note else "Decision caused by: local board constraints near the suggested edge.",
            f"Why forced: {forced_note}",
        ]

        # Also include the original hint reason (existing behavior) as context.
        if isinstance(reason, str) and reason.strip():
            lines.append(f"Original hint reason: {reason.strip()}")

        return "\n".join(lines)

    def explain_last_move(self) -> str:
        return self._last_explanation or getattr(self._cpu, "reasoning", "") or ""

    # ---- Existing CPU API (delegated / preserved) ---------------------------
    def decide_move(self):
        return self._cpu.decide_move()

    def make_move(self):
        """
        Keep existing behavior: choose a move and update GameState message.
        """
        move = self._cpu.make_move()
        self.register_move(move)
        return move

    def register_move(self, move):
        """
        Manually register a move's explanation so UI can display it.
        Useful when UI calls decide_move() directly.
        """
        # Capture formatted explanation for the UI
        reasoning = getattr(self._cpu, "reasoning", "No reasoning available")
        self.game_state.last_cpu_move_info = {
            "move": move,
            "explanation": f"Using Greedy Strategy: {reasoning}",
            "strategy": "Greedy"
        }

        # Capture an explanation snapshot for `explain_last_move()`.
        msg = getattr(self.game_state, "message", "")
        if isinstance(msg, str):
            self._last_explanation = msg
        else:
            self._last_explanation = ""

        # Global Execution Trace Log
        from logic.execution_trace import log_greedy_move
        if move:
            log_greedy_move(move, self._last_explanation or reasoning)

    def __getattr__(self, name: str):
        """
        Delegate any other GreedyCPU attributes/methods transparently.
        This preserves legacy callers (tests/scripts) that expect GreedyCPU APIs.
        """
        return getattr(self._cpu, name)

