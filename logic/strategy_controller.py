"""
Strategy Controller
===================
Hierarchical orchestration layer for CPU move selection.

Rules:
- Keep solver implementations pure.
- Apply fallback only at controller level.
- Re-evaluate full priority order on every CPU turn.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.solvers.greedy_solver import GreedySolver

Move = Tuple[Tuple[int, int], Tuple[int, int]]


class StrategyController:
    def __init__(self, game_state: Any, selected_mode: str):
        self.game_state = game_state
        self.selected_mode = self._normalize_mode(selected_mode)

        # Reuse currently attached solver instance when compatible.
        cpu_solver = getattr(game_state, "cpu", None)
        self.dp_solver = cpu_solver if isinstance(cpu_solver, DynamicProgrammingSolver) else DynamicProgrammingSolver(game_state)
        self.advanced_dp_solver = cpu_solver if isinstance(cpu_solver, AdvancedDPSolver) else AdvancedDPSolver(game_state)
        self.dnc_solver = cpu_solver if isinstance(cpu_solver, DivideConquerSolver) else DivideConquerSolver(game_state)
        self.greedy_solver = cpu_solver if isinstance(cpu_solver, GreedySolver) else GreedySolver(game_state)

    def get_next_cpu_move(self) -> Tuple[Optional[Move], str]:
        if self.selected_mode == "dp":
            move = self._try_solver(self.dp_solver)
            if move is not None:
                return move, "DP"

            # Still within DP family: advanced DP decomposition before any non-DP fallback.
            move = self._try_solver(self.advanced_dp_solver)
            if move is not None:
                return move, "DP (Advanced)"

            move = self._try_solver(self.dnc_solver)
            if move is not None:
                return move, "D&C (Fallback)"

            move = self._try_solver(self.greedy_solver)
            if move is not None:
                return move, "Greedy (Final Fallback)"

            return None, "No moves available"

        if self.selected_mode == "dnc":
            move = self._try_solver(self.dnc_solver)
            if move is not None:
                return move, "D&C"

            move = self._try_solver(self.greedy_solver)
            if move is not None:
                return move, "Greedy (Fallback)"

            return None, "No moves available"

        move = self._try_solver(self.greedy_solver)
        if move is not None:
            return move, "Greedy"

        return None, "No moves available"

    def get_solver_for_source(self, source: str):
        if source.startswith("DP"):
            if source.startswith("DP (Advanced)"):
                return self.advanced_dp_solver
            return self.dp_solver
        if source.startswith("D&C"):
            return self.dnc_solver
        if source.startswith("Greedy"):
            return self.greedy_solver
        return None

    def get_fallback_message(self, source: str) -> Optional[str]:
        if self.selected_mode == "dp":
            if source == "D&C (Fallback)":
                return "No DP move available. Switched to D&C for this move."
            if source == "Greedy (Final Fallback)":
                return "No DP or D&C move available. Switched to Greedy for this move."

        if self.selected_mode == "dnc" and source == "Greedy (Fallback)":
            return "No D&C move available. Switched to Greedy for this move."

        return None

    def _try_solver(self, solver: Any) -> Optional[Move]:
        if solver is None:
            return None

        try:
            if hasattr(solver, "decide_move"):
                _, move = solver.decide_move()
                return move

            if hasattr(solver, "solve"):
                return solver.solve()
        except Exception:
            return None

        return None

    def _normalize_mode(self, mode: str) -> str:
        if mode in ("dynamic_programming", "dp"):
            return "dp"
        if mode in ("divide_conquer", "divide_and_conquer", "dnc"):
            return "dnc"
        return "greedy"
