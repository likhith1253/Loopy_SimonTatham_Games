
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.graph import Graph
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver

class TestDynamicProgrammingSolver(unittest.TestCase):
    def setUp(self):
        # Default 3x3 game
        self.game = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        self.solver = self.game.cpu

    def test_instantiation(self):
        self.assertIsInstance(self.solver, DynamicProgrammingSolver)

    def test_3x3_hint_finding(self):
        # 3x3 with random clues (default init)
        # Should either return a hint OR explain multiple/no solutions
        hint = self.solver.generate_hint()
        self.assertIn("strategy", hint)
        self.assertEqual(hint["strategy"], "Dynamic Programming (State Compression)")
        
        if hint["move"]:
            self.assertIsInstance(hint["move"], tuple) # Edge tuple
            self.assertEqual(len(hint["move"]), 2)
        else:
            # If no move, explanation should be present
            self.assertTrue(hint["explanation"])

    def test_impossible_2x2_all_3s(self):
        # 2x2 board with all 3s is unsolvable (requires 2 loops or loose ends)
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game.clues = {
            (0,0): 3, (0,1): 3,
            (1,0): 3, (1,1): 3
        }
        hint = game.cpu.generate_hint()
        self.assertIsNone(hint["move"])
        self.assertIn("No valid solutions", hint["explanation"])

    def test_solve_method(self):
        move = self.solver.solve()
        if move:
            self.assertIsInstance(move, tuple)
            self.assertEqual(len(move), 2)
        else:
            # If unsolvable generated board, None is valid
            pass

    def test_decide_move(self):
        # Determine move and candidates
        candidates, best_move = self.solver.decide_move()
        
        self.assertIsInstance(candidates, list)
        if best_move:
            self.assertIsInstance(best_move, tuple)
            # Candidates should contain best_move
            self.assertTrue(any(c[0] == best_move for c in candidates))
            self.assertEqual(len(candidates), 1) # DP only returns best move
        else:
            self.assertIsNone(best_move)
            self.assertEqual(len(candidates), 0)

    def test_dp_constraint_injection_forced_and_forbidden(self):
        game = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        solution_edges = set(game.solution_edges)
        game.clues = self._build_full_clues(game.rows, game.cols, solution_edges)
        solver = DynamicProgrammingSolver(game)

        baseline = solver._run_dp(game, limit=1)
        self.assertIsInstance(baseline, list)

        edge = solver._get_all_potential_edges()[0]
        forced_solutions = solver._run_dp(game, limit=1, forced_edges={edge})
        self.assertIsInstance(forced_solutions, list)

        forbidden_solutions = solver._run_dp(game, limit=1, forbidden_edges={edge})
        self.assertIsInstance(forbidden_solutions, list)

        contradictory = solver._run_dp(game, limit=1, forced_edges={edge}, forbidden_edges={edge})
        self.assertEqual(contradictory, [])

    def test_dp_returns_none_when_no_forced_deduction_exists(self):
        game = GameState(rows=1, cols=1, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game.clues = {(0, 0): 2}
        solver = game.cpu
        move = solver.solve()
        self.assertIsNone(move)

    def test_dp_determinism_same_board_same_move(self):
        base = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        solution_edges = set(base.solution_edges)
        full_clues = self._build_full_clues(base.rows, base.cols, solution_edges)
        missing_edge = sorted(solution_edges)[0]

        game1 = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game1.clues = full_clues
        game1.graph = Graph(game1.rows, game1.cols)
        for edge in sorted(solution_edges):
            if edge == missing_edge:
                continue
            game1.graph.add_edge(edge[0], edge[1])
        game1.cpu = DynamicProgrammingSolver(game1)
        move1 = game1.cpu.solve()

        game2 = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game2.clues = full_clues
        game2.graph = Graph(game2.rows, game2.cols)
        for edge in sorted(solution_edges):
            if edge == missing_edge:
                continue
            game2.graph.add_edge(edge[0], edge[1])
        game2.cpu = DynamicProgrammingSolver(game2)
        move2 = game2.cpu.solve()

        self.assertEqual(move1, move2)

    def _build_full_clues(self, rows, cols, solution_edges):
        clues = {}
        for r in range(rows):
            for c in range(cols):
                count = 0
                if tuple(sorted(((r, c), (r, c + 1)))) in solution_edges:
                    count += 1
                if tuple(sorted(((r + 1, c), (r + 1, c + 1)))) in solution_edges:
                    count += 1
                if tuple(sorted(((r, c), (r + 1, c)))) in solution_edges:
                    count += 1
                if tuple(sorted(((r, c + 1), (r + 1, c + 1)))) in solution_edges:
                    count += 1
                clues[(r, c)] = count
        return clues

if __name__ == '__main__':
    unittest.main()
