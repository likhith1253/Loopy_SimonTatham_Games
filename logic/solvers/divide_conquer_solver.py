"""
Divide & Conquer Solver
=======================

Implements a recursive Divide & Conquer strategy for Slitherlink.
Falls back to Greedy strategy if D&C cannot determine a move.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, List, Set, Dict

from logic.solvers.solver_interface import AbstractSolver, HintPayload
from logic.solvers.greedy_solver import GreedySolver
from logic.validators import is_valid_move, check_clue_constraint

# Type Alias for a Move: (u, v) where u, v are (row, col) coordinates
Move = Tuple[Tuple[int, int], Tuple[int, int]]


class DivideConquerSolver(AbstractSolver):
    """
    Solves Slitherlink using Divide & Conquer.

    Algorithm:
    1.  **Divide**: Recursively split the board into 4 quadrants until base case size.
    2.  **Conquer**: Solve base cases (small subgrids <= 3x3) using local constraints.
    3.  **Merge**: Resolve boundary edges between quadrants.
        - Check for forced connections to satisfy cell counts.
        - Avoid premature loops.
    4.  **Fallback**: If no move found, delegate to GreedySolver.
    """

    def __init__(self, game_state: Any):
        self.game_state = game_state
        # Fallback solver
        self.greedy_solver = GreedySolver(game_state)
        self._last_explanation: str = ""
        
        # Threshold for base case (e.g., 3x3 subgrid)
        self.BASE_CASE_SIZE = 3

    def decide_move(self) -> Tuple[List[Any], Optional[Move]]:
        """
        Implicit interface required by UI (main_window.py calls decide_move).
        Returns (candidates, best_move).
        """
        # We don't really have 'candidates' in the same way GreedyCPU does (which visualizes thinking).
        # We just return the final move.
        move = self.solve()
        return [], move


    def solve(self, board: Any = None):
        """
        Attempt to find a move using Divide & Conquer.
        If D&C finds nothing, return Greedy's move.
        """
        # 1. Try D&C
        move, reason = self._run_divide_and_conquer()
        
        if move:
            # Prepare explanation for UI
            self._last_explanation = f"Divide & Conquer: {reason}"
            return move
            
        # 2. Fallback
        self._last_explanation = "Divide & Conquer found no deterministic moves. Falling back to Greedy."
        fallback_move = self.greedy_solver.solve(board)
        
        # Append Greedy's explanation if available
        if fallback_move:
             # The fallback GreedySolver likely wrote to game_state.last_cpu_move_info automatically
             # via its own register_move. We should intercept/append to it.
             greedy_reason = self.greedy_solver.explain_last_move()
             self._last_explanation += f"\n(Fallback triggered)\nGreedy Reason: {greedy_reason}"
             
             # Overwrite/Augment the GameState info to reflect it was D&C Fallback
             self.game_state.last_cpu_move_info["strategy"] = "Divide & Conquer (Fallback)"
             self.game_state.last_cpu_move_info["explanation"] = self._last_explanation
             
        return fallback_move

    def register_move(self, move: Move):
        """
        Called by UI/Game Loop to finalize the move explanation.
        """
        # If we just performed a fallback, `last_cpu_move_info` is already set by GreedySolver
        # and updated by solve().
        # If we performed a D&C move, we need to set it here.
        
        # Check if the current info matches our move.
        current_info = getattr(self.game_state, "last_cpu_move_info", None)
        
        # If it's a D&C move (or we need to force update)
        if not current_info or current_info.get("move") != move or "Fallback" not in current_info.get("strategy", ""):
             self.game_state.last_cpu_move_info = {
                 "move": move,
                 "explanation": self._last_explanation,
                 "strategy": "Divide & Conquer"
             }

    def generate_hint(self, board: Any = None) -> HintPayload:
        """
        Generate a hint using D&C, or fallback to Greedy.
        """
        # Check if it's human turn (only provide hints for human player)
        is_human_turn = (self.game_state.game_mode in ["vs_cpu", "expert"] and 
                        self.game_state.turn == "Player 1 (Human)") or \
                       (self.game_state.game_mode not in ["vs_cpu", "expert"])
        
        if not is_human_turn:
            return {
                "move": None,
                "strategy": "Divide & Conquer",
                "explanation": "Hints are only available during your turn."
            }
        
        # 1. Try D&C
        move, reason = self._run_divide_and_conquer()
        
        if move:
            # Check if this move conflicts with CPU strategy
            # Don't give away the exact move CPU would make
            cpu_move, _ = self._run_divide_and_conquer()
            if cpu_move and move == cpu_move:
                # If D&C suggests the same move CPU would make, use fallback
                fallback_hint = self.greedy_solver.generate_hint(board)
                if fallback_hint.get("move"):
                    original_expl = fallback_hint.get("explanation", "")
                    fallback_hint["explanation"] = f"Strategy: Divide & Conquer (Alternative)\n(D&C found the same move as CPU, suggesting alternative)\n{original_expl}"
                    fallback_hint["strategy"] = "Divide & Conquer (Alternative)"
                    return fallback_hint
            
            return {
                "move": move,
                "explanation": f"Strategy: Divide & Conquer\nReason: {reason}",
                "strategy": "Divide & Conquer"
            }
             
        # 2. Fallback
        fallback_hint = self.greedy_solver.generate_hint(board)
        # Augment the explanation
        original_expl = fallback_hint.get("explanation", "")
        fallback_hint["explanation"] = f"Strategy: Fallback (Greedy)\n(D&C found no obvious moves)\n{original_expl}"
        fallback_hint["strategy"] = "Divide & Conquer (Fallback)"
        
        return fallback_hint

    def explain_last_move(self) -> str:
        return self._last_explanation

    # ---- Core Divide & Conquer Logic ----------------------------------------

    def _run_divide_and_conquer(self) -> Tuple[Optional[Move], str]:
        """
        Entry point for the recursive algorithm.
        Returns (move, reason) or (None, "").
        """
        rows = self.game_state.rows
        cols = self.game_state.cols
        
        # Region: (r_min, c_min, r_max, c_max) inclusive
        full_region = (0, 0, rows - 1, cols - 1)
        
        return self._solve_region(full_region)

    def _solve_region(self, region: Tuple[int, int, int, int]) -> Tuple[Optional[Move], str]:
        """
        Recursive function to solve a subgrid.
        """
        r_min, c_min, r_max, c_max = region
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        
        # Base Case
        if height <= self.BASE_CASE_SIZE and width <= self.BASE_CASE_SIZE:
            return self._solve_base_case(region)
            
        # Recursive Step: Split into Quadrants
        mid_r = r_min + height // 2
        mid_c = c_min + width // 2
        
        # Define 4 quadrants (handling odd sizes roughly)
        # TL: (r_min, c_min) -> (mid_r-1, mid_c-1)
        # TR: (r_min, mid_c) -> (mid_r-1, c_max)
        # BL: (mid_r, c_min) -> (r_max, mid_c-1)
        # BR: (mid_r, mid_c) -> (r_max, c_max)
        
        quadrants = [
            (r_min, c_min, mid_r - 1, mid_c - 1),  # TL
            (r_min, mid_c, mid_r - 1, c_max),      # TR
            (mid_r, c_min, r_max, mid_c - 1),      # BL
            (mid_r, mid_c, r_max, c_max)           # BR
        ]
        
        # 1. Try to solve each quadrant recursively
        for quad in quadrants:
            # Validate quadrant existence (might be empty if split line is near edge)
            q_r_min, q_c_min, q_r_max, q_c_max = quad
            if q_r_max >= q_r_min and q_c_max >= q_c_min:
                move, reason = self._solve_region(quad)
                if move:
                    return move, f"Subgrid {quad}: {reason}"

        # 2. Merge Phase: Check boundaries between quadrants
        # We check the "cross" cut: horizontal line at mid_r, vertical line at mid_c
        
        # Horizontal boundary check (between row mid_r-1 and mid_r)
        if mid_r > r_min and mid_r <= r_max:
             move, reason = self._check_boundary_line(
                 axis="horizontal", 
                 fixed_idx=mid_r, 
                 start=c_min, 
                 end=c_max
             )
             if move: return move, f"Merge Phase (Horizontal): {reason}"
             
        # Vertical boundary check (between col mid_c-1 and mid_c)
        if mid_c > c_min and mid_c <= c_max:
             move, reason = self._check_boundary_line(
                 axis="vertical", 
                 fixed_idx=mid_c, 
                 start=r_min, 
                 end=r_max
             )
             if move: return move, f"Merge Phase (Vertical): {reason}"

        return None, ""

    def _solve_base_case(self, region: Tuple[int, int, int, int]) -> Tuple[Optional[Move], str]:
        """
        Solve a small grid using exhaustive local logic (0, 3, corners).
        """
        r_min, c_min, r_max, c_max = region
        
        # Iterate over all cells in this base region
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                cell = (r, c)
                if cell in self.game_state.clues:
                     val = self.game_state.clues[cell]
                     
                     # Check classic Slitherlink heuristics for this cell
                     move, reason = self._apply_cell_heuristics(cell, val)
                     if move:
                         return move, f"Base Case ({r},{c}): {reason}"
                         
        return None, ""

    def _apply_cell_heuristics(self, cell, val) -> Tuple[Optional[Move], str]:
        """
        Apply simple deterministic rules for a single cell.
        """
        r, c = cell
        graph = self.game_state.graph
        
        # Get edges around this cell
        # Top, Bottom, Left, Right
        edges = [
            ((r, c), (r, c+1)),     # Top
            ((r+1, c), (r+1, c+1)), # Bottom
            ((r, c), (r+1, c)),     # Left
            ((r, c+1), (r+1, c+1))  # Right
        ]
        
        # Standardize edge format
        edges = [tuple(sorted(e)) for e in edges]
        
        # Count status
        existing = [e for e in edges if e in graph.edges]
        # "Blocked" means we cannot place an edge there (e.g. 'x' markers in UI, 
        # but here we rely on 'is_valid_move' or implicit knowledge).
        # We can detect "effectively blocked" if adding it would violate a neighbor clue.
        
        # Rule 1: Satisfied Cell
        if len(existing) == val:
            # All other edges around this cell must be REMOVED (if they exist)
            # Actually, in this game engine, we don't 'mark as cross'.
            # We just don't add them.
            # But if there's an heuristic that says "This edge CANNOT be here", 
            # we typically don't express that as a 'move' unless it's removing an existing one.
            pass

        # Rule 2: Remaining Edges Forced
        # If (val - existing) == (available_empty_edges), then all available must be added.
        
        potential_positive_moves = []
        for e in edges:
            if e not in graph.edges:
                # Check if it's possible to add this edge
                u, v = e
                # A robust check would look at neighbor clues too.
                valid, _ = is_valid_move(graph, u, v, self.game_state.clues, check_global=False)
                if valid:
                    potential_positive_moves.append(e)
                    
        needed = val - len(existing)
        if needed > 0 and len(potential_positive_moves) == needed:
            # FORCE ADD
            return potential_positive_moves[0], f"Cell {cell} needs {needed} more edges and only {len(potential_positive_moves)} are available."
            
        # Rule 3: 0 Clue
        if val == 0:
            for e in edges:
                 if e in graph.edges:
                     return e, f"Cell {cell} has clue 0 but has an edge."
                     
            # Also, 'x' out neighbors? 
            # We don't have 'x' moves. But we can ignore.

        # ---------------------------------------------------------------------
        # ADVANCED HEURISTICS
        # ---------------------------------------------------------------------

        # Rule 4: Corner 3
        # If a 3 is in a literal board corner, the two edges touching the corner vertex ARE FORCED.
        corner_move = self._check_corner_3(cell, val, edges)
        if corner_move:
             return corner_move

        # Rule 5: Adjacent 3s (Orthogonal)
        # If this cell is a 3, is there a neighbor 3?
        adj_3_move = self._check_adjacent_3s(cell, val)
        if adj_3_move:
             return adj_3_move
             
        # Rule 6: Diagonal 3s
        # If this cell is a 3, is there a diagonal neighbor 3?
        diag_3_move = self._check_diagonal_3s(cell, val)
        if diag_3_move:
             return diag_3_move

        return None, ""

    def _check_corner_3(self, cell, val, edges) -> Tuple[Optional[Move], str]:
        if val != 3: return None, ""
        
        r, c = cell
        rows = self.game_state.rows
        cols = self.game_state.cols
        graph = self.game_state.graph
        
        # Check if it's a corner cell
        is_tl = (r == 0 and c == 0)
        is_tr = (r == 0 and c == cols - 1)
        is_bl = (r == rows - 1 and c == 0)
        is_br = (r == rows - 1 and c == cols - 1)
        
        if not (is_tl or is_tr or is_bl or is_br):
            return None, ""
            
        # For a Corner 3, the two edges touching the extreme corner are FORCED.
        # TL (0,0) -> Vertex (0,0). Edges: Top (0,0)-(0,1), Left (0,0)-(1,0).
        target_edges = []
        if is_tl:
             target_edges = [((0,0), (0,1)), ((0,0), (1,0))]
        elif is_tr: # Vertex (0, cols)
             target_edges = [((0, cols-1), (0, cols)), ((0, cols), (1, cols))]
        elif is_bl: # Vertex (rows, 0)
             target_edges = [((rows, 0), (rows, 1)), ((rows-1, 0), (rows, 0))]
        elif is_br: # Vertex (rows, cols)
             target_edges = [((rows, cols-1), (rows, cols)), ((rows-1, cols), (rows, cols))]
             
        for e in target_edges:
            e_sorted = tuple(sorted(e))
            if e_sorted not in graph.edges:
                 # Check validity
                 valid, _ = is_valid_move(graph, e_sorted[0], e_sorted[1], self.game_state.clues, check_global=False)
                 if valid:
                     return e_sorted, f"Corner 3 at {cell} forces corner edges."
                     
        return None, ""

    def _check_adjacent_3s(self, cell, val) -> Tuple[Optional[Move], str]:
        if val != 3: return None, ""
        r, c = cell
        
        # Check neighbors: Up, Down, Left, Right
        neighbors = [
            (r-1, c, "Up"), (r+1, c, "Down"), (r, c-1, "Left"), (r, c+1, "Right")
        ]
        
        for nr, nc, direction in neighbors:
            if (nr, nc) in self.game_state.clues and self.game_state.clues[(nr, nc)] == 3:
                # Found 3-3 adjacency.
                # Rule: The shared edge is ON. The outer Parallel edges are ON.
                # Just forcing the shared edge is a good start.
                
                # Shared Edge:
                u, v = self._get_shared_edge(cell, (nr, nc))
                shared_edge = tuple(sorted((u, v)))
                
                graph = self.game_state.graph
                if shared_edge not in graph.edges:
                     valid, reason = is_valid_move(graph, u, v, self.game_state.clues, check_global=False)
                     if valid:
                         return shared_edge, f"Adjacent 3s ({cell} and {(nr,nc)}) force shared edge."

                
                # If shared edge is already done (or valid to be there), check Outer Edges.
                # Outer Edges for Horizontal Pair (Left, Right):
                # Left Cell's Left Edge. Right Cell's Right Edge.
                outer_edges = []
                if r == nr: # Horizontal
                    # Determine which is left/right
                    c_left, c_right = (c, nc) if c < nc else (nc, c)
                    # Left Edge of Left Cell (r, c_left): ((r, c_left), (r+1, c_left))
                    e1 = ((r, c_left), (r+1, c_left))
                    # Right Edge of Right Cell (r, c_right): ((r, c_right+1), (r+1, c_right+1))
                    e2 = ((r, c_right+1), (r+1, c_right+1))
                    outer_edges = [e1, e2]
                else: # Vertical
                    # Determine which is top/bottom
                    r_top, r_bot = (r, nr) if r < nr else (nr, r)
                    # Top Edge of Top Cell (r_top, c): ((r_top, c), (r_top, c+1))
                    e1 = ((r_top, c), (r_top, c+1))
                    # Bottom Edge of Bottom Cell (r_bot, c): ((r_bot+1, c), (r_bot+1, c+1))
                    e2 = ((r_bot+1, c), (r_bot+1, c+1))
                    outer_edges = [e1, e2]
                    
                for oe in outer_edges:
                    oe = tuple(sorted(oe))
                    if oe not in graph.edges:
                         valid, reason = is_valid_move(graph, oe[0], oe[1], self.game_state.clues, check_global=False)
                         if valid:
                             return oe, f"Adjacent 3s ({cell} and {(nr,nc)}) force outer edge."


 
                
        return None, ""

    def _check_diagonal_3s(self, cell, val) -> Tuple[Optional[Move], str]:
        if val != 3: return None, ""
        r, c = cell
        
        # Diagonals
        diags = [
            (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)
        ]
        
        for nr, nc in diags:
            if (nr, nc) in self.game_state.clues and self.game_state.clues[(nr, nc)] == 3:
                # Diagonal 3s touching at a vertex.
                # Forces the two edges touching the shared vertex to be ON (loop goes around the vertex).
                # Shared Vertex:
                # If (r,c) is TL and (nr, nc) is BR (r+1, c+1).
                # Shared vertex is (r+1, c+1).
                # Edges touching (r+1, c+1) belonging to (r,c) are Right and Bottom.
                # Edges touching (r+1, c+1) belonging to (r+1, c+1) are Top and Left.
                # These are actually the same 2 edges? No.
                # (r, c) Right edge: ((r, c+1), (r+1, c+1)). touches V(r+1, c+1).
                # (r, c) Bottom edge: ((r+1, c), (r+1, c+1)). touches V(r+1, c+1).
                
                # So for the pair, 4 edges touch the shared vertex.
                # These 4 valid moves? 
                # Slitherlink Logic: Loop cannot cross itself.
                # If two 3s share a vertex, the loop boundaries must typically go OUTWARD from the vertex.
                # Actually, the standard deduction is: The outer edges are ON?
                pass
                
        return None, ""

    def _get_shared_edge(self, c1, c2):
        r1, col1 = c1
        r2, col2 = c2
        if r1 == r2: # Horizontal neighbors
            c_min = min(col1, col2)
            # Shared is vertical edge between them: ((r1, c_min+1), (r1+1, c_min+1))
            # Wait. Cell (0,0) and (0,1). Shared is Right of (0,0) / Left of (0,1).
            # Right of (0,0) is ((0,1), (1,1)).
            return ((r1, c_min+1), (r1+1, c_min+1)) # Vertical
        else: # Vertical neighbors
            r_min = min(r1, r2)
            # Shared is horizontal between them: ((r_min+1, col1), (r_min+1, col1+1))
            return ((r_min+1, col1), (r_min+1, col1+1)) # Horizontal

    def _check_boundary_line(self, axis: str, fixed_idx: int, start: int, end: int) -> Tuple[Optional[Move], str]:
        """
        Check edges along a dividing line (horizontal or vertical).
        """
        # Iterate along the boundary
        for i in range(start, end + 1):
            if axis == "horizontal":
                # Edge is vertical crossing the horizontal line? 
                # No, a horizontal split means we have a horizontal line of edges.
                # Row fixed_idx is the row index of the vertices on the split line.
                # Horizontal edges are ((fixed_idx, i), (fixed_idx, i+1))
                u = (fixed_idx, i)
                v = (fixed_idx, i + 1)
            else: # vertical
                # Column fixed_idx
                # Vertical edges are ((i, fixed_idx), (i+1, fixed_idx))
                u = (i, fixed_idx)
                v = (i + 1, fixed_idx)
                
            edge = tuple(sorted((u, v)))
            
            # Check consistency of this edge
            # This is where "Merge" logic happens. 
            # Does this edge connect two components required by clues on either side?
            
            # Get adjacent cells to this edge
            cells = self._get_adjacent_cells(u, v)
            
            # Simple check: If both adjacent cells effectively force this edge?
            # Or if one cell forces it?
            
            # Iterate clues of adjacent cells
            for cell in cells:
                if cell in self.game_state.clues:
                    # Reuse cell heuristic
                    move, reason = self._apply_cell_heuristics(cell, self.game_state.clues[cell])
                    if move == edge:
                         return move, f"Boundary edge forced by cell {cell}"
                         
        return None, ""

    def _get_adjacent_cells(self, u: Tuple[int, int], v: Tuple[int, int]) -> List[Tuple[int, int]]:
        rows = self.game_state.rows
        cols = self.game_state.cols
        r1, c1 = u
        r2, c2 = v
        adj = []
        
        if r1 == r2: # Horizontal Edge
            # Cell above: (r1-1, min_c)
            # Cell below: (r1, min_c)
            c_min = min(c1, c2)
            if r1 > 0: adj.append((r1 - 1, c_min))
            if r1 < rows: adj.append((r1, c_min))
        else: # Vertical Edge
            # Cell left: (min_r, c1-1)
            # Cell right: (min_r, c1)
            r_min = min(r1, r2)
            if c1 > 0: adj.append((r_min, c1 - 1))
            if c1 < cols: adj.append((r_min, c1))
            
        return adj
