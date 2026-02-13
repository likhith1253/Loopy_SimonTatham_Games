"""
Dynamic Programming Solver
==========================
Implements a strong Pure Dynamic Programming solver for Slitherlink using State Compression (Row-Based DP).
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Set, Tuple

from logic.solvers.solver_interface import AbstractSolver, HintPayload


class DynamicProgrammingSolver(AbstractSolver):
    def __init__(self, game_state: Any):
        self.game_state = game_state
        self.rows = game_state.rows
        self.cols = game_state.cols
        self.last_explanation = ""
        self.solution_moves = []
        self.current_move_index = 0
        self._solution_computed = False
        self.dp_state_count = 0  # Trace Metric
        self.memo_hits = 0
        # Don't compute solution yet - wait for explicit call
        # self._compute_full_solution()

    def decide_move(self) -> Tuple[List[Tuple[Tuple[int, int], int]], Optional[Tuple[int, int]]]:
        """
        Required by UI to simulate 'thinking' and return candidates + best move.
        DP Solver uses precomputed solution moves.
        """
        # Check if it's actually CPU's turn
        is_cpu_turn = (self.game_state.game_mode in ["vs_cpu", "expert"] and 
                      self.game_state.turn == "Player 2 (CPU)")
        
        if not is_cpu_turn:
            return [], None
        
        # Ensure solution is computed before using it
        if not self._solution_computed:
            self._compute_full_solution()
            
        # Find next valid move from precomputed solution
        while self.current_move_index < len(self.solution_moves):
            move_data = self.solution_moves[self.current_move_index]
            move = move_data["move"]
            
            # Check if move is still valid
            from logic.validators import is_valid_move
            valid, reason = is_valid_move(self.game_state.graph, move[0], move[1], self.game_state.clues)
            
            if valid:
                self.last_explanation = move_data.get("explanation", "Dynamic Programming precomputed move.")
                
                # Global Execution Trace Log
                from logic.execution_trace import log_pure_dp_move
                log_pure_dp_move(
                    move=move,
                    explanation=self.last_explanation,
                    dp_state_count=self.dp_state_count
                )

                return [(move, 100)], move
            else:
                # Skip invalid move and increment index
                self.current_move_index += 1
                self.last_explanation = f"Skipping invalid precomputed move: {reason}"
        
        self.last_explanation = "Dynamic Programming has completed all precomputed moves."
        return [], None

    def solve(self, board: Any = None):
        """
        Returns the next move from precomputed solution.
        """
        # Ensure solution is computed before using it
        if not self._solution_computed:
            self._compute_full_solution()
            
        # Find next valid move from precomputed solution
        while self.current_move_index < len(self.solution_moves):
            move_data = self.solution_moves[self.current_move_index]
            move = move_data["move"]
            
            # Check if move is still valid
            from logic.validators import is_valid_move
            valid, reason = is_valid_move(self.game_state.graph, move[0], move[1], self.game_state.clues)
            
            if valid:
                from logic.execution_trace import log_pure_dp_move
                log_pure_dp_move(
                    move=move,
                    explanation=move_data.get("explanation", ""),
                    dp_state_count=self.dp_state_count
                )
                return move
            else:
                # Skip invalid move and increment index
                self.current_move_index += 1
                
        return None

    def generate_hint(self, board: Any = None) -> HintPayload:
        target = board if board is not None else self.game_state
        
        # Ensure solution is computed before using it
        if not self._solution_computed:
            self._compute_full_solution()
        
        # Check if it's human turn (only provide hints for human player)
        is_human_turn = (self.game_state.game_mode in ["vs_cpu", "expert"] and 
                        self.game_state.turn == "Player 1 (Human)") or \
                       (self.game_state.game_mode not in ["vs_cpu", "expert"])
        
        if not is_human_turn:
            return {
                "move": None,
                "strategy": "Dynamic Programming",
                "explanation": "Hints are only available during your turn."
            }
        
        # Get solution edges from the computed solution
        solution_edges = set()
        for move_data in self.solution_moves:
            solution_edges.add(move_data["move"])
        
        current_edges = set(self.game_state.graph.edges)
        
        # PRIORITY 1: Look for solution moves to add (constructive hints first)
        for i in range(self.current_move_index, len(self.solution_moves)):
            move_data = self.solution_moves[i]
            move = move_data["move"]
            
            # Skip if this is the next CPU move (don't give away CPU's immediate move)
            if i == self.current_move_index:
                continue
            
            # Check if move is still valid and not already on board
            if move not in current_edges:
                from logic.validators import is_valid_move
                valid, reason = is_valid_move(self.game_state.graph, move[0], move[1], self.game_state.clues)
                
                if valid:
                    return {
                        "move": move,
                        "strategy": "Dynamic Programming (State Compression)",
                        "explanation": move_data.get("explanation", f"This edge is part of the optimal solution path found by dynamic programming.")
                    }
        
        # PRIORITY 2: Use DP analysis for forced/forbidden edges
        all_solutions = self._run_dp(target, limit=2000)
        
        if not all_solutions:
             return {"move": None, "strategy": "Dynamic Programming", "explanation": "No valid solutions found via DP."}

        total_solutions = len(all_solutions)
        edge_counts = collections.defaultdict(int)
        
        for sol in all_solutions:
            for edge in sol:
                edge_counts[edge] += 1
                
        # Find Forced Edges (Present in ALL solutions) - these are great hints
        forced_edges = []
        for edge, count in edge_counts.items():
            if count == total_solutions:
                if edge not in target.graph.edges:
                    forced_edges.append(edge)
                    
        # Find Forbidden Edges (Present in NO solutions) - only suggest if no forced edges
        forbidden_edges = []
        all_potential_edges = self._get_all_potential_edges()
        for edge in all_potential_edges:
            if edge_counts[edge] == 0:
                if edge in target.graph.edges:
                    forbidden_edges.append(edge)

        # Priority: 1) Forced edges to add, 2) Forbidden edges to remove
        if forced_edges:
            move = forced_edges[0]
            explanation = f"Using Dynamic Programming: This edge must be selected because it appears in all {total_solutions} valid solution(s)."
            strategy = "Dynamic Programming (State Compression)"
        elif forbidden_edges:
            move = forbidden_edges[0]
            explanation = f"Using Dynamic Programming: This edge must be removed because it appears in 0 of {total_solutions} valid solution(s)."
            strategy = "Dynamic Programming (State Compression)"
        else:
            # PRIORITY 3: If no forced/forbidden edges, look for any incorrect edges to remove
            incorrect_edges = []
            for edge in current_edges:
                if edge not in solution_edges:
                    incorrect_edges.append(edge)
            
            if incorrect_edges:
                # Return first incorrect edge for removal with specific reasoning
                move = incorrect_edges[0]
                
                # Generate specific reasoning for why this edge should be removed
                explanation = self._generate_edge_removal_reasoning(move, target)
                
                return {
                    "move": move,
                    "strategy": "Dynamic Programming (State Compression)",
                    "explanation": explanation
                }
            else:
                # No hints available
                return {
                    "move": None,
                    "strategy": "Dynamic Programming",
                    "explanation": "No specific hints available. Try any valid move."
                }
        
        return {
            "move": move,
            "strategy": strategy,
            "explanation": explanation
        }

    def explain_last_move(self) -> str:
        return self.last_explanation
    
    def _generate_edge_removal_reasoning(self, edge: Edge, target) -> str:
        """
        Generate specific reasoning for why an edge should be removed.
        """
        from logic.validators import is_valid_move
        
        # Check if removing this edge fixes any clue violations
        u, v = edge
        
        # Find affected cells (cells that this edge borders)
        affected_cells = []
        
        # Edge is horizontal: ((r, c), (r, c+1))
        if u[0] == v[0]:  # Same row
            r = u[0]
            c1 = min(u[1], v[1])
            c2 = max(u[1], v[1])
            
            # Cell above the edge
            if r > 0:
                affected_cells.append((r-1, c1))
            # Cell below the edge  
            if r < target.rows:
                affected_cells.append((r, c1))
                
        # Edge is vertical: ((r, c), (r+1, c))
        else:  # Same column
            c = u[1]
            r1 = min(u[0], v[0])
            r2 = max(u[0], v[0])
            
            # Cell to the left of the edge
            if c > 0:
                affected_cells.append((r1, c-1))
            # Cell to the right of the edge
            if c < target.cols:
                affected_cells.append((r1, c))
        
        # Check each affected cell for clue violations
        violations = []
        for cell in affected_cells:
            if cell in target.clues:
                clue_val = target.clues[cell]
                
                # Count current edges around this cell
                current_count = 0
                cell_r, cell_c = cell
                
                # Check all 4 edges around the cell
                edges_around = [
                    ((cell_r, cell_c), (cell_r, cell_c+1)),  # right
                    ((cell_r, cell_c), (cell_r+1, cell_c)),  # bottom
                    ((cell_r-1, cell_c), (cell_r, cell_c)),  # top
                    ((cell_r, cell_c-1), (cell_r, cell_c)),  # left
                ]
                
                for check_edge in edges_around:
                    check_edge_sorted = tuple(sorted(check_edge))
                    if check_edge_sorted in target.graph.edges:
                        # Don't count the edge we're about to remove
                        if check_edge_sorted != edge:
                            current_count += 1
                
                # If current count exceeds clue, this edge must be removed
                if current_count > clue_val:
                    violations.append(f"Cell ({cell_r}, {cell_c}) has clue {clue_val} but currently has {current_count} edges")
                # If removing this edge would make it impossible to satisfy the clue
                elif current_count == clue_val and edge in target.graph.edges:
                    violations.append(f"Cell ({cell_r}, {cell_c}) already satisfies clue {clue_val}")
        
        if violations:
            return f"Remove this edge because: {violations[0]}. This violates the puzzle constraints."
        
        # Check for degree violations (nodes with degree > 2)
        for node in [u, v]:
            degree = 0
            for neighbor in target.graph.get_neighbors(node):
                if tuple(sorted((node, neighbor))) in target.graph.edges:
                    degree += 1
            
            # Count the edge we're checking
            if edge in target.graph.edges:
                degree += 1
                
            if degree > 2:
                return f"Remove this edge because node {node} would have degree {degree} (maximum allowed is 2)."
        
        # Default reasoning
        return f"Remove this edge because it doesn't appear in any valid solutions found by dynamic programming analysis."
    
    def register_move(self, move):
        """
        Called after a move is made to update the current move index.
        """
        if self.current_move_index < len(self.solution_moves):
            expected_move = self.solution_moves[self.current_move_index]["move"]
            if move == expected_move:
                self.current_move_index += 1
            else:
                # Move doesn't match expected - recompute solution
                self._recompute_solution()
    
    def _recompute_solution(self):
        """
        Recompute solution based on current board state
        """
        # Reset solution computation flag to force recompute
        self._solution_computed = False
        self.current_move_index = 0
        self._compute_full_solution()
    
    def _compute_full_solution(self):
        """
        Compute full solution once at initialization and store as ordered list.
        Uses the game state's precomputed solution edges if available.
        """
        # Use the game state's solution edges if they exist
        solution_edges = getattr(self.game_state, 'solution_edges', None)
        
        if solution_edges is not None and len(solution_edges) > 0:
            # Use game state's precomputed solution
            pass
        else:
            # Fallback to computing solution via DP
            solutions = self._run_dp(self.game_state, limit=1)
            
            if not solutions:
                self.solution_moves = []
                self._solution_computed = True
                return
                
            solution_edges = solutions[0]
            
        current_edges = set(self.game_state.graph.edges)
        
        # Build ordered list of moves to transform current state to solution
        self.solution_moves = []
        
        # Add all missing solution edges
        edges_to_add = []
        for edge in solution_edges:
            if edge not in current_edges:
                edges_to_add.append(edge)
                
        for edge in edges_to_add:
            self.solution_moves.append({
                "move": edge,
                "explanation": f"Add edge {edge} - part of the optimal solution path.",
                "dp_state_reference": "solution_edge"
            })
        
        # Remove all incorrect edges (if any)
        for edge in current_edges:
            if edge not in solution_edges:
                self.solution_moves.append({
                    "move": edge,
                    "explanation": f"Remove edge {edge} - not part of the optimal solution.",
                    "dp_state_reference": "incorrect_edge"
                })
        
        # Mark solution as computed
        self._solution_computed = True

    def _get_all_potential_edges(self) -> List[Tuple[int, int]]:
        edges = []
        for r in range(self.rows + 1):
            for c in range(self.cols):
                 edges.append(tuple(sorted(((r, c), (r, c+1)))))
        for r in range(self.rows):
            for c in range(self.cols + 1):
                 edges.append(tuple(sorted(((r, c), (r+1, c)))))
        return edges

    def _run_dp(self, target, limit=1) -> List[Set[Tuple[int, int]]]:
        """
        Executes DP to find 'limit' valid solutions.
        Returns a list of edge-sets.
        """
        self.dp_state_count = 0
        self.memo_hits = 0

        rows = self.rows
        cols = self.cols
        clues = target.clues
        
        # State: (h_mask, comps_tuple, has_closed_loop)
        # parent pointer: (prev_state, edges_added_in_step)
        
        # Row 0 Init
        # Initial boundary has no vertical edges coming from above.
        # Initial comps is all 0.
        # Initial closed_loop is False.
        # However, Top Row edges (H0) need to be decided.
        # Since we treat Row 0 as boundary, we just iterate H0.
        
        dp = [collections.defaultdict(list) for _ in range(rows + 1)]
        
        # Init with clean state at Row 0
        # In Row 0, we decide H-edges for top line.
        # Vertical edges from -1 don't exist.
        for h_val in range(1 << cols):
            h_bits = [(h_val >> c) & 1 for c in range(cols)]
            
            # Check valid components at Line 0
            # Degree at (0, c) = H[c-1] + H[c]
            # Must be 0 or 2 (Closed/Corner) or 1 (Endpoint). 
            # Cannot be > 2.
            
            valid_h = True
            current_comps = [0] * (cols + 1)
            uf = UnionFind(cols + 1)
            
            # Form Union Find for Line 0 connections
            for c in range(cols):
                if h_bits[c]:
                    # Edge exists between c and c+1
                    uf.union(c, c+1)
            
            # Determine signatures
            sig = []
            has_edge = any(h_bits)
            
            for c in range(cols + 1):
                left = h_bits[c-1] if c > 0 else 0
                right = h_bits[c] if c < cols else 0
                deg = left + right
                if deg > 2:
                    valid_h = False; break
                
                if deg == 1:
                    sig.append(uf.find(c) + 1) # Shift to 1-based (0 is reserved for closed)
                else:
                    sig.append(0) 
            
            if not valid_h: continue
            
            # Check for closed loop at Top Line?
            # A line cannot form a loop unless it wraps around cylinder, which this is not.
            # So has_closed_loop is False.
            
            # Canonicalize
            can_sig = normalize_signature(tuple(sig))
            
            state = (h_val, can_sig, False) # initial has_closed_loop = False
            
            # Edges added
            edges = set()
            for c in range(cols):
                if h_bits[c]: edges.add(tuple(sorted(((0, c), (0, c+1)))))
            
            if state in dp[0]:
                self.memo_hits += 1
            dp[0][state].append((None, edges))

        # Iterate Rows
        for r in range(rows):
            prev_layer_idx = r
            curr_layer_idx = r + 1
            prev_dp = dp[prev_layer_idx]
            
            if not prev_dp:
                break
            
            # Precompute clues for this row to save lookups
            row_clue_dict = {}
            for c in range(cols):
                if (r, c) in clues: row_clue_dict[c] = clues[(r, c)]
            
            count_solutions_found = 0
            
            # Iterate prev states
            for (prev_h, prev_comps, prev_closed_loop), parents in prev_dp.items():
                self.dp_state_count += 1

                
                # If we already have enough solutions at end, stop?
                # Optimization: We only check limit at the end usually, but if count explodes...
                
                prev_h_bits = [(prev_h >> c) & 1 for c in range(cols)]
                
                # Constraint: If closed loop exists, we CANNOT add any more edges.
                # So if prev_closed_loop is True:
                #   Only valid curr_h is 0.
                #   Vertical edges MUST be 0.
                #   State remains (0, 0...0, True).
                #   We still check clues (must allow empty).
                
                possible_h = [0] if prev_closed_loop else range(1 << cols)
                
                for curr_h in possible_h:
                    curr_h_bits = [(curr_h >> c) & 1 for c in range(cols)]
                    
                    # 1. Determine Vertical Edges
                    v_edges = []
                    valid_transition = True
                    
                    for c in range(cols + 1):
                        deg_in = 1 if prev_comps[c] != 0 else 0
                        # V-edge is strictly determined by parity of node (r, c)
                        # prev_comps captures (Up + Left + Right) % 2
                        
                        if deg_in == 1:
                            v_edges.append(1)
                        else:
                            # If deg_in is 0 (Even), V MUST be 0 to keep it Even.
                            # (Cannot start new path downwards from nothing)
                            v_edges.append(0)
                            
                    if not valid_transition: continue
                    
                    # If closed loop existed, ensure V edges are 0 (should be auto if H=0 and deg_in=0)
                    if prev_closed_loop and (any(v_edges) or any(curr_h_bits)):
                        continue
                        
                    # 2. Check Clues
                    clue_fail = False
                    for c in range(cols):
                        if c in row_clue_dict:
                            target_val = row_clue_dict[c]
                            actual_val = prev_h_bits[c] + curr_h_bits[c] + v_edges[c] + v_edges[c+1]
                            if actual_val != target_val:
                                clue_fail = True; break
                    if clue_fail: continue

                    # 3. Connectivity & Cycle Detection
                    # Map IDs from Line r (prev_comps) to Line r+1
                    # Nodes: 0..cols at Line r+1.
                    
                    # Logic:
                    # - Vertical edges connect Node(r, c) ID to Node(r+1, c).
                    # - Horizontal edges connect Node(r+1, c) to Node(r+1, c+1).
                    
                    next_id_counter = 1
                    # Reuse IDs from prev where possible to keep them stable before normalizing
                    # But simpler to just map to a temp space 1..100
                    
                    # We use a localized Union-Find for the transition
                    # Nodes to track:
                    #   - Components from Line r (active ones)
                    #   - New nodes at Line r+1
                    
                    # Let's assign temporary unique labels to active components in prev_comps
                    # e.g. if prev_comps is (1, 0, 1, 2), we have global components 1 and 2.
                    # We perform union operations on these global IDs.
                    
                    # Max ID in prev_comps
                    curr_max_id = max(prev_comps) if prev_comps else 0
                    next_id_gen = curr_max_id + 1
                    
                    uf_map = {} # map component ID to parent
                    def find(i):
                        if i not in uf_map: uf_map[i] = i
                        if uf_map[i] != i: uf_map[i] = find(uf_map[i])
                        return uf_map[i]
                    def union(i, j):
                        root_i, root_j = find(i), find(j)
                        if root_i != root_j: uf_map[root_j] = root_i; return True
                        return False
                        
                    # 3a. Vertical Connections
                    # Establish IDs for nodes on Line r+1
                    node_ids_r_plus_1 = [0] * (cols + 1)
                    
                    for c in range(cols + 1):
                        if v_edges[c]:
                            # Connected to Top
                            if prev_comps[c] != 0:
                                val = prev_comps[c]
                                node_ids_r_plus_1[c] = val
                            else:
                                # New component started (Degree was 0 or 2 at top, but now edge down -> must be new path start?)
                                # Wait. If deg at top was 0, sum was 0. V=0.
                                # If deg at top was 2, sum was 2. V=0.
                                # If sum was 1, V=1.
                                # So if V=1, prev_comps[c] MUST be != 0 ??
                                # NO.
                                # Check logic: current_sum = deg_in + h_left + h_right.
                                # If prev_comps[c] == 0 (deg_in=0), and h_left=1, h_right=0 -> sum=1 -> V=1.
                                # So yes, we can start a vertical edge from a Horizontal corner.
                                # In this case, the component ID comes from the H-edge?
                                # No, H-edge connects to neighbor.
                                # We need to assign a temporary ID for the "start" of this strand.
                                node_ids_r_plus_1[c] = next_id_gen
                                next_id_gen += 1
                        else:
                            node_ids_r_plus_1[c] = 0 # No vertical edge to this node
                            
                    # 3b. Horizontal Connections (Merge)
                    loop_just_closed = False
                    
                    for c in range(cols):
                        if curr_h_bits[c]:
                            # Nodes c and c+1 are connected
                            id1 = node_ids_r_plus_1[c]
                            id2 = node_ids_r_plus_1[c+1]
                            
                            # If IDs are 0 (meaning no V-edge connected to them), they are "fresh" nodes.
                            # Assign new ID if both 0.
                            if id1 == 0 and id2 == 0:
                                new_id = next_id_gen
                                next_id_gen += 1
                                node_ids_r_plus_1[c] = new_id
                                node_ids_r_plus_1[c+1] = new_id
                                # Register in UF
                                find(new_id)
                            elif id1 != 0 and id2 == 0:
                                node_ids_r_plus_1[c+1] = id1
                            elif id1 == 0 and id2 != 0:
                                node_ids_r_plus_1[c] = id2
                            else:
                                # Both have IDs. Merge.
                                if find(id1) == find(id2):
                                    loop_just_closed = True
                                else:
                                    union(id1, id2)
                    
                    # 4. Generate Next Signature
                    next_sig_list = []
                    current_open_ends = 0
                    
                    for c in range(cols + 1):
                        # Calculate degree at bottom node (Line r+1)
                        # We only care if it passes "down" or is "loose" for next step.
                        # Wait, the signature for next step is defined by loose ends at Line r+1.
                        # Degree at (r+1, c) SO FAR = V_up (v_edges[c]) + H_left + H_right.
                        
                        v_up = v_edges[c]
                        h_left = curr_h_bits[c-1] if c > 0 else 0
                        h_right = curr_h_bits[c] if c < cols else 0
                        deg = v_up + h_left + h_right
                        
                        if deg % 2 == 1:
                            # Loose end.
                            # Get root ID
                            root = find(node_ids_r_plus_1[c]) # Use the ID assigned to this node
                            next_sig_list.append(root)
                            current_open_ends += 1
                        else:
                            next_sig_list.append(0)
                        
                        if deg > 2: # Invalid
                            valid_transition = False; break
                            
                    if not valid_transition: continue
                    
                    # 5. Cycle Validity via Disappearance
                    # A component 'closes' if it existed efficiently but is no longer "loose" (in next_sig)
                    
                    prev_ids = set(x for x in prev_comps if x > 0)
                    closed_loop_detected = False
                    
                    for pid in prev_ids:
                        root = find(pid)
                        if root not in next_sig_list:
                            closed_loop_detected = True
                            
                    next_closed_loop = prev_closed_loop
                    
                    if closed_loop_detected:
                        # If a loop closed, it must be valid
                        # 1. No other strands allowed (Disconnected loop check)
                        if any(x > 0 for x in next_sig_list):
                            # print(f"      Pruned: Loop closed but strands remain {next_sig_list}")
                            continue
                        
                        # 2. No double loops
                        if prev_closed_loop:
                            continue
                            
                        next_closed_loop = True
                    
                    # Normalization
                    can_sig = normalize_signature(tuple(next_sig_list))
                    
                    new_state = (curr_h, can_sig, next_closed_loop)
                    
                    # Store
                    edges_in_step = set()
                    for c in range(cols):
                        if curr_h_bits[c]: edges_in_step.add(tuple(sorted(((r+1, c), (r+1, c+1)))))
                    for c in range(cols + 1):
                        if v_edges[c]: edges_in_step.add(tuple(sorted(((r, c), (r+1, c)))))

                    if new_state in dp[curr_layer_idx]:
                        self.memo_hits += 1
                    dp[curr_layer_idx][new_state].append(((prev_h, prev_comps, prev_closed_loop), edges_in_step))
        
        
        # Extract Solutions from Final Row
        final_valid_states = []
        for state, parents in dp[rows].items():
            (h, sig, closed) = state
            # Valid end state:
            # 1. Closed Loop formed (closed=True).
            # 2. No open loose ends (sig is all 0s).
            # 3. Last row clues satisfied (Checked in loop).
            
            if closed and all(s == 0 for s in sig):
                final_valid_states.append(state)
                
        # Reconstruct paths
        full_solutions = []
        
        def backtrack(r, state, current_edges):
            if len(full_solutions) >= limit: return
            
            if r == 0:
                full_solutions.append(current_edges)
                return
                
            parents = dp[r][state]
            for (prev_state, edges_in_step) in parents:
                # Merge edges
                new_set = current_edges.union(edges_in_step)
                backtrack(r-1, prev_state, new_set)
                if len(full_solutions) >= limit: return
                
        for state in final_valid_states:
             backtrack(rows, state, set())
             
        return full_solutions

def normalize_signature(sig: Tuple[int, ...]) -> Tuple[int, ...]:
    mapping = {}
    new_sig = []
    next_id = 1
    for val in sig:
        if val == 0:
            new_sig.append(0)
        else:
            if val not in mapping:
                mapping[val] = next_id
                next_id += 1
            new_sig.append(mapping[val])
    return tuple(new_sig)

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False
