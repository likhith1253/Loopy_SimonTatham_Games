# Loopy DAA Project - Phase 2 Architecture Design

## 1. Overview
Phase 2 introduces a "Hybrid" architecture combining **Divide & Conquer (D&C)** and **Dynamic Programming (DP)**. The goal is to parallelize the core solver, introduce a campaign mode with procedural generation, and provide real-time visualization and benchmarking.

---

## 2. Hybrid Algorithm Logic: Scaling to Higher Orders
This architecture solves N × N grids efficiently by combining D&C's structural decomposition with DP's local optimization.

### 2.1. The "Divide": Isolating Complexity
- **Recursive Decomposition**: A large grid (e.g., 16 × 16) is recursively split into 4 quadrants (8 × 8 → 4 × 4) using **Divide & Conquer**. 
- **Effect**: This reduces the effective search space. Instead of solving one massive problem with 2^256 states, we solve multiple independent smaller problems.

### 2.2. The "Conquer": DP Base Cases
- **Leaf Node Efficiency**: At the leaf level (4 × 4 sectors), standard backtracking is still too slow if repeated thousands of times.
- **Bitmask DP**: We use **Dynamic Programming** to solve these leaves. 
    - **State Definition**: `dp(mask, boundary_profile)`.
    - **Pre-computation**: For a given set of boundary edges (profile), the DP calculates valid internal loop segments immediately without recursion.
    - **Caching**: Common boundary patterns are memoized.

### 2.3. The "Combine": Stitching Solutions
- **Merge Step**: The `ParallelDnCSolver` takes the fast DP results from the leaves and "stitches" them.
- **Conflict Resolution**: If Quadrant A's right edge doesn't match Quadrant B's left edge, the branch is pruned immediately at the D&C level, avoiding deep invalid searches.

**Summary**: D&C handles the *global* structure (parallelizable), while DP handles the *local* microstructure (deterministic and fast). This allows scaling to board sizes that would freeze a standard solver.

---

## 3. Module Design Specifications

### 3.1. The Parallel Solver Engine
**Directory**: `logic/solvers/`

#### Class: `ParallelDnCSolver`
Uses `concurrent.futures` to solve sub-quadrants in separate threads.
- **Algorithm**: Divide & Conquer (Parallel)
- **Key Methods**:
  - `solve(grid: Grid) -> Solution`: Main entry point. Divides grid into quadrants.
  - `_solve_quadrant(sub_grid: Grid) -> Future[Solution]`: Submits a quadrant task to the thread pool.
  - `merge_solutions(top_left, top_right, bottom_left, bottom_right) -> Solution`: Combines sub-solutions.

#### Class: `BitmaskDPSolver`
Base-case solver for small (e.g., 4x4) grids to ensure 100% accuracy at the recursion leaves.
- **Algorithm**: Dynamic Programming (Bitmask)
- **Key Methods**:
  - `solve_small_grid(grid: Grid) -> Solution`: Solves a small grid using DP state transitions.
  - `get_state(mask: int, profile: List[int]) -> int`: Returns DP table value for a boundary profile.

#### Class: `SolverManager`
Manages the solving process, allowing pause/resume/stop functionality.
- **Key Methods**:
  - `start_solving()`: Initializes the `ParallelDnCSolver`.
  - `pause()`: Suspends thread execution (using synchronization primitives).
  - `resume()`: Resumes execution.
  - `stop()`: Terminates all threads safely.

### 3.2. The Game Mode Engine
**Directory**: `logic/campaign/`

#### Class: `Knapsack01Logic`
Logic for "Campaign Mode". Players select edges to match a specific target sum.
- **Algorithm**: Dynamic Programming (0/1 Knapsack)
- **Key Methods**:
  - `calculate_optimal_selection(edges: List[Edge], target_sum: int) -> List[Edge]`: Returns the set of edges that sum to `target_sum`.
  - `validate_selection(player_selection: List[Edge], target_sum: int) -> bool`: Checks if player's move is valid.

#### Class: `RecursiveLevelGenerator`
Procedurally generates puzzle maps using D&C.
- **Algorithm**: Divide & Conquer
- **Key Methods**:
  - `generate_level(size: int, difficulty: int) -> Grid`: Top-level generator.
  - `_generate_sector(size: int) -> Grid`: Recursively generates a valid sub-sector.
  - `stitch_sectors(sectors: List[Grid]) -> Grid`: Combines sectors ensuring boundary consistency.

### 3.3. The Visualizer & Analyst
**Directory**: `ui/visualization/`

#### Class: `RecursionOverlay`
Draws "Cut Lines" on the board to visualize the D&C process.
- **Key Methods**:
  - `draw_cut_lines(canvas: Canvas, depth: int)`: Renders division lines based on recursion depth.
  - `update_progress(thread_id: int, status: str)`: Updates visual cues based on thread activity (thread-safe).

#### Class: `DashboardUI`
Tabs for live performance comparison.
- **Key Methods**:
  - `render_graph(sequential_data: List[float], parallel_data: List[float])`: Plots execution time comparison.
  - `update_metrics(cpu_usage: float, active_threads: int)`: Displays real-time system stats.

---

## 4. File Structure
```
root/
├── logic/
│   ├── solvers/
│   │   ├── parallel_solver.py
│   │   ├── bitmask_solver.py
│   │   └── solver_manager.py
│   ├── campaign/
│   │   ├── knapsack_logic.py
│   │   └── level_generator.py
│   └── ...
├── ui/
│   ├── visualization/
│   │   ├── recursion_overlay.py
│   │   └── dashboard_ui.py
│   └── ...
└── ...
```

---

## 5. Complexity Analysis: Recurrence Relations

### Sequential Divide & Conquer
In the standard sequential approach, the problem is divided into 4 sub-problems of size `n/2` (for a 2D grid), plus the overhead of splitting and merging (`O(n)`).

**T(n) = 4T(n/2) + O(n)**
- **Master Theorem Case**: a=4, b=2, d=1. Since a > b^d (4 > 2^1), complexity is dominated by leaves: O(n^(log_2 4)) = O(n^2).

### Parallel Divide & Conquer
In the parallel approach, if we assume infinite processors (or sufficiently many threads), the 4 sub-problems are solved simultaneously. The time complexity is determined by the *slowest* branch (one sub-problem) plus the merge overhead.

**T(n) = T(n/2) + O(n)**
- **Master Theorem Case**: a=1, b=2, d=1. Since a < b^d (1 < 2^1), complexity is dominated by the root work: O(n).
- **Note**: In practice, limited cores mean we don't achieve perfect scaling, but the critical path is significantly reduced.
