# Loopy DAA - Slitherlink AI Duel

**Course**: Design and Analysis of Algorithms (DAA)  
**Project**: Phase 1 - Slitherlink Game with AI  

## Overview
This is a Python-based implementation of the puzzle game **Slitherlink** (also known as Loopy). It features a polished "Apple Professional" dark UI, a smart Hint system, and an AI opponent powered by Greedy algorithms and Sorting.

## Features
- **Game Modes**: Easy (4x4), Medium (5x5), Hard (7x7).
- **AI Opponent**: A Greedy CPU that uses Sorting algorithms to pick the best moves.
- **Smart Hints**: Guarantees a winnable path using internal graph solving logic.
- **Undo/Redo**: Full history support.
- **Statistics**: Tracks wins, losses, and games played.
- **Professional UI**: Radiant Dark theme with neon accents and smooth animations.

## DAA Syllabus Integration
This project explicitly demonstrates the following Unit 1 & 2 topics:
1.  **Graph Representation**: The board is modeled as a graph (Adjacency List) in `logic/graph.py`.
2.  **Graph Algorithms**:
    *   **BFS/DFS**: Used for board generation and connectivity checks (`daa/graph_algos.py`).
    *   **Cycle Detection**: Used to validate loops.
3.  **Sorting**: Bubble, Insertion, and Selection Sort are implemented in `daa/sorting.py` and used by the CPU to rank moves.
4.  **Asymptotic Analysis**: Complexity notes are provided in `daa/analysis_notes.py`.

## How to Run
1.  Ensure you have Python 3.x installed.
2.  Run the main script:
    ```bash
    python main.py
    ```

## Project Structure
*   `main.py`: Entry point.
*   `ui/`: User Interface components (Tkinter).
*   `logic/`: Core game logic (GameState, Validators, CPU).
*   `daa/`: Algorithms implementation (Sorting, Graphs).
*   `data/`: Stores game statistics.

## Controls
*   **Left Click**: Toggle edge (Line / Empty).
*   **Undo/Redo**: Use on-screen buttons.
*   **Hint**: Request a move suggestion.

---
*Generated for DAA Course Submission*
