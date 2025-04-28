#!/usr/bin/env python3
import sys
import heapq

# Directions: North, South, West, East
MOVES = {
    'N': (-1,  0),
    'S': ( 1,  0),
    'W': ( 0, -1),
    'E': ( 0,  1),
}


def load_world(path):
    """
    Reads the world file (handles UTF-8 or UTF-16) and returns (cols, rows, grid).
    grid is a list of list of chars.
    """
    # Read raw bytes
    with open(path, 'rb') as bf:
        raw = bf.read()
    # Detect BOM for UTF-16
    if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        text = raw.decode('utf-16')
    else:
        text = raw.decode('utf-8-sig')
    lines = text.splitlines()
    # Parse header
    cols = int(lines[0].strip())
    rows = int(lines[1].strip())
    # Read grid rows
    grid = [list(lines[2 + r]) for r in range(rows)]
    return cols, rows, grid


def find_start_and_dirt(grid, rows, cols):
    """
    Scans the grid for '@' (robot start) and '*' (dirty cells).
    Returns (start_state).
    start_state = ((r, c), frozenset of dirt positions)
    """
    dirt = set()
    robot = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '@':
                robot = (r, c)
                grid[r][c] = '_'  # treat start as empty
            elif grid[r][c] == '*':
                dirt.add((r, c))
    return (robot, frozenset(dirt))


def successors(state, grid, rows, cols):
    """
    Yields (action, next_state) from the given state.
    state = (robot_pos, dirt_set)
    """
    (r, c), dirt = state
    # Movement
    for action, (dr, dc) in MOVES.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
            yield action, ((nr, nc), dirt)
    # Vacuum
    if (r, c) in dirt:
        new_dirt = set(dirt)
        new_dirt.remove((r, c))
        yield 'V', ((r, c), frozenset(new_dirt))


def goal(state):
    """True if no dirty cells remain."""
    return not state[1]


def depth_first(start, grid, rows, cols):
    """
    Depth-First Search with cycle detection.
    Returns (plan, gen_count, exp_count).
    """
    stack = [(start, [])]
    visited = set()
    generated = expanded = 0

    while stack:
        state, path = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        expanded += 1
        if goal(state):
            return path, generated, expanded
        for action, nxt in successors(state, grid, rows, cols):
            generated += 1
            stack.append((nxt, path + [action]))
    return None, generated, expanded


def uniform_cost(start, grid, rows, cols):
    """
    Uniform-Cost Search (all costs = 1).
    Returns (plan, gen_count, exp_count).
    """
    frontier = [(0, start, [])]
    best_cost = {start: 0}
    generated = expanded = 0

    while frontier:
        cost, state, path = heapq.heappop(frontier)
        if cost > best_cost.get(state, float('inf')):
            continue
        expanded += 1
        if goal(state):
            return path, generated, expanded
        for action, nxt in successors(state, grid, rows, cols):
            generated += 1
            ncost = cost + 1
            if ncost < best_cost.get(nxt, float('inf')):
                best_cost[nxt] = ncost
                heapq.heappush(frontier, (ncost, nxt, path + [action]))
    return None, generated, expanded


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 planner.py [uniform-cost|depth-first] <world-file>")
        sys.exit(1)
    algo, world_file = sys.argv[1], sys.argv[2]
    cols, rows, grid = load_world(world_file)
    start = find_start_and_dirt(grid, rows, cols)

    if algo == 'depth-first':
        plan, gen, exp = depth_first(start, grid, rows, cols)
    elif algo == 'uniform-cost':
        plan, gen, exp = uniform_cost(start, grid, rows, cols)
    else:
        print(f"Unknown algorithm '{algo}'")
        sys.exit(1)

    if plan is None:
        print("No solution")
    else:
        for a in plan:
            print(a)
        print(f"{gen} nodes generated")
        print(f"{exp} nodes expanded")

if __name__ == '__main__':
    main()
