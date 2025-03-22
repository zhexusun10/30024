# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

from .core import CellState, Coord, Direction, MoveAction, BOARD_N
from .utils import render_board
import heapq
from collections import defaultdict
import math


def search(
    board: dict[Coord, CellState]
) -> list[MoveAction] | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `CellState` instances which can be one of
            `CellState.RED`, `CellState.BLUE`, or `CellState.LILY_PAD`.
    
    Returns:
        A list of "move actions" as MoveAction instances, or `None` if no
        solution is possible.
    """

    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!
    print(render_board(board, ansi=True))

    # Find the initial position of the red frog
    red_pos = None
    for coord, state in board.items():
        if state == CellState.RED:
            red_pos = coord
            break
    
    if red_pos is None:
        return None  # No red frog found
    
    # Check if the coordinate is within the board boundaries (no looping allowed)
    def is_valid_coord(r: int, c: int) -> bool:
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N
    
    # Get target coordinate (without using Coord class's automatic looping feature)
    def get_target_coord(pos: Coord, dr: int, dc: int) -> tuple[int, int]:
        return pos.r + dr, pos.c + dc
    
    # Get all possible moves from a coordinate (including consecutive jumps)
    def get_possible_moves(current_pos: Coord) -> list[tuple[Coord, list[Direction]]]:
        possible_moves = []
        
        # Allowed movement directions (cannot move upward)
        allowed_directions = [
            Direction.Down, Direction.DownLeft, Direction.DownRight,
            Direction.Left, Direction.Right
        ]
        
        # Handle single-step moves (move to adjacent lily pad)
        for direction in allowed_directions:
            # Calculate target position (without automatic looping)
            r, c = get_target_coord(current_pos, direction.r, direction.c)
            
            # Check if within board boundaries
            if not is_valid_coord(r, c):
                continue
            
            # Try to create a coordinate for the target position
            try:
                next_pos = Coord(r, c)
            except ValueError:
                continue
                
            if next_pos in board and board[next_pos] == CellState.LILY_PAD:
                possible_moves.append((next_pos, [direction]))
        
        # Handle consecutive jumps
        # Recursive function: from a certain position, find all possible consecutive jump paths
        def find_jump_paths(pos, directions, visited):
        
            # Try jumps in each direction
            for direction in allowed_directions:
                # Calculate the blue frog's position
                r1, c1 = get_target_coord(pos, direction.r, direction.c)
                
                # Check if within boundaries
                if not is_valid_coord(r1, c1):
                    continue
                
                try:
                    blue_pos = Coord(r1, c1)
                except ValueError:
                    continue
                
                # Check if there is a blue frog
                if blue_pos not in board or board[blue_pos] != CellState.BLUE:
                    continue
                
                # Calculate the position after the jump
                r2, c2 = get_target_coord(blue_pos, direction.r, direction.c)
                
                # Check if within boundaries
                if not is_valid_coord(r2, c2):
                    continue
                
                try:
                    lily_pos = Coord(r2, c2)
                except ValueError:
                    continue
                
                # Check if the landing position is a lily pad and not visited before
                if (lily_pos in board and 
                    board[lily_pos] == CellState.LILY_PAD and 
                    lily_pos not in visited):
                    
                    # Update the direction list and visited record
                    new_directions = directions.copy()
                    new_directions.append(direction)
                    
                    # Add each valid jump point to possible moves (including intermediate points)
                    possible_moves.append((lily_pos, new_directions))
                    
                    new_visited = visited.copy()
                    new_visited.add(lily_pos)
                    
                    # Recursively find consecutive jumps from the new position
                    find_jump_paths(lily_pos, new_directions, new_visited)
            
            return 
        
        # Find all possible consecutive jumps starting from the current position
        find_jump_paths(current_pos, [], {current_pos})
        
        return possible_moves
    
    # Improved heuristic function: considers the maximum number of rows that can be advanced in one step
    def advanced_heuristic(pos: Coord) -> int:
        r = pos.r
        
        # If already at the target row, return 0
        if r == 7:
            return 0
        
        # Calculate all possible moves
        possible_moves = get_possible_moves(pos)
        
        # Find the maximum row that can be reached in one step
        max_row = r
        for next_pos, _ in possible_moves:
            max_row = max(max_row, next_pos.r)
        
        # Calculate the maximum number of rows that can be advanced per step (at least 1, to avoid division by zero)
        max_adv = max(max_row - r, 1)
        
        # Estimate the minimum number of steps needed to reach row 7
        return math.ceil((7 - r) / max_adv)
    
    # Implementation of A* search algorithm
    def a_star_search() -> list[MoveAction] | None:
        # Priority queue: (f-value, unique id, position)
        open_list = [(advanced_heuristic(red_pos), 0, red_pos)]
        # Unique id counter (for comparing items with the same f-value in the priority queue)
        counter = 1
        # g-value for each position (actual cost from start to current position)
        g_values = {red_pos: 0}
        # Predecessor for each position (position and move)
        came_from = {}
        # Set of visited positions
        closed_list = set()
        
        while open_list:
            # Extract the state with the smallest f-value
            _, _, current_pos = heapq.heappop(open_list)
            
            # If it's the goal state (reached row 7)
            if current_pos.r == 7:
                # Reconstruct the path
                path = []
                pos = current_pos
                while pos in came_from:
                    prev_pos, move_action = came_from[pos]
                    path.append(move_action)
                    pos = prev_pos
                return list(reversed(path))
            
            # If already visited, skip
            if current_pos in closed_list:
                continue
            
            # Mark as visited
            closed_list.add(current_pos)
            
            # Get all possible moves
            possible_moves = get_possible_moves(current_pos)
            
            # Iterate through all possible moves
            for next_pos, directions in possible_moves:
                # Calculate new g-value (add 1 to represent one move execution)
                new_g = g_values[current_pos] + 1
                
                # If this position hasn't been visited, or a shorter path to a position already in the g-value dictionary is found
                if next_pos not in g_values or new_g < g_values[next_pos]:
                    # Update g-value
                    g_values[next_pos] = new_g
                    # Calculate f-value: g-value + heuristic value
                    f_value = new_g + advanced_heuristic(next_pos)
                    # Add to open list
                    heapq.heappush(open_list, (f_value, counter, next_pos))
                    counter += 1
                    # Record predecessor
                    move_action = MoveAction(current_pos, directions)
                    came_from[next_pos] = (current_pos, move_action)
        
        # If the open list becomes empty without finding the goal, there is no solution
        return None
    
    # Execute A* search
    return a_star_search()

