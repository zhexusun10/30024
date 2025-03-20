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

    # 找到红蛙的初始位置
    red_pos = None
    for coord, state in board.items():
        if state == CellState.RED:
            red_pos = coord
            break
    
    if red_pos is None:
        return None  # 没有红蛙
    
    # 检查坐标是否在棋盘边界内（不允许循环）
    def is_valid_coord(r: int, c: int) -> bool:
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N
    
    # 获取目标坐标（不使用Coord类的自动循环功能）
    def get_target_coord(pos: Coord, dr: int, dc: int) -> tuple[int, int]:
        return pos.r + dr, pos.c + dc
    
    # 获取坐标的所有可能移动（包括连跳）
    def get_possible_moves(current_pos: Coord) -> list[tuple[Coord, list[Direction]]]:
        possible_moves = []
        
        # 允许的移动方向（不能往上）
        allowed_directions = [
            Direction.Down, Direction.DownLeft, Direction.DownRight,
            Direction.Left, Direction.Right
        ]
        
        # 处理单步移动（移动到相邻的荷叶）
        for direction in allowed_directions:
            # 计算目标位置（不使用自动循环）
            r, c = get_target_coord(current_pos, direction.r, direction.c)
            
            # 检查是否在棋盘边界内
            if not is_valid_coord(r, c):
                continue
            
            # 尝试创建目标位置的坐标
            try:
                next_pos = Coord(r, c)
            except ValueError:
                continue
                
            if next_pos in board and board[next_pos] == CellState.LILY_PAD:
                possible_moves.append((next_pos, [direction]))
        
        # 处理连续跳跃
        # 递归函数：从某个位置开始，找出所有可能的连续跳跃路径
        def find_jump_paths(pos, directions, visited):
            # 标记是否找到了下一步跳跃
            found_next_jump = False
            
            # 尝试各个方向的跳跃
            for direction in allowed_directions:
                # 计算蓝蛙的位置
                r1, c1 = get_target_coord(pos, direction.r, direction.c)
                
                # 检查是否在边界内
                if not is_valid_coord(r1, c1):
                    continue
                
                try:
                    blue_pos = Coord(r1, c1)
                except ValueError:
                    continue
                
                # 检查是否有蓝蛙
                if blue_pos not in board or board[blue_pos] != CellState.BLUE:
                    continue
                
                # 计算跳跃后的位置
                r2, c2 = get_target_coord(blue_pos, direction.r, direction.c)
                
                # 检查是否在边界内
                if not is_valid_coord(r2, c2):
                    continue
                
                try:
                    lily_pos = Coord(r2, c2)
                except ValueError:
                    continue
                
                # 检查跳跃后是否为荷叶，且未访问过
                if (lily_pos in board and 
                    board[lily_pos] == CellState.LILY_PAD and 
                    lily_pos not in visited):
                    
                    # 找到了一个有效跳跃
                    found_next_jump = True
                    
                    # 更新方向列表和访问记录
                    new_directions = directions.copy()
                    new_directions.append(direction)
                    
                    # 将每个有效的跳跃点添加到可能移动中（包括中间点）
                    possible_moves.append((lily_pos, new_directions))
                    
                    new_visited = visited.copy()
                    new_visited.add(lily_pos)
                    
                    # 递归寻找从新位置开始的连续跳跃
                    find_jump_paths(lily_pos, new_directions, new_visited)
            
            # 返回是否找到了下一步跳跃
            return found_next_jump
        
        # 从当前位置开始寻找所有可能的连续跳跃
        find_jump_paths(current_pos, [], {current_pos})
        
        return possible_moves
    
    # 改进的启发函数：考虑一步内能前进的最大行数
    def heuristic(pos: Coord) -> int:
        r = pos.r
        
        # 如果已经到达目标行，返回0
        if r == 7:
            return 0
        
        # 计算所有可能的移动
        possible_moves = get_possible_moves(pos)
        
        # 找出一步内能达到的最大行数
        max_row = r
        for next_pos, _ in possible_moves:
            max_row = max(max_row, next_pos.r)
        
        # 计算每步能前进的最大行数（至少为1，避免除以0）
        max_adv = max(max_row - r, 1)
        
        # 估计到达第7行需要的最少步数
        return math.ceil((7 - r) / max_adv)
    
    # 简单启发函数：直接使用距离终点的行数
    def simple_heuristic(pos: Coord) -> int:
        return 7 - pos.r
    
    # 缓存启发函数的结果，避免重复计算
    heuristic_cache = {}
    
    # 包装启发函数，加入缓存机制
    def cached_heuristic(pos: Coord) -> int:
        # 如果已经计算过，直接返回缓存结果
        if pos in heuristic_cache:
            return heuristic_cache[pos]
        
        # 计算启发值并缓存
        h_value = heuristic(pos)
        heuristic_cache[pos] = h_value
        return h_value
    
    # A*搜索算法的实现
    def a_star_search() -> list[MoveAction] | None:
        # 优先队列：(f值, 唯一id, 位置)
        open_list = [(simple_heuristic(red_pos), 0, red_pos)]
        # 唯一id计数器（用于优先队列中相同f值的比较）
        counter = 1
        # 每个位置的g值（从起点到当前位置的实际成本）
        g_values = {red_pos: 0}
        # 每个位置的前驱（位置和移动）
        came_from = {}
        # 已访问的位置集合
        closed_list = set()
        
        while open_list:
            # 取出f值最小的状态
            _, _, current_pos = heapq.heappop(open_list)
            
            # 如果是目标状态（到达第7行）
            if current_pos.r == 7:
                # 重建路径
                path = []
                pos = current_pos
                while pos in came_from:
                    prev_pos, move_action = came_from[pos]
                    path.append(move_action)
                    pos = prev_pos
                return list(reversed(path))
            
            # 如果已经访问过，跳过
            if current_pos in closed_list:
                continue
            
            # 标记为已访问
            closed_list.add(current_pos)
            
            # 获取所有可能的移动
            possible_moves = get_possible_moves(current_pos)
            
            # 遍历所有可能的移动
            for next_pos, directions in possible_moves:
                # 计算新的g值（增加1表示执行了一次移动）
                new_g = g_values[current_pos] + 1
                
                # 如果这个位置没有访问过，或者找到了更短的路径
                if next_pos not in g_values or new_g < g_values[next_pos]:
                    # 更新g值
                    g_values[next_pos] = new_g
                    # 计算f值：g值 + 启发值
                    f_value = new_g + simple_heuristic(next_pos)
                    # 添加到开启列表
                    heapq.heappush(open_list, (f_value, counter, next_pos))
                    counter += 1
                    # 记录前驱
                    move_action = MoveAction(current_pos, directions)
                    came_from[next_pos] = (current_pos, move_action)
        
        # 如果开启列表为空仍未找到目标，则无解
        return None
    
    # 执行A*搜索
    return a_star_search()

