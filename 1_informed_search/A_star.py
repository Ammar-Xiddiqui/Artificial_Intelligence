import heapq

class PuzzleNode:
    def __init__(self, state, parent=None, move=None, g_cost=0, h_cost=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def generate_children(self, goal_state):
        children = []
        zero_pos = self.state.index(0)  # Find the empty tile (0)
        row, col = divmod(zero_pos, 3)  # Convert 1D index to 2D position
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

        for move, (dr, dc) in moves.items():
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:  # Check bounds
                new_pos = new_row * 3 + new_col
                new_state = self.state[:]
                new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]
                h_cost = self.calculate_heuristic(new_state, goal_state)
                children.append(PuzzleNode(new_state, self, move, self.g_cost + 1, h_cost))
        return children

    @staticmethod
    def calculate_heuristic(state, goal_state):
        # Manhattan Distance heuristic
        distance = 0
        for i, tile in enumerate(state):
            if tile != 0:  # Skip the empty tile
                goal_pos = goal_state.index(tile)
                distance += abs(goal_pos // 3 - i // 3) + abs(goal_pos % 3 - i % 3)
        return distance

class AStarSolver:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state

    def is_solvable(self, state):
        inversions = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] > state[j] != 0:
                    inversions += 1
        return inversions % 2 == 0

    def solve(self):
        if not self.is_solvable(self.start_state):
            return "Unsolvable puzzle."

        open_list = []
        closed_set = set()
        start_node = PuzzleNode(self.start_state, h_cost=PuzzleNode.calculate_heuristic(self.start_state, self.goal_state))
        heapq.heappush(open_list, start_node)

        while open_list:
            current_node = heapq.heappop(open_list)

            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

            closed_set.add(tuple(current_node.state))

            for child in current_node.generate_children(self.goal_state):
                if tuple(child.state) in closed_set:
                    continue
                heapq.heappush(open_list, child)

        return "No solution found."

    def trace_solution(self, node):
        solution = []
        while node:
            solution.append(node.state)
            node = node.parent
        return solution[::-1]

# Example Usage
start = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # Start state
goal = [1, 2, 3, 4, 5, 6, 7, 0, 8]   # Goal state

solver = AStarSolver(start, goal)
solution = solver.solve()
if isinstance(solution, str):
    print(solution)
else:
    for step in solution:
        print(step)
