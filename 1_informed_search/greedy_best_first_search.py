import heapq

class Node:
    def __init__(self, state, parent=None, move=None, h_cost=0):
        self.state = state  # Current position (state) as a tuple, e.g., (x, y)
        self.parent = parent  # Parent node from which this node was generated
        self.move = move  # Move that led to this state (up, down, left, right)
        self.h_cost = h_cost  # Heuristic cost (Manhattan Distance)

    # Comparison function to order nodes by heuristic cost
    def __lt__(self, other):
        return self.h_cost < other.h_cost

    def generate_children(self, goal_state):
        children = []
        x, y = self.state  # Current position (state)
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up directions
        for dx, dy in moves:
            new_state = (x + dx, y + dy)  # New position after the move
            if 0 <= new_state[0] < 5 and 0 <= new_state[1] < 5:  # Grid bounds check
                # Calculate Manhattan Distance for the new state
                child = Node(
                    state=new_state,
                    parent=self,
                    move=(dx, dy),
                    h_cost=abs(new_state[0] - goal_state[0]) + abs(new_state[1] - goal_state[1])
                )
                children.append(child)
        return children

class GreedyBestFirstSearch:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state  # Initial position
        self.goal_state = goal_state  # Target position

    def solve(self):
        # Open list to store nodes to explore, starting with the initial node
        open_list = []
        # Closed set to track visited nodes (states)
        closed_set = set()
        
        # Create the start node and calculate its heuristic (Manhattan Distance)
        start_node = Node(self.start_state, h_cost=self.calculate_heuristic(self.start_state))
        
        # Add the start node to the open list
        heapq.heappush(open_list, start_node)

        while open_list:
            # Pop the node with the smallest heuristic value (closest to goal)
            current_node = heapq.heappop(open_list)
            
            # If the goal is reached, return the solution path
            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

            closed_set.add(current_node.state)  # Add current node to closed set

            # Generate children (possible moves) for the current node
            for child in current_node.generate_children(self.goal_state):
                if child.state not in closed_set:  # Avoid revisiting nodes
                    heapq.heappush(open_list, child)  # Add child to the open list

        return None  # No solution found

    def calculate_heuristic(self, state):
        # Calculate Manhattan Distance: sum of absolute differences in row and column
        x1, y1 = state
        x2, y2 = self.goal_state
        return abs(x1 - x2) + abs(y1 - y2)

    def trace_solution(self, node):
        # Trace back the solution from the goal node to the start node
        path = []
        while node:
            path.append(node.state)  # Add node state to the path
            node = node.parent  # Move to parent node
        return path[::-1]  # Reverse the path to get start-to-goal order

# Test GBFS with a 5x5 grid (adjusted to your requirement, can also work for 3x3 puzzle)
start = (0, 0)  # Start state
goal = (4, 4)   # Goal state
gbfs = GreedyBestFirstSearch(start, goal)
solution_path = gbfs.solve()  # Find the solution path

if solution_path:
    print("Solution Path:", solution_path)
else:
    print("No solution found.")
