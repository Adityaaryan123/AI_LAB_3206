import sys
import heapq
import math
import itertools

class AssignmentProblem:
    def __init__(self, filename):
        self.costs = {}
        self.group_size = 1
        self.initial_items = set()
        self.target_items = set()
        self.assignments = {} 
        self.parse(filename)

    def parse(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            if parts[0] == 'C':
                self.costs[parts[1]] = int(parts[2])
            elif parts[0] == 'G':
                self.group_size = int(parts[1])
            elif parts[0] == 'A':
                task_id = parts[1]
                inputs = [int(x) for x in parts[2:-2] if x != '-1']
                outcome = int(parts[-2])
                food = parts[-1]
                self.assignments[task_id] = {'inputs': inputs, 'out': outcome, 'food': food}
            elif parts[0] == 'O':
                self.target_items = set([int(x) for x in parts[1:] if x != '-1'])
            elif parts[0].isdigit():
                self.initial_items = set([int(x) for x in parts if x != '-1'])

    def get_available_tasks(self, available_items, completed_tasks):
        available = []
        for t_id, data in self.assignments.items():
            if t_id not in completed_tasks:
                if all(req in available_items for req in data['inputs']):
                    available.append(t_id)
        return available

class State:
    def __init__(self, items, completed, cost, day, schedule_history):
        self.items = frozenset(items)
        self.completed = frozenset(completed)
        self.g = cost
        self.day = day
        self.schedule_history = schedule_history
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def a_star_scheduler(problem):
    start_state = State(problem.initial_items, set(), 0, 0, [])
    frontier = [start_state]
    explored = set()
    states_explored = 0

    while frontier:
        curr = heapq.heappop(frontier)
        states_explored += 1

        if len(curr.completed) == len(problem.assignments):
            return curr, states_explored

        if curr.completed in explored:
            continue
        explored.add(curr.completed)

        available = problem.get_available_tasks(curr.items, curr.completed)
        
        if available:
            # Generate all valid combinations of tasks up to the group size
            max_tasks_to_pick = min(len(available), problem.group_size)
            
            for r in range(1, max_tasks_to_pick + 1):
                for combo in itertools.combinations(available, r):
                    selected = list(combo)
                    
                    daily_menu = set([problem.assignments[t]['food'] for t in selected])
                    daily_cost = sum([problem.costs[f] for f in daily_menu])
                    
                    new_items = set(curr.items)
                    for t in selected:
                        new_items.add(problem.assignments[t]['out'])
                        
                    new_completed = set(curr.completed).union(selected)
                    new_history = curr.schedule_history + [(selected, daily_menu, daily_cost)]
                    
                    next_state = State(new_items, new_completed, curr.g + daily_cost, curr.day + 1, new_history)
                    
                    # Admissible Heuristic calculation
                    remaining_tasks = len(problem.assignments) - len(new_completed)
                    min_food_cost = min(problem.costs.values())
                    next_state.f = next_state.g + math.ceil(remaining_tasks / problem.group_size) * min_food_cost
                    
                    heapq.heappush(frontier, next_state)

    return None, states_explored

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python task2_astar.py <test_case.txt>")
        sys.exit(1)
        
    problem = AssignmentProblem(sys.argv[1])
    
    print("--- A* Search Optimal Schedule ---")
    optimal_state, explored_count = a_star_scheduler(problem)
    
    if optimal_state:
        for day, (tasks, menu, cost) in enumerate(optimal_state.schedule_history, 1):
             print(f"Day-{day}: {', '.join(['A'+t for t in tasks])} | Menu: {menu} | Cost: {cost}")
        print(f"Total Days Taken: {optimal_state.day}")
        print(f"Total Food Cost: {optimal_state.g}")
        print(f"Total States Explored: {explored_count}")
    else:
        print("No valid schedule found.")