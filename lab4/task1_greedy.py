import sys

class AssignmentProblem:
    def __init__(self, filename):
        self.costs = {}
        self.group_size = 1
        self.initial_items = set()
        self.target_items = set()
        self.assignments = {} 
        self.parse(filename)
        self.calculate_depths()

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

    def calculate_depths(self):
        self.depths = {t_id: 0 for t_id in self.assignments}
        changed = True
        while changed:
            changed = False
            for t_id, data in self.assignments.items():
                max_downstream = 0
                for other_id, other_data in self.assignments.items():
                    if data['out'] in other_data['inputs']:
                        max_downstream = max(max_downstream, self.depths[other_id] + 1)
                if self.depths[t_id] != max_downstream:
                    self.depths[t_id] = max_downstream
                    changed = True

    def get_available_tasks(self, available_items, completed_tasks):
        available = []
        for t_id, data in self.assignments.items():
            if t_id not in completed_tasks:
                if all(req in available_items for req in data['inputs']):
                    available.append(t_id)
        return available

def greedy_scheduler(problem, strategy="cost"):
    available_items = set(problem.initial_items)
    completed_tasks = set()
    schedule = []
    total_cost = 0

    while len(completed_tasks) < len(problem.assignments):
        available = problem.get_available_tasks(available_items, completed_tasks)
        if not available:
            print("Warning: Graph deadlock or disjointed tasks.")
            break 

        if strategy == "cost":
            available.sort(key=lambda x: (problem.costs[problem.assignments[x]['food']], x))
        elif strategy == "depth":
            available.sort(key=lambda x: (-problem.depths[x], x))

        selected = available[:problem.group_size]
        
        daily_menu = set([problem.assignments[t]['food'] for t in selected])
        daily_cost = sum([problem.costs[f] for f in daily_menu])
        
        schedule.append((selected, daily_menu, daily_cost))
        total_cost += daily_cost
        
        completed_tasks.update(selected)
        for t in selected:
            available_items.add(problem.assignments[t]['out'])

    return schedule, total_cost

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python task1_greedy.py <test_case.txt>")
        sys.exit(1)
        
    problem = AssignmentProblem(sys.argv[1])
    
    print("--- Greedy Strategy: Food Cost ---")
    s_cost, t_cost = greedy_scheduler(problem, "cost")
    for day, (tasks, menu, cost) in enumerate(s_cost, 1):
        print(f"Day-{day}: {', '.join(['A'+t for t in tasks])} | Menu: {menu} | Cost: {cost}")
    print(f"Total Days: {len(s_cost)}")
    print(f"Total Cost: {t_cost}\n")
    
    print("--- Greedy Strategy: Dependency Depth ---")
    s_depth, t_depth = greedy_scheduler(problem, "depth")
    for day, (tasks, menu, cost) in enumerate(s_depth, 1):
         print(f"Day-{day}: {', '.join(['A'+t for t in tasks])} | Menu: {menu} | Cost: {cost}")
    print(f"Total Days: {len(s_depth)}")
    print(f"Total Cost: {t_depth}")