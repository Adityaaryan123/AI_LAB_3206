1. PROJECT OVERVIEW

This project solves a complex scheduling and optimization problem. Given a set 
of course assignments with specific prerequisites, a maximum student group size, 
and varying daily food costs associated with each task, the goal is to schedule 
the assignments to minimize the total cost of food ordered across all days.

The project is divided into two distinct parts:
- Task 1 (task1_greedy.py): Implements fast, heuristic-based Greedy algorithms.
- Task 2 (task2_astar.py): Implements an A* Search algorithm to find the 
  mathematically optimal schedule with the absolute minimum food cost.

-----------------------------------------------------------------------------
2. PREREQUISITES & DEPENDENCIES
-----------------------------------------------------------------------------
- Python 3.x
- matplotlib (Required for generating comparison plots in the final report)
- numpy (Required for numerical operations and data handling)

To install the required external libraries, run the following command:
pip install matplotlib numpy

-----------------------------------------------------------------------------
3. FILE STRUCTURE
-----------------------------------------------------------------------------
- task1_greedy.py   : Contains the DAG parser and Greedy scheduling logic.
- task2_astar.py    : Contains the DAG parser and A* search scheduling logic.
- test1.txt         : Sample test case 1 (Linear/Bottleneck graph).
- test2.txt         : Sample test case 2 (Wide graph, high group size).
- test3.txt         : Sample test case 3 (Complex dependencies, mixed costs).
- README.txt        : This documentation file.

-----------------------------------------------------------------------------
4. HOW TO RUN THE PROGRAMS
-----------------------------------------------------------------------------
To run the Greedy algorithms (Task 1):
    python task1_greedy.py test1.txt

To run the Optimal A* Search (Task 2):
    python task2_astar.py test1.txt

The programs will output the day-by-day schedule, the daily menu, the daily 
cost, and the final totals directly to the console.

-----------------------------------------------------------------------------
5. IMPLEMENTED GREEDY STRATEGIES (TASK 1)
-----------------------------------------------------------------------------
The Greedy approach generates a schedule by selecting the "best" available 
assignments at each step based on a specific heuristic. Two strategies are 
implemented:

A. Greedy by Food Cost (Cost-Centric)
   - Logic: At the start of each day, all assignments whose prerequisites have 
     been met are identified. These available assignments are then sorted in 
     ascending order based on the cost of their required food item.
   - Justification: By aggressively prioritizing the cheapest tasks, this 
     strategy attempts to keep the daily menu cost as low as possible.

B. Greedy by Dependency Depth (Critical Path)
   - Logic: The graph is pre-processed to calculate the "depth" of each node 
     (the longest downstream path to a terminal node). Available assignments 
     are sorted in descending order by this depth score.
   - Justification: This strategy prioritizes clearing bottleneck tasks early. 
     By unlocking the highest number of downstream dependencies, it widens 
     the pool of available tasks for future days, allowing for more optimal 
     grouping of identical food items later.

-----------------------------------------------------------------------------
6. A* SEARCH IMPLEMENTATION (TASK 2)
-----------------------------------------------------------------------------
The A* search algorithm guarantees the optimal schedule (lowest possible total 
food cost) by evaluating states using the function f(n) = g(n) + h(n).

- State Representation: A state tracks the currently available inputs, the set 
  of completed assignments, the accumulated food cost, the current day, and the 
  schedule history.
- Path Cost g(n): The exact, accumulated food cost of all daily menus 
  generated from Day 1 up to the current state.
- Heuristic h(n): 
  Calculated as: ceil(Remaining Tasks / Group Size) * Minimum Food Cost.
  This heuristic evaluates the remaining unscheduled tasks and assumes we can 
  process them in perfect, full-capacity groups (g) using the cheapest possible 
  food item available in the entire problem space. Because it represents a 
  best-case, conflict-free scenario, it never overestimates the true remaining 
  cost, making it an *admissible* heuristic.


=============================================================================
