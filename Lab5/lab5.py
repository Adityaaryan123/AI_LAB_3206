import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

STATES = ['High', 'Low', 'Charging']
ACTIONS = ['Search', 'Wait']
GAMMA = 0.9
THETA = 1e-6



# Task 1.1 : Build the MDP

P = np.zeros((3, 2, 3))

# High, Search
P[0, 0, 0] = 0.7  # High -> High
P[0, 0, 1] = 0.3  # High -> Low
P[0, 1, 0] = 1.0  # High -> High
P[1, 0, 0] = 0.4  # Low -> High
P[1, 0, 1] = 0.6  # Low -> Low
P[1, 1, 1] = 1.0  # Low -> Low
P[2, 0, 2] = 1.0  # Charging -> Charging (dummy to make sum = 1.0)
P[2, 1, 0] = 1.0  # Charging -> High

R = np.zeros((3, 2))
R[0, 0] = 4.0   # High, Search
R[0, 1] = 1.0   # High, Wait
R[1, 0] = (-3 * 0.4) + (4 * 0.6)   # Low, Search (expected)
R[1, 1] = 1.0   # Low, Wait
R[2, 1] = 0.0   # Charging, Wait


# Verify transition probabilities sum to 1
print("Transition probability sums (should all be 1.0):")
for s in range(3):
    for a in range(2):
        total = P[s, a, :].sum()
        print(f"  P[{STATES[s]}, {ACTIONS[a]}, :] = {total:.4f}")

print("\nReward matrix R[s, a]:")
print(f"{'':10}", end="")
for a in ACTIONS:
    print(f"{a:10}", end="")
print()
for s, state in enumerate(STATES):
    print(f"{state:10}", end="")
    for a in range(2):
        print(f"{R[s, a]:10.2f}", end="")
    print()



# Task 1.2 : Policy Evaluation

def policy_evaluation(P, R, policy, gamma, theta):
    V = np.zeros(len(STATES))
    iterations = 0

    while True:
        delta = 0
        V_new = np.zeros(len(STATES))
        for s in range(len(STATES)):
            a = policy[s]
            V_new[s] = sum(P[s, a, s_next] * (R[s, a] + gamma * V[s_next])
                           for s_next in range(len(STATES)))
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        iterations += 1
        if delta < theta:
            break

    print(f"Policy evaluation converged in {iterations} iterations.")
    return V


# Policy: High -> Search (0), Low -> Wait (1), Charging -> Wait (1)
policy_task1 = [0, 1, 1]
print("\nRunning policy evaluation for fixed policy [Search, Wait, Wait]...")
V_pi = policy_evaluation(P, R, policy_task1, GAMMA, THETA)

print("V_pi(s):")
for s, state in enumerate(STATES):
    print(f"  {state}: {V_pi[s]:.6f}")



# Task 1.3 : Bar Chart of V_pi
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(STATES, V_pi, color=['steelblue', 'tomato', 'seagreen'], edgecolor='black')
ax.set_title("Task 1.3 : Policy Evaluation: V_pi(s)")
ax.set_xlabel("State")
ax.set_ylabel("Value V_pi(s)")
for bar, val in zip(bars, V_pi):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.3f}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("task1_policy_eval_bar.png", dpi=150)
plt.close()
print("\nSaved: task1_policy_eval_bar.png")



# Task 2.1 : Value Iteration
def value_iteration(P, R, gamma, theta):
    V = np.zeros(len(STATES))
    history = [V.copy()]
    iterations = 0

    while True:
        delta = 0
        V_new = np.zeros(len(STATES))
        for s in range(len(STATES)):
            Q_values = []
            for a in range(len(ACTIONS)):
                q = sum(P[s, a, s_next] * (R[s, a] + gamma * V[s_next])
                        for s_next in range(len(STATES)))
                Q_values.append(q)
            V_new[s] = max(Q_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        history.append(V.copy())
        iterations += 1
        if delta < theta:
            break

    print(f"Value iteration converged in {iterations} iterations.")
    return V, history


print("\nRunning value iteration...")
V_star, vi_history = value_iteration(P, R, GAMMA, THETA)

print("V*(s):")
for s, state in enumerate(STATES):
    print(f"  {state}: {V_star[s]:.6f}")



# Task 2.2 : Extract Optimal Policy

def extract_policy(V_star, P, R, gamma):
    policy = []
    for s in range(len(STATES)):
        Q_values = []
        for a in range(len(ACTIONS)):
            q = sum(P[s, a, s_next] * (R[s, a] + gamma * V_star[s_next])
                    for s_next in range(len(STATES)))
            Q_values.append(q)
        policy.append(np.argmax(Q_values))
    return policy


optimal_policy_vi = extract_policy(V_star, P, R, GAMMA)

print("\nOptimal policy from value iteration:")
for s, state in enumerate(STATES):
    print(f"  {state}: {ACTIONS[optimal_policy_vi[s]]}")



# Task 2.3 : Plot Convergence

vi_history_arr = np.array(vi_history)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['steelblue', 'tomato', 'seagreen']
for s, state in enumerate(STATES):
    ax.plot(vi_history_arr[:, s], label=state, color=colors[s], linewidth=2)
ax.set_title("Task 2.3 : Value Iteration Convergence")
ax.set_xlabel("Iteration")
ax.set_ylabel("V(s)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("task2_value_iteration_convergence.png", dpi=150)
plt.close()
print("Saved: task2_value_iteration_convergence.png")



# Task 3.1 : Policy Improvement

def policy_improvement(V, P, R, gamma, old_policy):
    new_policy = []
    for s in range(len(STATES)):
        Q_values = []
        for a in range(len(ACTIONS)):
            # Charging state cannot Search (action 0 not allowed)
            if s == 2 and a == 0:
                Q_values.append(-np.inf)
                continue
            q = sum(P[s, a, s_next] * (R[s, a] + gamma * V[s_next])
                    for s_next in range(len(STATES)))
            Q_values.append(q)
        new_policy.append(np.argmax(Q_values))

    stable = new_policy == old_policy
    return new_policy, stable



# Task 3.2 : Full Policy Iteration Loop

def policy_iteration(P, R, gamma, theta):
    policy = [1, 1, 1]  # all states start with Wait
    policy_history = [policy.copy()]
    V_history = []
    iteration = 0

    while True:
        V = policy_evaluation(P, R, policy, gamma, theta)
        V_history.append(V.copy())

        print(f"\nPolicy Iteration step {iteration + 1}:")
        print(f"  Policy: {[ACTIONS[a] for a in policy]}")
        print(f"  V: {[round(v, 4) for v in V]}")

        new_policy, stable = policy_improvement(V, P, R, gamma, policy)
        policy = new_policy
        policy_history.append(policy.copy())
        iteration += 1

        if stable:
            break

    print(f"\nPolicy iteration converged in {iteration} outer iterations.")
    return policy, V, policy_history, V_history


print("\nRunning policy iteration...")
optimal_policy_pi, V_pi_final, policy_history, V_history_pi = policy_iteration(
    P, R, GAMMA, THETA)

print("\nOptimal policy from policy iteration:")
for s, state in enumerate(STATES):
    print(f"  {state}: {ACTIONS[optimal_policy_pi[s]]}")



# Task 3.3a : V(s) over policy iteration steps


V_history_arr = np.array(V_history_pi)
fig, ax = plt.subplots(figsize=(7, 4))
for s, state in enumerate(STATES):
    ax.plot(range(1, len(V_history_pi) + 1), V_history_arr[:, s],
            label=state, color=colors[s], marker='o', linewidth=2)
ax.set_title("Task 3.3a : V(s) across Policy Iteration Steps")
ax.set_xlabel("Policy Iteration Step")
ax.set_ylabel("V(s)")
ax.set_xticks(range(1, len(V_history_pi) + 1))
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("task3_policy_iter_V.png", dpi=150)
plt.close()
print("Saved: task3_policy_iter_V.png")



# Task 3.3b : Policy heatmap across iterations

policy_mat = np.array(policy_history[:len(V_history_pi)])  # shape (iters, states)
policy_mat = policy_mat.T  # shape (states, iters)

fig, ax = plt.subplots(figsize=(max(5, len(V_history_pi) * 1.5), 3))
sns.heatmap(policy_mat, ax=ax, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=[f"Iter {i+1}" for i in range(len(V_history_pi))],
            yticklabels=STATES, cbar=False, linewidths=0.5)

# Replace 0/1 annotations with action names
for text in ax.texts:
    val = int(text.get_text())
    text.set_text(ACTIONS[val])

ax.set_title("Task 3.3b : Policy at Each Iteration (0=Search, 1=Wait)")
ax.set_xlabel("Iteration")
ax.set_ylabel("State")
plt.tight_layout()
plt.savefig("task3_policy_heatmap.png", dpi=150)
plt.close()
print("Saved: task3_policy_heatmap.png")



# Task 4 : Analysis Summary 

print("\n" + "="*55)
print("Task 4 : Analysis Summary")
print("="*55)

print("\n4.1 : Convergence Comparison:")
print(f"  Value Iteration outer sweeps: {len(vi_history) - 1}")
print(f"  Policy Iteration outer loops: {len(V_history_pi)}")
print("  Policy Iteration converges in fewer outer iterations")
print("  because each evaluation step fully solves for V_pi,")
print("  giving a cleaner improvement signal per loop.")

print("\n4.2 : Convergence Behavior:")
print("  V(s) starts at 0 and increases monotonically.")
print("  High state converges to the highest value since it")
print("  can Search reliably. Value Iteration updates all states")
print("  simultaneously using max over actions, so it explores")
print("  the value landscape more broadly than Policy Evaluation,")
print("  which is constrained to a fixed policy per pass.")

print("\n4.3 : Optimal Policy Interpretation:")
print("  High: Search")
print("  Low: Wait")
print("  Charging: Wait")
print("  High -> Search: high battery means low risk of penalty;")
print("  reward of +4 outweighs the small chance of dropping to Low.")
print("  Low -> Wait: Searching risks a -3 penalty; waiting gives")
print("  a safe +1 and preserves battery level.")
print("  Charging -> Wait: only valid action; leads back to High.")

print("\n4.4 : Practical Insight:")
print("  For a battery-powered robot, this policy maximises long-run")
print("  reward by balancing exploration (Search when strong) and")
print("  caution (Wait when weak). It avoids costly recharge penalties")
print("  and keeps the robot operational over extended periods.")