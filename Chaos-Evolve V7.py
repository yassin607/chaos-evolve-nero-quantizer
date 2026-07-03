import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Chaos-Evolve V6 — Ultimate Chaos Edition (UCE)
# Core improvements in V6:
# 1. Quantization + Pruning (compression and pruning combined)
# 2. Moving Obstacle (the obstacle moves sinusoidally, adding a time factor)
# 3. Target Lockdown (episode ends immediately upon reaching the target)
# ============================================================

# === 1. Neural Quantizer & Pruner Utility ===
class NeroQuantizerCore:
    def __init__(self):
        self.qmin, self.qmax = -8, 7

    def quantize_and_prune(self, W, block_size, prune_mask):
        # Apply genetic pruning first (zero out the pruned neurons)
        W_pruned = W * prune_mask

        if block_size is None or block_size <= 0:
            return W_pruned

        orig_shape = W_pruned.shape
        W_flat = W_pruned.flatten()

        remainder = len(W_flat) % block_size
        if remainder != 0:
            padding = block_size - remainder
            W_flat = np.concatenate([W_flat, np.zeros(padding)])
        else:
            padding = 0

        num_blocks = len(W_flat) // block_size
        W_blocks = W_flat.reshape(num_blocks, block_size)

        b_min = np.min(W_blocks, axis=1, keepdims=True)
        b_max = np.max(W_blocks, axis=1, keepdims=True)
        scales = (b_max - b_min) / (self.qmax - self.qmin)
        scales = np.where(scales == 0, 1.0, scales)
        zero_points = np.clip(np.round(-b_min / scales) + self.qmin, self.qmin, self.qmax).astype(np.int8)

        q_blocks = np.clip(np.round(W_blocks / scales) + zero_points, self.qmin, self.qmax).astype(np.int8)
        dq_flat = ((q_blocks.astype(np.float32) - zero_points) * scales).flatten()

        if padding > 0:
            dq_flat = dq_flat[:-padding]

        return dq_flat.reshape(orig_shape)


# === 2. Quantization & Pruning Aware Robot Brain ===
class RobotBrainV6:
    def __init__(self, block_size=None, prune_mask=None):
        self.W1 = np.random.randn(4, 12) * 0.5
        self.b1 = np.zeros((1, 12))
        self.W2 = np.random.randn(12, 2) * 0.5
        self.b2 = np.zeros((1, 2))
        self.fitness = 0

        self.block_size = block_size if block_size is not None else np.random.choice([2, 4, 8, 16])

        # New pruning gene: a mask vector that zeroes out some neurons in the hidden layer (12 neurons)
        if prune_mask is not None:
            self.prune_mask = prune_mask
        else:
            # Default: 100% of neurons are active (all ones)
            self.prune_mask = np.ones((12,))

        self.quantizer = NeroQuantizerCore()

    def forward(self, X, simulate_qat=True):
        if simulate_qat:
            # Build 2D masks matching the shapes of the first and second layer weights
            mask_W1 = self.prune_mask.reshape(1, 12)
            mask_W2 = self.prune_mask.reshape(12, 1)

            W1_eval = self.quantizer.quantize_and_prune(self.W1, self.block_size, mask_W1)
            b1_eval = self.quantizer.quantize_and_prune(self.b1, self.block_size, mask_W1)
            W2_eval = self.quantizer.quantize_and_prune(self.W2, self.block_size, mask_W2)
            b2_eval = self.b2  # Output layer is left as-is
        else:
            mask_W1 = self.prune_mask.reshape(1, 12)
            mask_W2 = self.prune_mask.reshape(12, 1)
            W1_eval, b1_eval, W2_eval, b2_eval = self.W1 * mask_W1, self.b1 * mask_W1, self.W2 * mask_W2, self.b2

        return np.dot(np.maximum(0, np.dot(X, W1_eval) + b1_eval), W2_eval) + b2_eval

    def mutate(self, rate=0.35, scale=0.3):
        if np.random.rand() < rate:
            self.W1 += np.random.randn(*self.W1.shape) * scale
            self.b1 += np.random.randn(*self.b1.shape) * scale
            self.W2 += np.random.randn(*self.W2.shape) * scale
            self.b2 += np.random.randn(*self.b2.shape) * scale

        if np.random.rand() < 0.15:
            self.block_size = np.random.choice([2, 4, 8, 16])

        # Pruning gene mutation: flip the state of a random neuron (from 1 to 0 or vice versa)
        if np.random.rand() < 0.20:
            idx = np.random.randint(0, 12)
            self.prune_mask[idx] = 1.0 - self.prune_mask[idx]


# === 3. Dynamic Obstacle Environment ===
class DynamicObstacleEnv:
    def __init__(self, fixed_seed=None):
        self.fixed_seed = fixed_seed
        self.reset()

    def reset(self):
        if self.fixed_seed is not None:
            rng = np.random.RandomState(self.fixed_seed)
            self.agent_pos = rng.uniform(-9, -5, (1, 2))
            self.target_pos = rng.uniform(5, 9, (1, 2))
            self.base_obstacle_y = rng.uniform(-2, 2)
        else:
            self.agent_pos = np.random.uniform(-9, -5, (1, 2))
            self.target_pos = np.random.uniform(5, 9, (1, 2))
            self.base_obstacle_y = np.random.uniform(-2, 2)

        self.steps_taken = 0
        self.obstacle_pos = np.array([[0.0, self.base_obstacle_y]])
        self.prev_distance = np.linalg.norm(self.target_pos - self.agent_pos)
        return self.get_state()

    def get_state(self):
        return np.hstack((self.target_pos - self.agent_pos,
                          self.obstacle_pos - self.agent_pos))

    def step(self, action):
        self.steps_taken += 1

        # Improvement 3: sinusoidal motion for the black obstacle, driven by time (step count)
        dynamic_y = self.base_obstacle_y + 3.5 * np.sin(self.steps_taken * 0.4)
        self.obstacle_pos[0, 1] = dynamic_y

        move = np.clip(action, -1.2, 1.2)
        next_pos = self.agent_pos + move

        hit_obstacle = False
        crossed_x = (self.agent_pos[0, 0] < 0 and next_pos[0, 0] >= 0) or \
                    (self.agent_pos[0, 0] > 0 and next_pos[0, 0] <= 0)
        if crossed_x and abs(next_pos[0, 1] - self.obstacle_pos[0, 1]) < 3.0:
            hit_obstacle = True

        if not hit_obstacle:
            self.agent_pos = next_pos

        distance = np.linalg.norm(self.target_pos - self.agent_pos)

        if hit_obstacle:
            return self.get_state(), 30.0, True

        # Improvement 1: perfect landing - stop immediately upon reaching the target
        # to prevent spiral looping
        if distance < 0.3:
            return self.get_state(), 0.0, True

        progress = self.prev_distance - distance
        adjusted_distance = distance - (progress * 0.5)
        self.prev_distance = distance

        done = self.steps_taken >= 60
        return self.get_state(), adjusted_distance, done


# === 4. Brain Evaluation with Pruning Bonus ===
def evaluate_brain(brain, env, episodes=5, simulate_qat=True):
    total_dist = 0
    crossed_wall = False

    for _ in range(episodes):
        state = env.reset()
        start_x = env.agent_pos[0, 0]
        done = False

        while not done:
            state, distance, done = env.step(brain.forward(state, simulate_qat=simulate_qat))
            if env.agent_pos[0, 0] > 0 and start_x < 0:
                crossed_wall = True
            total_dist += distance

    avg_dist = total_dist / (episodes * 20)  # Calibrated by the approximate step count
    base_fitness = 1000.0 / (avg_dist + 0.001)
    if crossed_wall:
        base_fitness *= 2.5

    # Improvement 2: pruning and compression bonus
    # The fewer active neurons and the larger the block size, the higher the genetic bonus!
    active_neurons = np.sum(brain.prune_mask)
    prune_rate = (12 - active_neurons) / 12.0  # Fraction of neurons switched off

    compression_bonus = 1.0 + (brain.block_size / 32.0) + (prune_rate * 0.5)
    brain.fitness = base_fitness * compression_bonus
    return brain.fitness


def evaluate_fixed(brain, seeds, simulate_qat=True):
    total_dist = 0
    crashes = 0

    for seed in seeds:
        fixed_env = DynamicObstacleEnv(fixed_seed=seed)
        state = fixed_env.reset()
        done, ep_dist, steps = False, 0, 0
        crashed = False

        while not done:
            state, distance, done = fixed_env.step(brain.forward(state, simulate_qat=simulate_qat))
            ep_dist += distance
            steps += 1
            if distance >= 30.0:
                crashed = True

        total_dist += ep_dist / (steps if steps > 0 else 1)
        if crashed:
            crashes += 1

    avg_dist = total_dist / len(seeds)
    fitness = 1000.0 / (avg_dist + 0.001)
    return fitness, avg_dist, crashes


# === 5. Final Dynamic Journey Plot ===
def visualize_dynamic_journey(champion, seed=45):
    env = DynamicObstacleEnv(fixed_seed=seed)
    state = env.reset()

    robot_x, robot_y = [env.agent_pos[0, 0]], [env.agent_pos[0, 1]]
    target_x, target_y = env.target_pos[0, 0], env.target_pos[0, 1]

    obs_history_x = []
    obs_history_ymin = []
    obs_history_ymax = []

    done = False
    while not done:
        obs_history_x.append(env.obstacle_pos[0, 0])
        obs_history_ymin.append(env.obstacle_pos[0, 1] - 3.0)
        obs_history_ymax.append(env.obstacle_pos[0, 1] + 3.0)

        action = champion.forward(state, simulate_qat=True)
        state, _, done = env.step(action)
        robot_x.append(env.agent_pos[0, 0])
        robot_y.append(env.agent_pos[0, 1])

    plt.figure(figsize=(10, 6))
    plt.plot(robot_x, robot_y, '-o', label='Robot Path (INT4 QAT + Pruned)', color='blue')
    plt.scatter([robot_x[0]], [robot_y[0]], color='green', s=150, label='Start', zorder=5)
    plt.scatter([target_x], [target_y], color='red', marker='*', s=200, label='Target', zorder=5)

    # Plot the full motion range of the moving obstacle to show how the robot dodged the dynamic wall
    for i in range(0, len(obs_history_x), max(1, len(obs_history_x)//7)):
        plt.vlines(x=obs_history_x[i], ymin=obs_history_ymin[i], ymax=obs_history_ymax[i], colors='black', alpha=0.15, linewidth=4)
    plt.vlines(x=obs_history_x[-1], ymin=obs_history_ymin[-1], ymax=obs_history_ymax[-1], colors='black', linewidth=6, label='Final Obstacle Wall Position')

    active_neurons = int(np.sum(champion.prune_mask))
    plt.title(f"Chaos-Evolve V6: Dynamic Dodge! (Block={champion.block_size} | Active Neurons={active_neurons}/12)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.show()


# === 6. Main Breeding Loop ===
if __name__ == "__main__":
    train_env = DynamicObstacleEnv()
    pop_size = 120
    generations = 100
    pop = [RobotBrainV6() for _ in range(pop_size)]

    print("Phase 1: Breeding the Ultimate Chaos Overlord (V6)...")
    print("=" * 75)

    for g in range(1, generations + 1):
        for brain in pop:
            evaluate_brain(brain, train_env, simulate_qat=True)

        pop.sort(key=lambda x: x.fitness, reverse=True)
        elites = pop[:18]
        new_pop = list(elites)

        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(elites, size=2, replace=False)
            child_block = p1.block_size if np.random.rand() < 0.5 else p2.block_size

            # Cross the pruning gene with equal-chance inheritance
            child_mask = p1.prune_mask.copy() if np.random.rand() < 0.5 else p2.prune_mask.copy()

            child = RobotBrainV6(block_size=child_block, prune_mask=child_mask)
            child.W1 = p1.W1.copy() if np.random.rand() < 0.5 else p2.W1.copy()
            child.b1 = p1.b1.copy() if np.random.rand() < 0.5 else p2.b1.copy()
            child.W2 = p1.W2.copy() if np.random.rand() < 0.5 else p2.W2.copy()
            child.b2 = p1.b2.copy() if np.random.rand() < 0.5 else p2.b2.copy()

            child.mutate(rate=0.22, scale=0.09)
            new_pop.append(child)

        pop = new_pop

        if g % 25 == 0 or g == 1:
            best_brain = pop[0]
            act = int(np.sum(best_brain.prune_mask))
            print(f"Gen {g:3d} | Best Fitness: {best_brain.fitness:8.2f} | Trait: block={best_brain.block_size}, Active Neurons: {act}/12")

    champion = pop[0]
    eval_seeds = list(range(42, 62))

    print("\nPhase 2: Final Validation of the Ultimate V6 Champion")
    print("=" * 75)

    fit_fp, dist_fp, crash_fp = evaluate_fixed(champion, eval_seeds, simulate_qat=False)
    fit_q, dist_q, crash_q = evaluate_fixed(champion, eval_seeds, simulate_qat=True)

    print(f"{'Configuration':<25} | {'Avg Distance':^14} | {'Crashes':^10} | {'Fitness':^10}")
    print("-" * 70)
    print(f"{'FP32 Pruned Mode':<25} | {dist_fp:^14.2f} | {crash_fp:^10} | {fit_fp:^10.2f}")
    print(f"{'INT4 QAT + Pruned Mode':<25} | {dist_q:^14.2f} | {crash_q:^10} | {fit_q:^10.2f}")

    retention = (fit_q / fit_fp) * 100
    print("-" * 70)
    print(f"Optimal Brain Spec -> Block Size: {champion.block_size} | Active Neurons: {int(np.sum(champion.prune_mask))}/12")
    print(f"Intelligence Retention Rate under Ultimate Stress: {retention:.2f}%")

    visualize_dynamic_journey(champion, seed=45)