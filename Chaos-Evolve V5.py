import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# Chaos-Evolve V5 — All Fixes Edition
# Fix 1: Per-weight mutation (not all-or-nothing)
# Fix 2: Real crossover (per-layer alpha)
# Fix 3: Bigger network 4 -> 24 -> 12 -> 2
# Fix 4: Population diversity via fitness sharing
# Fix 5: Obstacles at different positions each episode
# Fix 6: Path visualization to see what the robot learned
# ============================================================


# === 1. Neural Network Brain (Bigger + Better Mutation) ===
class RobotBrain:
    def __init__(self):
        # Fix 3: Bigger network — two hidden layers
        self.W1 = np.random.randn(4, 24) * 0.5
        self.b1 = np.zeros((1, 24))
        self.W2 = np.random.randn(24, 12) * 0.5
        self.b2 = np.zeros((1, 12))
        self.W3 = np.random.randn(12, 2) * 0.5
        self.b3 = np.zeros((1, 2))
        self.fitness = 0

    def forward(self, X):
        a1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        a2 = np.maximum(0, np.dot(a1, self.W2) + self.b2)
        return np.dot(a2, self.W3) + self.b3

    def mutate(self, rate=0.15, scale=0.1):
        # Fix 1: Per-weight mutation — each weight mutates independently
        def mutate_matrix(W):
            mask = np.random.rand(*W.shape) < rate
            W[mask] += np.random.randn(np.sum(mask)) * scale
            return W

        self.W1 = mutate_matrix(self.W1)
        self.b1 = mutate_matrix(self.b1)
        self.W2 = mutate_matrix(self.W2)
        self.b2 = mutate_matrix(self.b2)
        self.W3 = mutate_matrix(self.W3)
        self.b3 = mutate_matrix(self.b3)

    def copy_weights_from(self, other):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


# === 2. Crossover (Fix 2: Real per-layer alpha) ===
def crossover(parent1, parent2):
    child = RobotBrain()
    # Fix 2: Each layer gets its own independent alpha
    for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        alpha = np.random.rand()
        p1_w = getattr(parent1, attr)
        p2_w = getattr(parent2, attr)
        setattr(child, attr, alpha * p1_w + (1 - alpha) * p2_w)
    return child


# === 3. Obstacle Environment (Fix 5: Multiple obstacle positions) ===
class ObstacleEnv:
    def __init__(self, fixed_seed=None, num_obstacles=1):
        self.fixed_seed = fixed_seed
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        rng = np.random.RandomState(self.fixed_seed) if self.fixed_seed is not None \
              else np.random

        self.agent_pos = rng.uniform(-9, -5, (1, 2))
        self.target_pos = rng.uniform(5, 9, (1, 2))

        # Fix 5: Multiple obstacles at different X positions, not just X=0
        obstacle_xs = np.linspace(-3, 3, self.num_obstacles)
        self.obstacles = [
            np.array([[x, rng.uniform(-4, 4)]]) for x in obstacle_xs
        ]

        self.steps_taken = 0
        self.prev_distance = np.linalg.norm(self.target_pos - self.agent_pos)
        self.path = [self.agent_pos.copy()]
        return self.get_state()

    def get_state(self):
        # Use closest obstacle for state input
        closest = min(self.obstacles,
                      key=lambda o: np.linalg.norm(o - self.agent_pos))
        return np.hstack((self.target_pos - self.agent_pos,
                          closest - self.agent_pos))

    def step(self, action):
        move = np.clip(action, -1.2, 1.2)
        next_pos = self.agent_pos + move

        # Check collision with all obstacles
        hit_obstacle = False
        for obs in self.obstacles:
            crossed_x = (self.agent_pos[0, 0] < obs[0, 0] and next_pos[0, 0] >= obs[0, 0]) or \
                        (self.agent_pos[0, 0] > obs[0, 0] and next_pos[0, 0] <= obs[0, 0])
            if crossed_x and abs(next_pos[0, 1] - obs[0, 1]) < 2.5:
                hit_obstacle = True
                break

        if not hit_obstacle:
            self.agent_pos = next_pos

        self.steps_taken += 1
        self.path.append(self.agent_pos.copy())

        distance = np.linalg.norm(self.target_pos - self.agent_pos)

        if hit_obstacle:
            return self.get_state(), 30.0, True

        # Progressive reward
        progress = self.prev_distance - distance
        adjusted_distance = distance - (progress * 0.5)
        self.prev_distance = distance

        done = self.steps_taken >= 60 or distance < 0.3
        return self.get_state(), adjusted_distance, done


# === 4. Brain Evaluation ===
def evaluate_brain(brain, episodes=6, num_obstacles=1):
    total_dist = 0
    crossed_wall = False
    env = ObstacleEnv(num_obstacles=num_obstacles)

    for _ in range(episodes):
        state = env.reset()
        start_x = env.agent_pos[0, 0]
        done, ep_dist, steps = False, 0, 0

        while not done:
            state, distance, done = env.step(brain.forward(state))
            ep_dist += distance
            steps += 1
            if env.agent_pos[0, 0] > 0 and start_x < 0:
                crossed_wall = True

        total_dist += ep_dist / steps

    base_fitness = 1000.0 / ((total_dist / episodes) + 0.001)
    brain.fitness = base_fitness * 2.5 if crossed_wall else base_fitness
    return brain.fitness


# === 5. Fixed Evaluation for Fair Comparison ===
def evaluate_fixed(brain, seeds, num_obstacles=1):
    total_dist = 0
    crashes = 0

    for seed in seeds:
        env = ObstacleEnv(fixed_seed=seed, num_obstacles=num_obstacles)
        state = env.reset()
        done, ep_dist, steps = False, 0, 0
        crashed = False

        while not done:
            state, distance, done = env.step(brain.forward(state))
            ep_dist += distance
            steps += 1
            if distance >= 30.0:
                crashed = True

        total_dist += ep_dist / steps
        if crashed:
            crashes += 1

    avg_dist = total_dist / len(seeds)
    fitness = 1000.0 / (avg_dist + 0.001)
    return fitness, avg_dist, crashes


# === 6. Fitness Sharing (Fix 4: Population Diversity) ===
def apply_fitness_sharing(population, sigma=1.5):
    """
    Penalize brains that are too similar to each other.
    This forces the population to stay diverse and explore more solutions.
    Similarity is measured by comparing W1 weights (fingerprint of the brain).
    """
    fingerprints = [b.W1.flatten() for b in population]
    for i, brain in enumerate(population):
        niche_count = 0
        for j, other in enumerate(population):
            if i == j:
                continue
            dist = np.linalg.norm(fingerprints[i] - fingerprints[j])
            if dist < sigma:
                niche_count += 1 - (dist / sigma)
        if niche_count > 0:
            brain.fitness /= (1 + niche_count * 0.05)


# === 7. Nero-Quantizer Core (INT4 Block-wise) ===
class NeroQuantizerCore:
    def __init__(self):
        self.qmin, self.qmax = -8, 7

    def quantize_tensor(self, W, block_size=8):
        orig_shape = W.shape
        W_flat = W.flatten()
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
        zero_points = np.clip(
            np.round(-b_min / scales) + self.qmin,
            self.qmin, self.qmax
        ).astype(np.int8)
        q_blocks = np.clip(
            np.round(W_blocks / scales) + zero_points,
            self.qmin, self.qmax
        ).astype(np.int8)
        dq_flat = ((q_blocks.astype(np.float32) - zero_points) * scales).flatten()
        if padding > 0:
            dq_flat = dq_flat[:-padding]
        return dq_flat.reshape(orig_shape)

    def compress_brain(self, brain, block_size=8):
        compressed = RobotBrain()
        compressed.W1 = self.quantize_tensor(brain.W1, block_size)
        compressed.b1 = self.quantize_tensor(brain.b1, block_size)
        compressed.W2 = self.quantize_tensor(brain.W2, block_size)
        compressed.b2 = self.quantize_tensor(brain.b2, block_size)
        compressed.W3 = self.quantize_tensor(brain.W3, block_size)
        compressed.b3 = self.quantize_tensor(brain.b3, block_size)
        return compressed


# === 8. Path Visualization (Fix 6) ===
def visualize_paths(fp32_brain, int4_brain, seeds, num_obstacles=1):
    """
    Show the actual path the robot takes in 4 fixed environments.
    Blue = FP32, Orange = INT4, Red X = obstacle, Green star = target.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle("Robot Path: FP32 (blue) vs INT4 (orange)", fontsize=14)

    for idx, seed in enumerate(seeds[:4]):
        ax = axes[idx]
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title(f"Environment seed={seed}")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_facecolor('#f8f8f8')

        for brain, color, label in [(fp32_brain, 'steelblue', 'FP32'),
                                     (int4_brain, 'darkorange', 'INT4')]:
            env = ObstacleEnv(fixed_seed=seed, num_obstacles=num_obstacles)
            state = env.reset()
            done = False

            # Draw start, target, obstacles
            ax.plot(env.agent_pos[0, 0], env.agent_pos[0, 1],
                    's', color=color, markersize=8)
            ax.plot(env.target_pos[0, 0], env.target_pos[0, 1],
                    '*', color='green', markersize=12)
            for obs in env.obstacles:
                ax.add_patch(patches.Rectangle(
                    (obs[0, 0] - 0.2, obs[0, 1] - 2.5), 0.4, 5.0,
                    color='red', alpha=0.3
                ))

            while not done:
                state, _, done = env.step(brain.forward(state))

            path = np.array([p[0] for p in env.path])
            ax.plot(path[:, 0], path[:, 1], '-o', color=color,
                    markersize=2, linewidth=1.5, label=label, alpha=0.8)

        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('robot_paths.png', dpi=120, bbox_inches='tight')
    print("\nPath visualization saved to: robot_paths.png")
    plt.show()


# === 9. Main ===
if __name__ == "__main__":
    pop_size = 120
    generations = 120
    num_obstacles = 2  # Fix 5: Two obstacles now
    pop = [RobotBrain() for _ in range(pop_size)]
    fitness_history = []

    print("Chaos-Evolve V5 — All Fixes Active")
    print(f"Network: 4 -> 24 -> 12 -> 2 | Obstacles: {num_obstacles} | Pop: {pop_size}")
    print("=" * 60)

    # --- Phase 1: Genetic Training ---
    print("Phase 1: Breeding the Genetic Overlord...")

    for g in range(1, generations + 1):
        for brain in pop:
            evaluate_brain(brain, episodes=6, num_obstacles=num_obstacles)

        # Fix 4: Apply fitness sharing before selection
        apply_fitness_sharing(pop, sigma=1.5)

        pop.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness = pop[0].fitness
        fitness_history.append(best_fitness)

        elite_count = int(pop_size * 0.15)
        elites = pop[:elite_count]
        new_pop = []

        # Keep elites unchanged
        for e in elites:
            new_pop.append(e)

        # Fill rest with crossover + mutation
        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(elites, size=2, replace=False)
            child = crossover(p1, p2)       # Fix 2: Real crossover
            child.mutate(rate=0.15, scale=0.1)  # Fix 1: Per-weight mutation
            new_pop.append(child)

        pop = new_pop

        if g % 20 == 0 or g == 1:
            approx_dist = (1000.0 / best_fitness) - 0.001
            print(f"Gen {g:3d} | Best Fitness: {best_fitness:8.2f} | Approx Avg Distance: {approx_dist:.2f}")

    champion = pop[0]
    eval_seeds = list(range(42, 62))

    # --- Phase 2: Quantization Sensitivity Analysis ---
    print("\nPhase 2: Quantization Sensitivity Analysis...")
    print("=" * 60)

    fp32_fitness, fp32_dist, fp32_crashes = evaluate_fixed(
        champion, eval_seeds, num_obstacles=num_obstacles)
    print(f"FP32 Baseline -> Fitness: {fp32_fitness:.2f} | Avg Dist: {fp32_dist:.2f} | Crashes: {fp32_crashes}/20\n")

    block_sizes = [2, 4, 8, 16]
    best_block_size = 8
    best_retention = -1
    best_quantized = None
    quantizer = NeroQuantizerCore()

    print(f"{'Block Size':<12} | {'Fitness':^10} | {'Avg Dist':^10} | {'Crashes':^10} | {'Retention':^10} | Verdict")
    print("-" * 75)

    for bs in block_sizes:
        q_brain = quantizer.compress_brain(champion, block_size=bs)
        fit, dist, crashes = evaluate_fixed(q_brain, eval_seeds, num_obstacles=num_obstacles)
        retention = (fit / fp32_fitness) * 100

        if abs(retention - 100) < abs(best_retention - 100):
            best_retention = retention
            best_block_size = bs
            best_quantized = q_brain

        verdict = "✅ Best" if 90 <= retention <= 110 else ("⚠️  OK" if retention >= 75 else "❌ Poor")
        print(f"block={bs:<8} | {fit:^10.2f} | {dist:^10.2f} | {crashes:^10} | {retention:^9.1f}% | {verdict}")

    print(f"\n-> Best block size: block={best_block_size} (Retention: {best_retention:.1f}%)")

    # --- Phase 3: Final Benchmark ---
    print(f"\nPhase 3: Final Benchmark (block={best_block_size}, 20 Fixed Environments)...")
    print("=" * 60)

    class RandomAgent:
        def forward(self, X):
            return np.random.uniform(-1.2, 1.2, (1, 2))

    agents = {
        "FP32 Champion ": champion,
        "INT4 Quantized": best_quantized,
        "Random Blind  ": RandomAgent()
    }

    print(f"\n{'Agent':<20} | {'Avg Distance':^14} | {'Crashes':^10} | {'Fitness':^10}")
    print("-" * 65)
    for name, agent in agents.items():
        fit, dist, crashes = evaluate_fixed(agent, eval_seeds, num_obstacles=num_obstacles)
        print(f"{name:<20} | {dist:^14.2f} | {crashes:^10} | {fit:^10.2f}")

    # --- Final Report ---
    final_fit, _, _ = evaluate_fixed(best_quantized, eval_seeds, num_obstacles=num_obstacles)
    final_retention = (final_fit / fp32_fitness) * 100

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("-" * 60)
    print(f"FP32 Fitness (20 fixed envs) : {fp32_fitness:.2f}")
    print(f"Best INT4 Block Size         : block={best_block_size}")
    print(f"Intelligence Retention Rate  : {final_retention:.2f}%")

    if 90 <= final_retention <= 110:
        print("✅ Quantization SUCCESS — Nearly lossless compression!")
    elif final_retention > 110:
        print("⚠️  INT4 scored higher — network is robust to quantization noise.")
    elif final_retention >= 75:
        print("⚠️  Quantization OK — Minor intelligence loss.")
    else:
        print("❌ Quantization FAILED — Network too sensitive to INT4 compression.")
    print("=" * 60)

    # --- Phase 4: Path Visualization ---
    print("\nPhase 4: Generating Path Visualization...")
    visualize_paths(champion, best_quantized, eval_seeds, num_obstacles=num_obstacles)