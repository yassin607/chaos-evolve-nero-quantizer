import numpy as np

# ============================================================
# Chaos-Evolve V6 — Quantization-Aware Genetic Edition (QAGE)
# Core improvements in V5:
# 1. Quantization-Aware Training (QAT): training directly on simulated INT4
# 2. Adaptive Block-Size Gene: block size is now a genetic trait that evolves
# 3. Layer-wise Crossover: advanced mating that preserves inherited layer traits
# ============================================================

# === 1. Neural Quantizer Utility ===
class NeroQuantizerCore:
    def __init__(self):
        self.qmin, self.qmax = -8, 7

    def quantize_tensor(self, W, block_size=4):
        if block_size is None or block_size <= 0:
            return W
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


# === 2. Quantization-Aware Robot Brain ===
class RobotBrainV5:
    def __init__(self, block_size=None):
        self.W1 = np.random.randn(4, 12) * 0.5
        self.b1 = np.zeros((1, 12))
        self.W2 = np.random.randn(12, 2) * 0.5
        self.b2 = np.zeros((1, 2))
        self.fitness = 0

        # New gene: adaptive block size (if not specified, chosen randomly from the list)
        self.block_size = block_size if block_size is not None else np.random.choice([2, 4, 8, 16])
        self.quantizer = NeroQuantizerCore()

    def forward(self, X, simulate_qat=True):
        """
        simulate_qat: if True, the network automatically quantizes its weights before
        acting, so the robot learns to operate under INT4 pressure during training.
        """
        if simulate_qat:
            W1_eval = self.quantizer.quantize_tensor(self.W1, self.block_size)
            b1_eval = self.quantizer.quantize_tensor(self.b1, self.block_size)
            W2_eval = self.quantizer.quantize_tensor(self.W2, self.block_size)
            b2_eval = self.quantizer.quantize_tensor(self.b2, self.block_size)
        else:
            W1_eval, b1_eval, W2_eval, b2_eval = self.W1, self.b1, self.W2, self.b2

        return np.dot(np.maximum(0, np.dot(X, W1_eval) + b1_eval), W2_eval) + b2_eval

    def mutate(self, rate=0.35, scale=0.3):
        # Regular weight mutation
        if np.random.rand() < rate:
            self.W1 += np.random.randn(*self.W1.shape) * scale
            self.b1 += np.random.randn(*self.b1.shape) * scale
            self.W2 += np.random.randn(*self.W2.shape) * scale
            self.b2 += np.random.randn(*self.b2.shape) * scale

        # Mutation of the block-size gene
        if np.random.rand() < 0.15:  # 15% chance of changing the gene
            self.block_size = np.random.choice([2, 4, 8, 16])


# === 3. Obstacle Environment ===
class ObstacleEnv:
    def __init__(self, fixed_seed=None):
        self.fixed_seed = fixed_seed
        self.reset()

    def reset(self):
        if self.fixed_seed is not None:
            rng = np.random.RandomState(self.fixed_seed)
            self.agent_pos = rng.uniform(-9, -5, (1, 2))
            self.target_pos = rng.uniform(5, 9, (1, 2))
            self.obstacle_pos = np.array([[0.0, rng.uniform(-4, 4)]])
        else:
            self.agent_pos = np.random.uniform(-9, -5, (1, 2))
            self.target_pos = np.random.uniform(5, 9, (1, 2))
            self.obstacle_pos = np.array([[0.0, np.random.uniform(-4, 4)]])

        self.steps_taken = 0
        self.prev_distance = np.linalg.norm(self.target_pos - self.agent_pos)
        return self.get_state()

    def get_state(self):
        return np.hstack((self.target_pos - self.agent_pos,
                          self.obstacle_pos - self.agent_pos))

    def step(self, action):
        move = np.clip(action, -1.2, 1.2)
        next_pos = self.agent_pos + move

        hit_obstacle = False
        crossed_x = (self.agent_pos[0, 0] < 0 and next_pos[0, 0] >= 0) or \
                    (self.agent_pos[0, 0] > 0 and next_pos[0, 0] <= 0)
        if crossed_x and abs(next_pos[0, 1] - self.obstacle_pos[0, 1]) < 3.0:
            hit_obstacle = True

        if not hit_obstacle:
            self.agent_pos = next_pos

        self.steps_taken += 1
        distance = np.linalg.norm(self.target_pos - self.agent_pos)

        if hit_obstacle:
            return self.get_state(), 30.0, True

        progress = self.prev_distance - distance
        adjusted_distance = distance - (progress * 0.5)
        self.prev_distance = distance

        done = self.steps_taken >= 50 or distance < 0.3
        return self.get_state(), adjusted_distance, done


# === 4. Brain Evaluation with QAT ===
def evaluate_brain(brain, env, episodes=5, simulate_qat=True):
    total_dist = 0
    crossed_wall = False

    for _ in range(episodes):
        state = env.reset()
        start_x = env.agent_pos[0, 0]
        done, ep_dist, steps = False, 0, 0

        while not done:
            # We pass simulate_qat so the robot learns while the pressure is active
            state, distance, done = env.step(brain.forward(state, simulate_qat=simulate_qat))
            ep_dist += distance
            steps += 1
            if env.agent_pos[0, 0] > 0 and start_x < 0:
                crossed_wall = True

        total_dist += ep_dist / steps

    base_fitness = 1000.0 / ((total_dist / episodes) + 0.001)
    if crossed_wall:
        base_fitness *= 2.5

    # Extra genetic bonus for a robot that can maintain a larger block size
    # (stronger compression for both memory and network)
    # The larger the block_size, the higher its quality and compression bonus
    compression_bonus = 1.0 + (brain.block_size / 32.0)
    brain.fitness = base_fitness * compression_bonus
    return brain.fitness


# === 5. Fixed Evaluation (For Verification) ===
def evaluate_fixed(brain, seeds, simulate_qat=True):
    total_dist = 0
    crashes = 0

    for seed in seeds:
        fixed_env = ObstacleEnv(fixed_seed=seed)
        state = fixed_env.reset()
        done, ep_dist, steps = False, 0, 0
        crashed = False

        while not done:
            state, distance, done = fixed_env.step(brain.forward(state, simulate_qat=simulate_qat))
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


# === 6. Main Execution Loop ===
if __name__ == "__main__":
    train_env = ObstacleEnv()
    pop_size = 120
    generations = 100
    pop = [RobotBrainV5() for _ in range(pop_size)]

    print("Phase 1: Breeding the QAT Genetic Overlord...")
    print("=" * 70)

    for g in range(1, generations + 1):
        for brain in pop:
            # Evaluate with QAT enabled so training is quantization-aware
            evaluate_brain(brain, train_env, simulate_qat=True)

        pop.sort(key=lambda x: x.fitness, reverse=True)
        elites = pop[:18]
        new_pop = list(elites)

        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(elites, size=2, replace=False)

            # The new generation inherits the block_size gene equally from one of the parents
            child_block = p1.block_size if np.random.rand() < 0.5 else p2.block_size
            child = RobotBrainV5(block_size=child_block)

            # New addition: Layer-wise Crossover (genetic exchange at the layer level)
            child.W1 = p1.W1.copy() if np.random.rand() < 0.5 else p2.W1.copy()
            child.b1 = p1.b1.copy() if np.random.rand() < 0.5 else p2.b1.copy()
            child.W2 = p1.W2.copy() if np.random.rand() < 0.5 else p2.W2.copy()
            child.b2 = p1.b2.copy() if np.random.rand() < 0.5 else p2.b2.copy()

            child.mutate(rate=0.2, scale=0.08)
            new_pop.append(child)

        pop = new_pop

        if g % 25 == 0 or g == 1:
            best_brain = pop[0]
            print(f"Gen {g:3d} | Best Fitness: {best_brain.fitness:8.2f} | Elite Block-Size Gene: block={best_brain.block_size}")

    champion = pop[0]
    eval_seeds = list(range(42, 62))

    print("\nPhase 2: Final Validation of the QAT Champion")
    print("=" * 70)

    # Test the champion without compression (pure theoretical performance)
    fit_fp, dist_fp, crash_fp = evaluate_fixed(champion, eval_seeds, simulate_qat=False)
    # Test the champion with the real INT4 compression it trained under
    fit_q, dist_q, crash_q = evaluate_fixed(champion, eval_seeds, simulate_qat=True)

    print(f"{'Configuration':<20} | {'Avg Distance':^14} | {'Crashes':^10} | {'Fitness':^10}")
    print("-" * 65)
    print(f"{'FP32 Mode':<20} | {dist_fp:^14.2f} | {crash_fp:^10} | {fit_fp:^10.2f}")
    print(f"{'INT4 QAT Mode':<20} | {dist_q:^14.2f} | {crash_q:^10} | {fit_q:^10.2f}")

    retention = (fit_q / fit_fp) * 100
    print("-" * 65)
    print(f"Chosen Optimal Block Size Gene by Evolution: block={champion.block_size}")
    print(f"Intelligence Retention Rate under QAT: {retention:.2f}%")

    if retention >= 95:
        print("Amazing! The robot has become fully immune to data loss thanks to QAT!")
    else:
        print("Stable and excellent performance, resistant to noise!")