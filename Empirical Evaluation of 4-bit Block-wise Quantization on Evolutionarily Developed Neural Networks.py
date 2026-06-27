import numpy as np

# === 1. Genetic Engine and Environment (Chaos-Evolve V2) ===
class RobotBrainV2:
    def __init__(self):
        self.W1 = np.random.randn(4, 12) * 0.5
        self.b1 = np.zeros((1, 12))
        self.W2 = np.random.randn(12, 2) * 0.5
        self.b2 = np.zeros((1, 2))
        self.fitness = 0

    def forward(self, X):
        return np.dot(np.maximum(0, np.dot(X, self.W1) + self.b1), self.W2) + self.b2

    def mutate(self, rate=0.35, scale=0.3):
        if np.random.rand() < rate:
            self.W1 += np.random.randn(*self.W1.shape) * scale
            self.b1 += np.random.randn(*self.b1.shape) * scale
            self.W2 += np.random.randn(*self.W2.shape) * scale
            self.b2 += np.random.randn(*self.b2.shape) * scale


class ObstacleEnv:
    def __init__(self): self.reset()

    def reset(self):
        self.agent_pos = np.random.uniform(-9, -5, (1, 2))
        self.target_pos = np.random.uniform(5, 9, (1, 2))
        self.obstacle_pos = np.array([[0.0, np.random.uniform(-4, 4)]])
        self.steps_taken = 0
        return self.get_state()

    def get_state(self):
        return np.hstack((self.target_pos - self.agent_pos, self.obstacle_pos - self.agent_pos))

    def step(self, action):
        move = np.clip(action, -1.2, 1.2)
        next_pos = self.agent_pos + move
        hit_obstacle = False
        if (self.agent_pos[0, 0] < 0 and next_pos[0, 0] >= 0) or (self.agent_pos[0, 0] > 0 and next_pos[0, 0] <= 0):
            if abs(next_pos[0, 1] - self.obstacle_pos[0, 1]) < 3.0: hit_obstacle = True
        if not hit_obstacle: self.agent_pos = next_pos
        self.steps_taken += 1
        distance = np.linalg.norm(self.target_pos - self.agent_pos)
        if hit_obstacle: distance += 15.0
        return self.get_state(), distance, self.steps_taken >= 50 or distance < 0.3


def evaluate_brain(brain, env, episodes=5):
    total_generation_distance = 0
    crossed_wall = False
    for _ in range(episodes):
        state = env.reset()
        start_x = env.agent_pos[0, 0]
        done, total_distance, steps = False, 0, 0
        while not done:
            state, distance, done = env.step(brain.forward(state))
            total_distance += distance; steps += 1
            if env.agent_pos[0, 0] > 0 and start_x < 0: crossed_wall = True
        total_generation_distance += (total_distance / steps)
    base_fitness = 1000.0 / ((total_generation_distance / episodes) + 0.001)
    brain.fitness = base_fitness * 2.5 if crossed_wall else base_fitness
    return brain.fitness


# === 2. Compression Engine (Nero-Quantizer Core) ===
class NeroQuantizerCore:
    def __init__(self):
        self.qmin, self.qmax = -8, 7

    def quantize_tensor(self, W, block_size=4):
        """Compress a single weight matrix of the genetic champion using Block-wise quantization."""
        orig_shape = W.shape
        W_flat = W.flatten()

        # Pad if the size is not aligned with the block size
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

        zero_points = np.round(-b_min / scales) + self.qmin
        zero_points = np.clip(zero_points, self.qmin, self.qmax).astype(np.int8)

        q_blocks = np.round(W_blocks / scales) + zero_points
        q_blocks = np.clip(q_blocks, self.qmin, self.qmax).astype(np.int8)

        # Immediately dequantize for simulation
        dq_blocks = (q_blocks.astype(np.float32) - zero_points) * scales
        dq_flat = dq_blocks.flatten()

        if padding > 0:
            dq_flat = dq_flat[:-padding]

        return dq_flat.reshape(orig_shape)


# === 3. Run the Experiment and Lab Integration ===
if __name__ == "__main__":
    env = ObstacleEnv()
    pop = [RobotBrainV2() for _ in range(120)]

    print("Phase 1: Breeding the Genetic Overlord (100 Generations)...")
    for g in range(1, 101):
        for brain in pop: evaluate_brain(brain, env)
        pop.sort(key=lambda x: x.fitness, reverse=True)
        elites = pop[:18]
        new_pop = list(elites)
        while len(new_pop) < 120:
            p1, p2 = np.random.choice(elites, size=2, replace=False)
            alpha = np.random.rand()
            child = RobotBrainV2()
            child.W1 = alpha * p1.W1 + (1 - alpha) * p2.W1
            child.W2 = alpha * p1.W2 + (1 - alpha) * p2.W2
            child.mutate()
            new_pop.append(child)
        pop = new_pop

    champion = pop[0]
    fp32_fitness = evaluate_brain(champion, env)
    print(f"-> FP32 Champion Fitness established: {fp32_fitness:.2f}")

    print("\nPhase 2: Injecting Nero-Quantizer 4-bit Block-wise Compression...")
    quantizer = NeroQuantizerCore()

    # Clone the champion and compress each layer of its neural network independently
    quantized_champion = RobotBrainV2()
    quantized_champion.W1 = quantizer.quantize_tensor(champion.W1, block_size=4)
    quantized_champion.b1 = quantizer.quantize_tensor(champion.b1, block_size=4)
    quantized_champion.W2 = quantizer.quantize_tensor(champion.W2, block_size=4)
    quantizer.b2 = quantizer.quantize_tensor(champion.b2, block_size=4)

    # Evaluate the compressed champion's performance in the same challenging environment
    int4_fitness = evaluate_brain(quantized_champion, env)
    print(f"-> INT4 Quantized Champion Fitness established: {int4_fitness:.2f}")

    # Calculate the intelligence retention rate
    retention = (int4_fitness / fp32_fitness) * 100
    print("\n" + "=" * 60)
    print(f"FINAL REPORT: Quantization Robustness of Evolutionary Networks")
    print("-" * 60)
    print(f"FP32 Base Fitness          : {fp32_fitness:.2f}")
    print(f"INT4 Quant Fitness         : {int4_fitness:.2f}")
    print(f"Intelligence Retention Rate: {retention:.2f}%")
    print("=" * 60)