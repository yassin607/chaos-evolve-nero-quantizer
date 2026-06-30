import numpy as np

# ============================================================
# Chaos-Evolve V4 — Final Edition
# التحسينات الجديدة عن V3:
# 1. بيئة ثابتة للمقارنة العادلة (Fixed Eval Env)
# 2. الاصطدام بالحيطة = نهاية فورية للجولة (done=True)
# 3. مكافأة تدريجية على الاقتراب من الهدف
# 4. retention موثوق لأن التقييم مش عشوائي
# ============================================================


# === 1. Neural Network Brain ===
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


# === 2. Obstacle Environment ===
class ObstacleEnv:
    def __init__(self, fixed_seed=None):
        """
        fixed_seed: لو حطيت رقم ثابت، البيئة هتبقى نفسها كل مرة.
        ده بيخلي المقارنة بين FP32 و INT4 عادلة تماماً.
        """
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

        # كشف الاصطدام
        hit_obstacle = False
        crossed_x = (self.agent_pos[0, 0] < 0 and next_pos[0, 0] >= 0) or \
                    (self.agent_pos[0, 0] > 0 and next_pos[0, 0] <= 0)
        if crossed_x and abs(next_pos[0, 1] - self.obstacle_pos[0, 1]) < 3.0:
            hit_obstacle = True

        if not hit_obstacle:
            self.agent_pos = next_pos

        self.steps_taken += 1
        distance = np.linalg.norm(self.target_pos - self.agent_pos)

        # FIX 2: الاصطدام = نهاية فورية بعقاب كبير
        if hit_obstacle:
            return self.get_state(), 30.0, True  # done=True فوراً

        # FIX 3: مكافأة تدريجية — لو اقترب من الهدف نكافئه
        progress = self.prev_distance - distance
        adjusted_distance = distance - (progress * 0.5)
        self.prev_distance = distance

        done = self.steps_taken >= 50 or distance < 0.3
        return self.get_state(), adjusted_distance, done


# === 3. Brain Evaluation (Random — for Training) ===
def evaluate_brain(brain, env, episodes=5):
    """بيئة عشوائية للتدريب عشان الروبوت يتعلم يتعامل مع مواقف مختلفة."""
    total_dist = 0
    crossed_wall = False

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


# === 4. Fixed Evaluation (for Fair Comparison) ===
def evaluate_fixed(brain, seeds):
    """
    FIX 1: نفس الـ seeds = نفس البيئة = مقارنة عادلة 100%.
    بنستخدمها بس للمقارنة النهائية مش للتدريب.
    """
    total_dist = 0
    crashes = 0

    for seed in seeds:
        fixed_env = ObstacleEnv(fixed_seed=seed)
        state = fixed_env.reset()
        done, ep_dist, steps = False, 0, 0
        crashed = False

        while not done:
            state, distance, done = fixed_env.step(brain.forward(state))
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


# === 5. Nero-Quantizer Core (INT4 Block-wise) ===
class NeroQuantizerCore:
    def __init__(self):
        self.qmin, self.qmax = -8, 7

    def quantize_tensor(self, W, block_size=4):
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

    def compress_brain(self, brain, block_size=4):
        """ياخد مخ كامل ويرجع نسخة مضغوطة — كل الـ layers بتتضغط بنفس الـ block_size."""
        compressed = RobotBrainV2()
        compressed.W1 = self.quantize_tensor(brain.W1, block_size=block_size)
        compressed.b1 = self.quantize_tensor(brain.b1, block_size=block_size)
        compressed.W2 = self.quantize_tensor(brain.W2, block_size=block_size)
        compressed.b2 = self.quantize_tensor(brain.b2, block_size=block_size)
        return compressed


# === 6. Benchmark ===
def run_benchmark(fp32_brain, int4_brain, seeds):
    class RandomAgent:
        def forward(self, X):
            return np.random.uniform(-1.2, 1.2, (1, 2))

    agents = {
        "FP32 Champion ": fp32_brain,
        "INT4 Quantized": int4_brain,
        "Random Blind  ": RandomAgent()
    }

    print(f"\n{'Agent':<20} | {'Avg Distance':^14} | {'Crashes':^10} | {'Fitness':^10}")
    print("-" * 65)

    all_results = {}
    for name, agent in agents.items():
        fitness, avg_dist, crashes = evaluate_fixed(agent, seeds)
        all_results[name] = {"fitness": fitness, "avg_dist": avg_dist, "crashes": crashes}
        print(f"{name:<20} | {avg_dist:^14.2f} | {crashes:^10} | {fitness:^10.2f}")

    return all_results


# === 7. Main ===
if __name__ == "__main__":
    train_env = ObstacleEnv()  # بيئة عشوائية للتدريب
    pop_size = 120
    generations = 100
    pop = [RobotBrainV2() for _ in range(pop_size)]

    # --- Phase 1: Genetic Training ---
    print("Phase 1: Breeding the Genetic Overlord...")
    print("=" * 60)

    for g in range(1, generations + 1):
        for brain in pop:
            evaluate_brain(brain, train_env)

        pop.sort(key=lambda x: x.fitness, reverse=True)
        elites = pop[:18]
        new_pop = list(elites)

        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(elites, size=2, replace=False)
            alpha = np.random.rand()
            child = RobotBrainV2()
            child.W1 = alpha * p1.W1 + (1 - alpha) * p2.W1
            child.b1 = alpha * p1.b1 + (1 - alpha) * p2.b1
            child.W2 = alpha * p1.W2 + (1 - alpha) * p2.W2
            child.b2 = alpha * p1.b2 + (1 - alpha) * p2.b2
            child.mutate(rate=0.2, scale=0.08)
            new_pop.append(child)

        pop = new_pop

        if g % 25 == 0 or g == 1:
            best = pop[0].fitness
            approx_dist = (1000.0 / best) - 0.001
            print(f"Gen {g:3d} | Best Fitness: {best:8.2f} | Approx Avg Distance: {approx_dist:.2f}")

    champion = pop[0]

    # نفس الـ 20 seed للكل — عدالة تامة
    eval_seeds = list(range(42, 62))

    # --- Phase 2: Quantization Sensitivity Analysis ---
    print("\nPhase 2: Quantization Sensitivity Analysis...")
    print("نجرب 4 block sizes مختلفة ونشوف أيهم أفضل للشبكة دي")
    print("=" * 60)

    fp32_fitness, fp32_dist, fp32_crashes = evaluate_fixed(champion, eval_seeds)
    print(f"FP32 Baseline  → Fitness: {fp32_fitness:.2f} | Avg Dist: {fp32_dist:.2f} | Crashes: {fp32_crashes}/20\n")

    block_sizes = [2, 4, 8, 16]
    best_block_size = None
    best_retention = -1
    best_quantized = None

    print(f"{'Block Size':<12} | {'Fitness':^10} | {'Avg Dist':^10} | {'Crashes':^10} | {'Retention':^10} | {'Verdict'}")
    print("-" * 75)

    quantizer = NeroQuantizerCore()
    for bs in block_sizes:
        q_brain = quantizer.compress_brain(champion, block_size=bs)
        fit, dist, crashes = evaluate_fixed(q_brain, eval_seeds)
        retention = (fit / fp32_fitness) * 100

        if 90 <= retention <= 110 and retention > best_retention:
            best_retention = retention
            best_block_size = bs
            best_quantized = q_brain
        elif best_quantized is None or retention > best_retention:
            best_retention = retention
            best_block_size = bs
            best_quantized = q_brain

        if 90 <= retention <= 110:
            verdict = "✅ Best"
        elif retention >= 75:
            verdict = "⚠️  OK"
        else:
            verdict = "❌ Poor"

        print(f"block={bs:<8} | {fit:^10.2f} | {dist:^10.2f} | {crashes:^10} | {retention:^9.1f}% | {verdict}")

    print(f"\n→ Best block size for this network: block={best_block_size} (Retention: {best_retention:.1f}%)")

    # --- Phase 3: Final Benchmark with best block size ---
    print(f"\nPhase 3: Final Benchmark using best block_size={best_block_size}...")
    print("=" * 60)

    run_benchmark(champion, best_quantized, eval_seeds)

    # --- Final Report ---
    final_retention = (evaluate_fixed(best_quantized, eval_seeds)[0] / fp32_fitness) * 100
    print("\n" + "=" * 60)
    print("FINAL REPORT: Quantization Robustness")
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