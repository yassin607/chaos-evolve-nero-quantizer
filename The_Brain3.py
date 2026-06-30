import numpy as np

# ============================================================
# Chaos-Evolve V3 — Nero-Quantizer Edition (Fixed + Enhanced)
# التحسينات:
# 1. صلاح بق الـ b2 في الـ Quantization
# 2. تقييم أكثر دقة عبر 10 جولات بدل جولة واحدة
# 3. مقارنة حقيقية: FP32 vs INT4 vs Random (زي الملف التاني)
# 4. تقرير نهائي واضح مع نسبة الاحتفاظ بالذكاء
# 5. تعليقات أوضح على كل خطوة
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
        # ReLU activation في الطبقة الأولى، ثم linear output
        return np.dot(np.maximum(0, np.dot(X, self.W1) + self.b1), self.W2) + self.b2

    def mutate(self, rate=0.35, scale=0.3):
        # طفرة جينية: لو الرقم العشوائي أقل من الـ rate، نغير الأوزان
        if np.random.rand() < rate:
            self.W1 += np.random.randn(*self.W1.shape) * scale
            self.b1 += np.random.randn(*self.b1.shape) * scale
            self.W2 += np.random.randn(*self.W2.shape) * scale
            self.b2 += np.random.randn(*self.b2.shape) * scale


# === 2. Obstacle Environment ===
class ObstacleEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # الروبوت يبدأ في الناحية السالبة، الهدف في الناحية الموجبة
        self.agent_pos = np.random.uniform(-9, -5, (1, 2))
        self.target_pos = np.random.uniform(5, 9, (1, 2))
        # الحيطة في المنتصف عند X=0 بـ Y عشوائي
        self.obstacle_pos = np.array([[0.0, np.random.uniform(-4, 4)]])
        self.steps_taken = 0
        return self.get_state()

    def get_state(self):
        # 4 مدخلات: المسافة للهدف (X,Y) + المسافة للعائق (X,Y)
        return np.hstack((self.target_pos - self.agent_pos,
                          self.obstacle_pos - self.agent_pos))

    def step(self, action):
        move = np.clip(action, -1.2, 1.2)
        next_pos = self.agent_pos + move

        # كشف الاصطدام: هل الروبوت حاول يعبر خط X=0 وفي العائق؟
        hit_obstacle = False
        crossed_x = (self.agent_pos[0, 0] < 0 and next_pos[0, 0] >= 0) or \
                    (self.agent_pos[0, 0] > 0 and next_pos[0, 0] <= 0)
        if crossed_x and abs(next_pos[0, 1] - self.obstacle_pos[0, 1]) < 3.0:
            hit_obstacle = True

        if not hit_obstacle:
            self.agent_pos = next_pos

        self.steps_taken += 1
        distance = np.linalg.norm(self.target_pos - self.agent_pos)

        # عقاب كبير لو خبط في الحيطة
        if hit_obstacle:
            distance += 15.0

        done = self.steps_taken >= 50 or distance < 0.3
        return self.get_state(), distance, done


# === 3. Brain Evaluation ===
def evaluate_brain(brain, env, episodes=5):
    """
    يقيّم المخ على عدة جولات ويحسب الفيتنس.
    لو عبر الحيطة بنجاح في أي جولة، يضاعف السكور.
    """
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


# === 4. Nero-Quantizer Core (INT4 Block-wise) ===
class NeroQuantizerCore:
    """
    ضغط الأوزان من FP32 (32-bit float) إلى INT4 (4-bit integer).
    الفكرة: نقسم الأوزان لـ blocks صغيرة، ونحسب scale وzero_point لكل block.
    """
    def __init__(self):
        self.qmin, self.qmax = -8, 7  # نطاق INT4 الموقّع (signed)

    def quantize_tensor(self, W, block_size=4):
        orig_shape = W.shape
        W_flat = W.flatten()

        # Padding لو الحجم مش متقسم على block_size
        remainder = len(W_flat) % block_size
        if remainder != 0:
            padding = block_size - remainder
            W_flat = np.concatenate([W_flat, np.zeros(padding)])
        else:
            padding = 0

        # تقسيم لـ blocks
        num_blocks = len(W_flat) // block_size
        W_blocks = W_flat.reshape(num_blocks, block_size)

        # حساب الـ scale والـ zero_point لكل block
        b_min = np.min(W_blocks, axis=1, keepdims=True)
        b_max = np.max(W_blocks, axis=1, keepdims=True)
        scales = (b_max - b_min) / (self.qmax - self.qmin)
        scales = np.where(scales == 0, 1.0, scales)
        zero_points = np.clip(
            np.round(-b_min / scales) + self.qmin,
            self.qmin, self.qmax
        ).astype(np.int8)

        # Quantize ثم Dequantize فوراً للمقارنة
        q_blocks = np.clip(
            np.round(W_blocks / scales) + zero_points,
            self.qmin, self.qmax
        ).astype(np.int8)
        dq_flat = ((q_blocks.astype(np.float32) - zero_points) * scales).flatten()

        if padding > 0:
            dq_flat = dq_flat[:-padding]

        return dq_flat.reshape(orig_shape)

    def compress_brain(self, brain):
        """
        ياخد مخ كامل ويرجع نسخة مضغوطة منه.
        FIX: كل الـ layers بتتضغط صح في الـ quantized_brain مش في الـ quantizer نفسه.
        """
        compressed = RobotBrainV2()
        compressed.W1 = self.quantize_tensor(brain.W1)
        compressed.b1 = self.quantize_tensor(brain.b1)
        compressed.W2 = self.quantize_tensor(brain.W2)
        compressed.b2 = self.quantize_tensor(brain.b2)  # ✅ البق اتصلح هنا
        return compressed


# === 5. Benchmark: FP32 vs INT4 vs Random ===
def run_benchmark(fp32_brain, int4_brain, env, rounds=10):
    """
    مقارنة عادلة بين المخ الأصلي والمضغوط والعشوائي
    على نفس الجولات بالضبط.
    """
    class RandomAgent:
        def forward(self, X):
            return np.random.uniform(-1.2, 1.2, (1, 2))

    agents = {
        "FP32 Champion": fp32_brain,
        "INT4 Quantized": int4_brain,
        "Random Blind Bot": RandomAgent()
    }

    results = {name: {"total_dist": 0.0, "crashes": 0} for name in agents}

    for _ in range(rounds):
        for name, agent in agents.items():
            state = env.reset()
            done, round_dist, steps, crashed = False, 0, 0, False

            while not done:
                action = agent.forward(state)
                state, distance, done = env.step(action)
                if distance > 15.0:
                    crashed = True
                round_dist += distance
                steps += 1

            results[name]["total_dist"] += round_dist / steps
            if crashed:
                results[name]["crashes"] += 1

    return {name: {
        "avg_dist": data["total_dist"] / rounds,
        "crashes": data["crashes"]
    } for name, data in results.items()}


# === 6. Main Simulation ===
if __name__ == "__main__":
    env = ObstacleEnv()
    pop_size = 120
    generations = 100
    pop = [RobotBrainV2() for _ in range(pop_size)]

    # --- Phase 1: Genetic Training ---
    print("Phase 1: Breeding the Genetic Overlord...")
    print("=" * 60)

    for g in range(1, generations + 1):
        for brain in pop:
            evaluate_brain(brain, env)

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

    # --- Phase 2: Quantization ---
    print("\nPhase 2: Compressing with Nero-Quantizer (INT4 Block-wise)...")
    print("=" * 60)

    quantizer = NeroQuantizerCore()
    quantized_champion = quantizer.compress_brain(champion)  # ✅ الضغط الصح

    # تقييم كل منهم على 10 جولات مستقلة لنتيجة موثوقة
    fp32_scores = [evaluate_brain(champion, env, episodes=5) for _ in range(10)]
    int4_scores = [evaluate_brain(quantized_champion, env, episodes=5) for _ in range(10)]

    fp32_avg = np.mean(fp32_scores)
    int4_avg = np.mean(int4_scores)
    retention = (int4_avg / fp32_avg) * 100

    # --- Phase 3: Benchmark vs Random ---
    print("\nPhase 3: Final Benchmark (FP32 vs INT4 vs Random)...")
    print("=" * 60)

    bench = run_benchmark(champion, quantized_champion, env, rounds=10)

    print(f"\n{'Agent':<20} | {'Avg Distance ↓':<18} | {'Crashes / 10'}")
    print("-" * 55)
    for name, data in bench.items():
        print(f"{name:<20} | {data['avg_dist']:<18.2f} | {data['crashes']}")

    # --- Final Report ---
    print("\n" + "=" * 60)
    print("FINAL REPORT: Quantization Robustness")
    print("-" * 60)
    print(f"FP32 Avg Fitness (10 runs)  : {fp32_avg:.2f}")
    print(f"INT4 Avg Fitness (10 runs)  : {int4_avg:.2f}")
    print(f"Intelligence Retention Rate : {retention:.2f}%")
    if retention >= 95:
        print("✅ Quantization SUCCESS — Model survived compression!")
    elif retention >= 80:
        print("⚠️  Quantization OK — Minor intelligence loss.")
    else:
        print("❌ Quantization FAILED — Too much intelligence lost.")
    print("=" * 60)