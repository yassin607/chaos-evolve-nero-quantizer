import numpy as np
import matplotlib.pyplot as plt

# --- 1. الكلاسات الأساسية (نسخة خفيفة ومطورة للرسم) ---
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

class HardcodedAgent:
    def forward(self, X):
        to_target_x, to_target_y, to_obs_x, to_obs_y = X[0]
        move_x = 1.0 if to_target_x > 0 else -1.0
        move_y = 1.0 if to_target_y > 0 else -1.0
        if abs(to_obs_x) < 2.0 and abs(to_obs_y) < 3.0:
            move_x = 0.0
            move_y = 1.2 if to_obs_y > 0 else -1.2
        return np.array([[move_x, move_y]])

class ObstacleEnv:
    def __init__(self):
        self.reset()
    def reset(self):
        self.agent_pos = np.array([[-7.0, 0.0]]) # تثبيت نقطة البداية للمقارنة العادلة
        self.target_pos = np.array([[7.0, 0.5]]) # الهدف في اليمين
        self.obstacle_pos = np.array([[0.0, 0.0]]) # الحيطة في السنتر بالظبط
        self.steps_taken = 0
        return self.get_state()
    def get_state(self):
        return np.hstack((self.target_pos - self.agent_pos, self.obstacle_pos - self.agent_pos))
    def step(self, action):
        move = np.clip(action, -1.2, 1.2)
        next_pos = self.agent_pos + move
        hit_obstacle = False
        if (self.agent_pos[0, 0] < 0 and next_pos[0, 0] >= 0) or (self.agent_pos[0, 0] > 0 and next_pos[0, 0] <= 0):
            if abs(next_pos[0, 1] - self.obstacle_pos[0, 1]) < 2.5: # عرض الحيطة
                hit_obstacle = True
        if not hit_obstacle:
            self.agent_pos = next_pos
        self.steps_taken += 1
        distance = np.linalg.norm(self.target_pos - self.agent_pos)
        if hit_obstacle: distance += 15.0
        return self.get_state(), distance, self.steps_taken >= 50 or distance < 0.3

# --- 2. محرك تدريب سريع لخطف البطل ---
pop = [RobotBrainV2() for _ in range(100)]
env = ObstacleEnv()
for g in range(80): # تدريب سريع
    for brain in pop:
        state = env.reset()
        done, total_dist, steps, crossed = False, 0, 0, False
        while not done:
            state, dist, done = env.step(brain.forward(state))
            total_dist += dist; steps += 1
            if env.agent_pos[0, 0] > 0: crossed = True
        fit = 1000.0 / ((total_dist / steps) + 0.001)
        brain.fitness = fit * 2.5 if crossed else fit
    pop.sort(key=lambda x: x.fitness, reverse=True)
    elites = pop[:15]
    new_pop = list(elites)
    while len(new_pop) < 100:
        p1, p2 = np.random.choice(elites, size=2, replace=False)
        alpha = np.random.rand()
        child = RobotBrainV2()
        child.W1 = alpha*p1.W1 + (1-alpha)*p2.W1
        child.W2 = alpha*p1.W2 + (1-alpha)*p2.W2
        child.mutate()
        new_pop.append(child)
    pop = new_pop

# --- 3. تشغيل المحاكاة النهائية وتسجيل المسارات ---
genetic_champion = pop[0]
hardcoded_bot = HardcodedAgent()

def record_trajectory(agent):
    state = env.reset()
    trajectory = [env.agent_pos.copy()[0]]
    done = False
    while not done:
        action = agent.forward(state)
        state, _, done = env.step(action)
        trajectory.append(env.agent_pos.copy()[0])
    return np.array(trajectory)

gen_traj = record_trajectory(genetic_champion)
hard_traj = record_trajectory(hardcoded_bot)

# --- 4. الرسم البياني بـ Matplotlib ---
plt.figure(figsize=(10, 6))
# رسم الحيطة (مستطيل أحمر في السنتر)
plt.fill_between([-0.2, 0.2], -2.5, 2.5, color='red', alpha=0.6, label='Obstacle Wall')
# رسم البداية والهدف
plt.scatter([-7], [0], color='blue', s=200, marker='X', label='Start')
plt.scatter([7], [0.5], color='green', s=200, marker='*', label='Target')

# رسم المسارات
plt.plot(gen_traj[:, 0], gen_traj[:, 1], '-o', color='purple', linewidth=2, label='Genetic Overlord')
plt.plot(hard_traj[:, 0], hard_traj[:, 1], '--s', color='orange', linewidth=2, label='Hardcoded (If/Else)')

plt.title("Chaos-Evolve V2: Trajectory Showdown")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.legend()
plt.xlim(-9, 9)
plt.ylim(-5, 5)
plt.show()