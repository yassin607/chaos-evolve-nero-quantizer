import numpy as np

class RobotBrainV2:
    def __init__(self, input_size=4, hidden_size=12, output_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # الأبعاد الجديدة للمصفوفات لاستيعاب 4 مدخلات
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.fitness = 0

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def mutate(self, rate=0.35, scale=0.3): # رفعنا المعدل والسكيل للجنون الجيني
        if np.random.rand() < rate:
            self.W1 += np.random.randn(*self.W1.shape) * scale
            self.b1 += np.random.randn(*self.b1.shape) * scale
            self.W2 += np.random.randn(*self.W2.shape) * scale
            self.b2 += np.random.randn(*self.b2.shape) * scale

class ObstacleEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # الروبوت يبدأ في اليسار، الهدف في اليمين، والحيطة في النص عند X = 0
        self.agent_pos = np.random.uniform(-9, -5, (1, 2))
        self.target_pos = np.random.uniform(5, 9, (1, 2))
        self.obstacle_pos = np.array([[0.0, np.random.uniform(-4, 4)]]) # الحيطة في المنتصف تماماً
        
        self.steps_taken = 0
        self.max_steps = 50
        return self.get_state()
        
    def get_state(self):
        # 4 مدخلات كاملة للمخ
        to_target = self.target_pos - self.agent_pos
        to_obstacle = self.obstacle_pos - self.agent_pos
        return np.hstack((to_target, to_obstacle)) # مصفوفة بأبعاد (1, 4)

    def step(self, action):
        move = np.clip(action, -1.2, 1.2)
        next_pos = self.agent_pos + move
        
        # نظام كشف الاصطدام: لو الروبوت حاول يعبر خط الـ X = 0 وقريب من الـ Y بتاعة الحيطة
        hit_obstacle = False
        if (self.agent_pos[0, 0] < 0 and next_pos[0, 0] >= 0) or (self.agent_pos[0, 0] > 0 and next_pos[0, 0] <= 0):
            if abs(next_pos[0, 1] - self.obstacle_pos[0, 1]) < 3.0: # عرض الحيطة 3 وحدات
                hit_obstacle = True
                
        if not hit_obstacle:
            self.agent_pos = next_pos
            
        self.steps_taken += 1
        distance = np.linalg.norm(self.target_pos - self.agent_pos)
        
        # عقاب شديد في الـ Distance لو خبط في الحيطة
        if hit_obstacle:
            distance += 15.0 
            
        done = self.steps_taken >= self.max_steps or distance < 0.3
        return self.get_state(), distance, done


def evaluate_brain(brain, env, episodes=4):
    total_generation_distance = 0
    crossed_wall_successfully = False
    
    for _ in range(episodes):
        state = env.reset()
        start_x = env.agent_pos[0, 0] # مكان البداية (سالب دايماً)
        done = False
        total_distance = 0
        steps = 0
        
        while not done:
            action = brain.forward(state)
            state, distance, done = env.step(action)
            total_distance += distance
            steps += 1
            
            # لو الروبوت نجح يعبر للناحية الموجبة (حيث الهدف) ومخبطش
            if env.agent_pos[0, 0] > 0 and start_x < 0:
                crossed_wall_successfully = True
                
        total_generation_distance += (total_distance / steps)
        
    grand_avg_distance = total_generation_distance / episodes
    base_fitness = 1000.0 / (grand_avg_distance + 0.001)
    
    # الجائزة الكبرى: لو عبر الحيطة بنجاح في أي إيبسود، ضاعف السكور بتاعه فوراً!
    if crossed_wall_successfully:
        brain.fitness = base_fitness * 2.5
    else:
        brain.fitness = base_fitness


class ChaosEvolveV2Engine:
    def __init__(self, pop_size=100):
        self.pop_size = pop_size
        self.env = ObstacleEnv()
        self.population = [RobotBrainV2() for _ in range(pop_size)]

    def crossover(self, parent1, parent2):
        # التكاثر المشترك: دمج جينات الأبوين بنسبة عشوائية ألفا
        child = RobotBrainV2()
        alpha = np.random.rand()
        
        child.W1 = alpha * parent1.W1.copy() + (1 - alpha) * parent2.W1.copy()
        child.b1 = alpha * parent1.b1.copy() + (1 - alpha) * parent2.b1.copy()
        child.W2 = alpha * parent1.W2.copy() + (1 - alpha) * parent2.W2.copy()
        child.b2 = alpha * parent1.b2.copy() + (1 - alpha) * parent2.b2.copy()
        return child

    def run_generation(self):
        for brain in self.population:
            evaluate_brain(brain, self.env)
            
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness = self.population[0].fitness
        
        # انتخاب الـ Elites (أفضل 15%)
        elite_count = int(self.pop_size * 0.15)
        elites = self.population[:elite_count]
        
        new_population = []
        new_population.extend(elites) # الحفاظ على الصفوة
        
        while len(new_population) < self.pop_size:
            # اختيار أَبين مختلفين عشوائياً من الصفوة للـ Crossover
            p1, p2 = np.random.choice(elites, size=2, replace=False)
            child = self.crossover(p1, p2)
            
            # تطبيق الطفرة الجينية بعد الدمج
            child.mutate(rate=0.2, scale=0.08)
            new_population.append(child)
            
        self.population = new_population
        return best_fitness

class HardcodedAgent:
    """الروبوت التقليدي: مبرمج بقواعد بشرية جامدة If/Else"""
    def forward(self, X):
        # X فيه: [to_target_x, to_target_y, to_obstacle_x, to_obstacle_y]
        to_target_x, to_target_y, to_obs_x, to_obs_y = X[0]
        
        move_x = 1.0 if to_target_x > 0 else -1.0
        move_y = 1.0 if to_target_y > 0 else -1.0
        
        # لو الحيطة قريبة جداً في الـ X والـ Y، حاول يهرب منها بشكل يدوي
        if abs(to_obs_x) < 2.0 and abs(to_obs_y) < 3.0:
            move_x = 0.0  # اقف في الـ X
            move_y = 1.2 if to_obs_y > 0 else -1.2 # هرب عمودي
            
        return np.array([[move_x, move_y]])

class RandomAgent:
    """الروبوت الأعمى: حظ عشوائي صِرَف"""
    def forward(self, X):
        return np.random.uniform(-1.2, 1.2, (1, 2))


if __name__ == "__main__":
    # 1. تدريب المحرك الجيني أولاً
    engine = ChaosEvolveV2Engine(pop_size=120)
    print("Step 1: Training the Genetic Overlord (150 Generations)...")
    print("=" * 60)
    
    for g in range(1, 151):
        best_fit = engine.run_generation()
        if g % 30 == 0 or g == 1:
            approx_dist = (1000.0 / best_fit) - 0.001
            print(f"Gen {g:3d} | Elite Fitness: {best_fit:8.2f} | Distance+Penalty: {approx_dist:5.2f}")
            
    # اختطاف بطل العالم الجيني بعد جيل 150
    genetic_champion = engine.population[0]
    
    # 2. تجهيز المنافسين
    hardcoded_bot = HardcodedAgent()
    random_bot = RandomAgent()
    test_env = ObstacleEnv()
    
    # 3. المعركة النهائية: 10 جولات اختبار قاسية
    print("\nStep 2: Commencing the Ultimate Benchmark (10 Hardcore Rounds)...")
    print("=" * 60)
    
    agents = {
        "Genetic Overlord": genetic_champion,
        "Hardcoded (If/Else)": hardcoded_bot,
        "Random Blind Bot": random_bot
    }
    
    results = {name: {"score": 0.0, "crashes": 0} for name in agents}
    
    for round_num in range(1, 11):
        for name, agent in agents.items():
            state = test_env.reset()
            done = False
            round_dist = 0
            steps = 0
            crashed = False
            
            while not done:
                action = agent.forward(state)
                state, distance, done = test_env.step(action)
                
                # كشف إذا كان خبط في الجولة دي (لو المسافة ضربت بسبب العقاب)
                if distance > 15.0:
                    crashed = True
                    
                round_dist += distance
                steps += 1
                
            avg_dist = round_dist / steps
            results[name]["score"] += avg_dist
            if crashed:
                results[name]["crashes"] += 1

    # 4. طباعة جدول الحسم المعياري
    print(f"{'Agent Name':<25} | {'Avg Distance (Lower is Better)':<30} | {'Total Crashes':<15}")
    print("-" * 75)
    for name, data in results.items():
        final_score = data["score"] / 10.0
        print(f"{name:<25} | {final_score:<30.2f} | {data['crashes']:<15}")
    print("=" * 75)