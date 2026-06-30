import numpy as np


class RobotBrain:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        self.fitness = 0  # Measure of how smart this robot is

    def forward(self, X):
        # Compute outputs: ReLU activation is sufficient here, no need for complex functions
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Unbounded outputs to represent free movement in space
        return self.z2

    def mutate(self, rate=0.1, scale=0.2):
        """
        Apply genetic mutation: iterate over each weight, and if the rate condition
        is met, add a small random value (scale) to alter the network's behavior.
        """
        if np.random.rand() < rate:
            self.W1 += np.random.randn(*self.W1.shape) * scale
            self.b1 += np.random.randn(*self.b1.shape) * scale
            self.W2 += np.random.randn(*self.W2.shape) * scale
            self.b2 += np.random.randn(*self.b2.shape) * scale


class TargetTrackerEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Generate random coordinates for the robot and target between -10 and 10
        self.agent_pos = np.random.uniform(-10, 10, (1, 2))
        self.target_pos = np.random.uniform(-10, 10, (1, 2))
        self.steps_taken = 0
        self.max_steps = 40
        return self.get_state()

    def get_state(self):
        # Neural network input: the difference between the robot's position and the target's position
        # Shape will be (1, 2), consistent with input_size=2 in the Brain
        return self.target_pos - self.agent_pos

    def step(self, action):
        """
        The action coming from the neural network is a (1, 2) array
        representing the force and direction of movement along the X and Y axes.
        """
        # Clip the movement so the robot doesn't move at an extreme speed
        move = np.clip(action, -1.0, 1.0)
        self.agent_pos += move
        self.steps_taken += 1

        # Calculate the current distance between the robot and the target (Euclidean Distance)
        distance = np.linalg.norm(self.target_pos - self.agent_pos)

        # Episode ends if max steps are reached or the robot gets close enough to the target
        done = self.steps_taken >= self.max_steps or distance < 0.2
        return self.get_state(), distance, done


def evaluate_brain(brain, env, episodes=5): # هنخليه يختبر 5 مرات في أماكن مختلفة
    total_generation_distance = 0
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_distance = 0
        steps = 0
        
        while not done:
            action = brain.forward(state)
            state, distance, done = env.step(action)
            total_distance += distance
            steps += 1
            
        total_generation_distance += (total_distance / steps)
        
    # السكور النهائي هو متوسط المسافة عبر الـ 5 محاولات كاملة
    grand_avg_distance = total_generation_distance / episodes
    brain.fitness = 1000.0 / (grand_avg_distance + 0.001)


class EvolutionManager:
    def __init__(self, pop_size, input_size, hidden_size, output_size):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.env = TargetTrackerEnv()

        # 1. Create the first generation (completely random population)
        self.population = [
            RobotBrain(input_size, hidden_size, output_size)
            for _ in range(pop_size)
        ]

    def run_generation(self):
        # 2. Evaluate every individual in the current generation and calculate their fitness
        for brain in self.population:
            evaluate_brain(brain, self.env)

        # 3. Sort the generation from highest fitness to lowest
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        best_fitness = self.population[0].fitness

        # 4. Selection: pick the top 10% as "elites"
        elite_count = int(self.pop_size * 0.1)
        elites = self.population[:elite_count]

        # 5. Reproduction: build the new generation
        new_population = []

        # Keep the elites unchanged in the new generation so they don't lose their advantage
        new_population.extend(elites)

        while len(new_population) < self.pop_size:
            # Pick a random parent from the elites to copy their genes
            parent = np.random.choice(elites)

            # Create a child with the same network dimensions as the parent
            child = RobotBrain(self.input_size, self.hidden_size, self.output_size)

            # Copy weights and biases from parent to child exactly
            child.W1 = parent.W1.copy()
            child.b1 = parent.b1.copy()
            child.W2 = parent.W2.copy()
            child.b2 = parent.b2.copy()

            # Apply random genetic mutation to the child to explore new behaviors
            child.mutate(rate=0.15, scale=0.1)

            new_population.append(child)

        self.population = new_population
        return best_fitness


# --- Run the Simulation ---
if __name__ == "__main__":
    # Settings: 100 neural networks in the population
    # Inputs: 2 (X, Y), Hidden: 8, Outputs: 2 (action in X, Y)
    pop_size = 100
    generations = 100

    manager = EvolutionManager(pop_size=pop_size, input_size=2, hidden_size=8, output_size=2)

    print("Starting Chaos-Evolve Simulation...")
    print("-" * 40)

    for g in range(1, generations + 1):
        best_fit = manager.run_generation()

        # Print results every 10 generations to monitor evolution
        if g == 1 or g % 10 == 0:
            # Approximate average distance of the best model based on the fitness formula
            approx_avg_dist = (1000.0 / best_fit) - 0.001
            print(f"Generation {g:3d} | Best Fitness: {best_fit:8.2f} | Approx Best Avg Distance: {approx_avg_dist:5.2f}")

    print("-" * 40)
    print("Simulation Complete!")