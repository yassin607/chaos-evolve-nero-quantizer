import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==============================================================================
# Chaos-Evolve V8: The Ultimate Megasystem (Predator, Prey, Swarm & Seasons)
# ==============================================================================

class NeroQuantizerCore:
    def __init__(self):
        self.qmin, self.qmax = -8, 7

    def quantize_and_prune(self, W, block_size, prune_mask):
        W_pruned = W * prune_mask
        if block_size is None or block_size <= 0:
            return W_pruned
        orig_shape = W_pruned.shape
        W_flat = W_pruned.flatten()
        remainder = len(W_flat) % block_size
        if remainder != 0:
            W_flat = np.concatenate([W_flat, np.zeros(block_size - remainder)])
        num_blocks = len(W_flat) // block_size
        W_blocks = W_flat.reshape(num_blocks, block_size)
        b_min = np.min(W_blocks, axis=1, keepdims=True)
        b_max = np.max(W_blocks, axis=1, keepdims=True)
        scales = (b_max - b_min) / (self.qmax - self.qmin)
        scales = np.where(scales == 0, 1.0, scales)
        zero_points = np.clip(np.round(-b_min / scales) + self.qmin, self.qmin, self.qmax).astype(np.int8)
        q_blocks = np.clip(np.round(W_blocks / scales) + zero_points, self.qmin, self.qmax).astype(np.int8)
        dq_flat = ((q_blocks.astype(np.float32) - zero_points) * scales).flatten()
        if remainder != 0:
            dq_flat = dq_flat[:-(block_size - remainder)]
        return dq_flat.reshape(orig_shape)


class MegasystemBrain:
    def __init__(self, input_dim=6, pos=None, block_size=16, is_predator=False):
        self.pos = np.array(pos if pos is not None else np.random.uniform(-9, 9, size=(2,)), dtype=np.float32)
        self.energy = 120.0 if is_predator else 100.0
        self.age = 0
        self.block_size = block_size
        self.is_predator = is_predator
        self.prune_mask = np.ones((16,))  # larger hidden layer to handle the added complexity
        self.quantizer = NeroQuantizerCore()

        # Neural network weights
        self.W1 = np.random.randn(input_dim, 16) * 0.5
        self.b1 = np.zeros((1, 16))
        self.W2 = np.random.randn(16, 2) * 0.5

    def forward(self, X):
        mask_W1 = self.prune_mask.reshape(1, 16)
        mask_W2 = self.prune_mask.reshape(16, 1)
        W1_eval = self.quantizer.quantize_and_prune(self.W1, self.block_size, mask_W1)
        b1_eval = self.quantizer.quantize_and_prune(self.b1, self.block_size, mask_W1)
        W2_eval = self.quantizer.quantize_and_prune(self.W2, self.block_size, mask_W2)
        return np.dot(np.maximum(0, np.dot(X, W1_eval) + b1_eval), W2_eval)

    def mutate(self):
        scale = 0.2
        self.W1 += np.random.randn(*self.W1.shape) * scale
        self.b1 += np.random.randn(*self.b1.shape) * scale
        self.W2 += np.random.randn(*self.W2.shape) * scale
        if np.random.rand() < 0.25:
            idx = np.random.randint(0, 16)
            self.prune_mask[idx] = 1.0 - self.prune_mask[idx]


class MegasystemEnv:
    def __init__(self, num_prey=15, num_food=25):
        self.num_food = num_food
        self.preys = [MegasystemBrain(input_dim=6, is_predator=False) for _ in range(num_prey)]
        self.predators = [MegasystemBrain(input_dim=4, is_predator=True, pos=[8.0, 8.0])]
        self.foods = np.random.uniform(-9, 9, size=(num_food, 2))

        # Cooperative pheromone grid
        self.pheromone_grid = np.zeros((20, 20))
        self.time_step = 0
        self.season = "Spring"

    def get_season_parameters(self):
        # Dynamic seasons and disasters
        cycle = (self.time_step // 150) % 3
        if cycle == 0:
            self.season = "Spring (Normal)"
            return 0.6, 25  # normal speed, food is plentiful
        elif cycle == 1:
            self.season = "Summer Famine (Low Food)"
            return 0.7, 8   # scarce food, faster predator movement
        else:
            self.season = "Winter Storm (High Energy Drain)"
            return 0.4, 20  # slower movement and higher energy consumption for everyone

    def update(self):
        self.time_step += 1
        max_speed, target_food_count = self.get_season_parameters()

        # Pheromones evaporate over time
        self.pheromone_grid *= 0.92

        # Replenish food based on the season
        if len(self.foods) < target_food_count:
            shortage = target_food_count - len(self.foods)
            self.foods = np.vstack([self.foods, np.random.uniform(-9, 9, size=(shortage, 2))])

        # 1. Update prey
        next_preys = []
        for prey in self.preys:
            prey.age += 1

            # Find the closest food and the closest predator
            closest_food = self.foods[np.argmin(np.linalg.norm(self.foods - prey.pos, axis=1))] if len(self.foods) > 0 else prey.pos
            closest_pred = self.predators[0].pos if len(self.predators) > 0 else np.array([100.0, 100.0])

            # Sense nearby pheromones
            p_x = int((prey.pos[0] + 10) % 20)
            p_y = int((prey.pos[1] + 10) % 20)
            ph_signal = self.pheromone_grid[p_x, p_y]

            # Prey inputs (6 inputs: food vector, predator vector, pheromone signal)
            state = np.array([
                closest_food[0] - prey.pos[0], closest_food[1] - prey.pos[1],
                closest_pred[0] - prey.pos[0], closest_pred[1] - prey.pos[1],
                ph_signal, 1.0
            ]).reshape(1, 6)

            action = prey.forward(state)[0]
            move = np.clip(action, -max_speed, max_speed)
            prey.pos = np.clip(prey.pos + move, -9.5, 9.5)

            # Seasonal energy drain
            drain_modifier = 1.5 if "Winter" in self.season else 1.0
            prey.energy -= (0.6 + np.linalg.norm(move) * 0.5) * drain_modifier

            # Eating and releasing pheromones
            if len(self.foods) > 0:
                dists = np.linalg.norm(self.foods - prey.pos, axis=1)
                eaten = np.where(dists < 0.6)[0]
                if len(eaten) > 0:
                    prey.energy += 40.0
                    self.foods = np.delete(self.foods, eaten, axis=0)
                    # Leave a strong pheromone trail where food was found, to help the group!
                    self.pheromone_grid[p_x, p_y] += 15.0

            if prey.energy <= 0 or prey.age > 350:
                continue
            next_preys.append(prey)

            # Self-reproduction
            if prey.energy >= 170.0:
                prey.energy -= 70.0
                child = MegasystemBrain(input_dim=6, pos=prey.pos + np.random.uniform(-0.4, 0.4, size=(2,)))
                child.W1, child.W2 = prey.W1.copy(), prey.W2.copy()
                child.mutate()
                next_preys.append(child)

        self.preys = next_preys if len(next_preys) > 0 else [MegasystemBrain(input_dim=6) for _ in range(5)]

        # 2. Update the intelligent predator (AI Predator)
        next_predators = []
        for pred in self.predators:
            pred.age += 1

            # The predator targets the nearest prey
            if len(self.preys) > 0:
                closest_prey = self.preys[np.argmin(np.linalg.norm([p.pos for p in self.preys] - pred.pos, axis=1))].pos
            else:
                closest_prey = pred.pos

            # Predator inputs (4 inputs: prey vector only)
            pred_state = np.array([closest_prey[0] - pred.pos[0], closest_prey[1] - pred.pos[1], 0.0, 1.0]).reshape(1, 4)
            pred_action = pred.forward(pred_state)[0]
            pred_move = np.clip(pred_action, -(max_speed + 0.15), (max_speed + 0.15))  # slightly faster than prey
            pred.pos = np.clip(pred.pos + pred_move, -9.5, 9.5)

            pred.energy -= 0.85  # continuous hunting energy cost

            # Hunt and eat prey!
            if len(self.preys) > 0:
                dists_to_prey = np.linalg.norm([p.pos for p in self.preys] - pred.pos, axis=1)
                caught_indices = np.where(dists_to_prey < 0.7)[0]
                if len(caught_indices) > 0:
                    pred.energy += 60.0 * len(caught_indices)
                    # Remove the eaten prey from existence
                    self.preys = [p for idx, p in enumerate(self.preys) if idx not in caught_indices]

            if pred.energy <= 0 or pred.age > 600:
                continue
            next_predators.append(pred)

            # Predator reproduces once fully sated
            if pred.energy >= 220.0:
                pred.energy -= 100.0
                p_child = MegasystemBrain(input_dim=4, pos=pred.pos + np.random.uniform(-0.5, 0.5, size=(2,)), is_predator=True)
                p_child.W1, p_child.W2 = pred.W1.copy(), pred.W2.copy()
                p_child.mutate()
                next_predators.append(p_child)

        self.predators = next_predators if len(next_predators) > 0 else [MegasystemBrain(input_dim=4, is_predator=True)]


# === Run the legendary visual display of the megasystem ===
env = MegasystemEnv()

fig, ax = plt.subplots(figsize=(11, 9))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

prey_scatter = ax.scatter([], [], c=[], cmap='cool', s=80, edgecolors='black', label='AI Prey (Cyan/Blue)')
pred_scatter = ax.scatter([], [], c='red', marker='v', s=120, edgecolors='black', label='AI Predators (Red)')
food_scatter = ax.scatter([], [], c='green', marker='o', s=35, label='Energy Food')

# Display pheromones as a dynamic glowing background
pheromone_img = ax.imshow(env.pheromone_grid, extent=[-10, 10, -10, 10], origin='lower', cmap='YlGn', alpha=0.25, zorder=-1)

ax.legend(loc='upper right')
ax.grid(True)
title_text = ax.set_title("")

def init():
    prey_scatter.set_offsets(np.empty((0, 2)))
    pred_scatter.set_offsets(np.empty((0, 2)))
    food_scatter.set_offsets(np.empty((0, 2)))
    return prey_scatter, pred_scatter, food_scatter

def animate(frame):
    env.update()

    # Update prey
    if len(env.preys) > 0:
        prey_scatter.set_offsets(np.array([p.pos for p in env.preys]))
        prey_scatter.set_array(np.array([p.energy for p in env.preys]))
    else:
        prey_scatter.set_offsets(np.empty((0, 2)))

    # Update predators
    if len(env.predators) > 0:
        pred_scatter.set_offsets(np.array([pr.pos for pr in env.predators]))
    else:
        pred_scatter.set_offsets(np.empty((0, 2)))

    # Update food and pheromones
    food_scatter.set_offsets(env.foods)
    pheromone_img.set_data(env.pheromone_grid.T)

    title_text.set_text(
        f"Chaos-Evolve V8 | Season: {env.season}\n"
        f"Prey Count: {len(env.preys)} | Predators: {len(env.predators)} | Step: {frame}"
    )
    return prey_scatter, pred_scatter, food_scatter, pheromone_img

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1500, interval=40, blit=True, cache_frame_data=False)
plt.show()