import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==============================================================================
# Chaos-Evolve V7: The Living Sandbox (Artificial Life Simulation)
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


class LivingAgent:
    def __init__(self, pos, block_size=16, prune_mask=None, parent_weights=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.energy = 100.0
        self.age = 0
        self.block_size = block_size
        self.prune_mask = prune_mask if prune_mask is not None else np.ones((12,))
        self.quantizer = NeroQuantizerCore()

        if parent_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = parent_weights
        else:
            self.W1 = np.random.randn(4, 12) * 0.6
            self.b1 = np.zeros((1, 12))
            self.W2 = np.random.randn(12, 2) * 0.6
            self.b2 = np.zeros((1, 2))

    def forward(self, X):
        mask_W1 = self.prune_mask.reshape(1, 12)
        mask_W2 = self.prune_mask.reshape(12, 1)
        W1_eval = self.quantizer.quantize_and_prune(self.W1, self.block_size, mask_W1)
        b1_eval = self.quantizer.quantize_and_prune(self.b1, self.block_size, mask_W1)
        W2_eval = self.quantizer.quantize_and_prune(self.W2, self.block_size, mask_W2)
        return np.dot(np.maximum(0, np.dot(X, W1_eval) + b1_eval), W2_eval) + self.b2

    def mutate(self):
        scale = 0.15
        self.W1 += np.random.randn(*self.W1.shape) * scale
        self.b1 += np.random.randn(*self.b1.shape) * scale
        self.W2 += np.random.randn(*self.W2.shape) * scale
        if np.random.rand() < 0.20:
            idx = np.random.randint(0, 12)
            self.prune_mask[idx] = 1.0 - self.prune_mask[idx]
        if np.random.rand() < 0.10:
            self.block_size = np.random.choice([4, 8, 16])


class EcosystemEnv:
    def __init__(self, num_agents=15, num_food=20):
        self.num_food = num_food
        self.agents = [LivingAgent(pos=np.random.uniform(-9, 9, size=(2,))) for _ in range(num_agents)]
        self.foods = np.random.uniform(-9, 9, size=(num_food, 2))
        self.obstacle_y = 0.0
        self.time_step = 0

    def find_closest_food(self, agent_pos):
        if len(self.foods) == 0:
            return np.array([0.0, 0.0])
        dists = np.linalg.norm(self.foods - agent_pos, axis=1)
        return self.foods[np.argmin(dists)]

    def update(self):
        self.time_step += 1
        # Update the obstacle's movement sinusoidally
        self.obstacle_y = 5.0 * np.sin(self.time_step * 0.15)

        next_agents = []

        # If food drops below the target count, spawn new food
        if len(self.foods) < self.num_food:
            shortage = self.num_food - len(self.foods)
            new_food = np.random.uniform(-9, 9, size=(shortage, 2))
            self.foods = np.vstack([self.foods, new_food])

        for agent in self.agents:
            agent.age += 1
            closest_food = self.find_closest_food(agent.pos)

            # Build the sensory input: (vector to food, vector to moving obstacle)
            state = np.array([
                closest_food[0] - agent.pos[0], closest_food[1] - agent.pos[1],
                0.0 - agent.pos[0], self.obstacle_y - agent.pos[1]
            ]).reshape(1, 4)

            # Get the movement decision from the compressed INT4 brain
            action = agent.forward(state)[0]
            move = np.clip(action, -0.6, 0.6)

            # Check for collision with the moving obstacle
            next_pos = agent.pos + move
            hit_obstacle = False
            crossed_x = (agent.pos[0] < 0 and next_pos[0] >= 0) or (agent.pos[0] > 0 and next_pos[0] <= 0)
            if crossed_x and abs(next_pos[1] - self.obstacle_y) < 2.5:
                hit_obstacle = True

            if hit_obstacle:
                agent.energy -= 25.0  # collision penalty
                # bounce back to avoid getting stuck
                agent.pos[0] -= np.sign(move[0]) * 0.5
            else:
                agent.pos = np.clip(next_pos, -9.5, 9.5)
                # Movement energy cost (pruned neurons consume less energy, preserving the pruning advantage!)
                active_neurons = np.sum(agent.prune_mask)
                energy_cost = 0.5 + (active_neurons * 0.05) + (np.linalg.norm(move) * 0.4)
                agent.energy -= energy_cost

            # Eating: if close enough to a food item
            if len(self.foods) > 0:
                dists_to_food = np.linalg.norm(self.foods - agent.pos, axis=1)
                food_indices = np.where(dists_to_food < 0.6)[0]
                if len(food_indices) > 0:
                    agent.energy += 45.0 * len(food_indices)
                    self.foods = np.delete(self.foods, food_indices, axis=0)

            # Death from natural causes or energy depletion
            if agent.energy <= 0 or agent.age > 400:
                continue  # dies and does not move to the next generation list

            next_agents.append(agent)

            # Reproduction: genetic self-splitting when energy is abundant!
            if agent.energy >= 180.0:
                agent.energy -= 80.0  # birth costs energy
                child_weights = (agent.W1.copy(), agent.b1.copy(), agent.W2.copy(), agent.b2.copy())
                child = LivingAgent(pos=agent.pos + np.random.uniform(-0.5, 0.5, size=(2,)),
                                    block_size=agent.block_size,
                                    prune_mask=agent.prune_mask.copy(),
                                    parent_weights=child_weights)
                child.mutate()
                next_agents.append(child)

        # Safeguard against total extinction
        if len(next_agents) == 0:
            next_agents = [LivingAgent(pos=np.random.uniform(-9, 9, size=(2,))) for _ in range(5)]

        self.agents = next_agents


# === Run the live animation of the Sandbox environment ===
env = EcosystemEnv(num_agents=20, num_food=25)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

agent_scatter = ax.scatter([], [], c=[], cmap='cool', s=80, edgecolors='black', label='Living Robots (Color=Energy)')
food_scatter = ax.scatter([], [], c='green', marker='o', s=35, label='Energy Food')
obstacle_line, = ax.plot([], [], color='black', linewidth=8, label='Moving Threat')

ax.legend(loc='upper right')
ax.grid(True)
title_text = ax.set_title("")

def init():
    agent_scatter.set_offsets(np.empty((0, 2)))
    food_scatter.set_offsets(np.empty((0, 2)))
    obstacle_line.set_data([], [])
    return agent_scatter, food_scatter, obstacle_line

def animate(frame):
    env.update()

    # Update robots and their colors based on energy level
    if len(env.agents) > 0:
        agent_positions = np.array([a.pos for a in env.agents])
        agent_energies = [a.energy for a in env.agents]
        agent_scatter.set_offsets(agent_positions)
        agent_scatter.set_array(np.array(agent_energies))
    else:
        agent_scatter.set_offsets(np.empty((0, 2)))

    # Update food
    if len(env.foods) > 0:
        food_scatter.set_offsets(env.foods)
    else:
        food_scatter.set_offsets(np.empty((0, 2)))

    # Update the moving wall
    obstacle_line.set_data([0, 0], [env.obstacle_y - 2.5, env.obstacle_y + 2.5])

    # Compute the average number of active neurons among survivors
    avg_active = np.mean([np.sum(a.prune_mask) for a in env.agents]) if len(env.agents) > 0 else 0

    title_text.set_text(f"Chaos-Evolve V7 | Population: {len(env.agents)} | Avg Active Neurons: {avg_active:.1f}/12 | Step: {frame}")
    return agent_scatter, food_scatter, obstacle_line

# Run the endless animation to watch life evolve in real time!
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=50, blit=True, cache_frame_data=False)
plt.show()