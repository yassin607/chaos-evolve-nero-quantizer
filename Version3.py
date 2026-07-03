import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==============================================================================
# Chaos-Evolve V9: Advanced Real Behaviors (Herd Instinct, Stamina & Balance)
# ==============================================================================

class AdvancedBrain:
    def __init__(self, input_dim=7, pos=None, is_predator=False):
        self.pos = np.array(pos if pos is not None else np.random.uniform(-9, 9, size=(2,)), dtype=np.float32)
        self.energy = 150.0 if is_predator else 90.0
        self.stamina = 100.0  # stamina feature for momentary sprinting
        self.age = 0
        self.is_predator = is_predator

        # Deeper neural network for environmental awareness
        self.W1 = np.random.randn(input_dim, 12) * 0.4
        self.b1 = np.zeros((1, 12))
        self.W2 = np.random.randn(12, 2) * 0.4

    def forward(self, X):
        # Process inputs with a stable simulated INT4 precision
        W1_q = np.clip(np.round(self.W1 * 4), -8, 7) / 4.0
        W2_q = np.clip(np.round(self.W2 * 4), -8, 7) / 4.0
        return np.dot(np.maximum(0, np.dot(X, W1_q) + self.b1), W2_q)

    def mutate(self):
        self.W1 += np.random.randn(*self.W1.shape) * 0.15
        self.W2 += np.random.randn(*self.W2.shape) * 0.15


class RealEcosystemEnv:
    def __init__(self, num_prey=25, num_pred=3, num_food=30):
        self.preys = [AdvancedBrain(input_dim=7, is_predator=False) for _ in range(num_prey)]
        self.predators = [AdvancedBrain(input_dim=5, is_predator=True) for _ in range(num_pred)]
        self.foods = np.random.uniform(-9, 9, size=(num_food, 2))
        self.time_step = 0

    def update(self):
        self.time_step += 1

        # Slowly replenish food naturally to prevent overpopulation
        if len(self.foods) < 25 and np.random.rand() < 0.4:
            self.foods = np.vstack([self.foods, np.random.uniform(-9, 9, size=(1, 2))])

        # 1. Prey behavior and update (Herd & Survival Logic)
        next_preys = []
        prey_positions = np.array([p.pos for p in self.preys]) if len(self.preys) > 0 else np.empty((0, 2))

        for prey in self.preys:
            prey.age += 1

            # Compute the center of mass of the nearby herd for flocking behavior
            if len(prey_positions) > 1:
                distances_to_herd = np.linalg.norm(prey_positions - prey.pos, axis=1)
                close_members = prey_positions[(distances_to_herd < 4.0) & (distances_to_herd > 0)]
                herd_vector = np.mean(close_members - prey.pos, axis=0) if len(close_members) > 0 else np.zeros(2)
            else:
                herd_vector = np.zeros(2)

            # Sense targets and threats
            closest_food = self.foods[np.argmin(np.linalg.norm(self.foods - prey.pos, axis=1))] if len(self.foods) > 0 else prey.pos
            closest_pred = self.predators[0].pos if len(self.predators) > 0 else np.array([100.0, 100.0])
            for pr in self.predators:
                if np.linalg.norm(pr.pos - prey.pos) < np.linalg.norm(closest_pred - prey.pos):
                    closest_pred = pr.pos

            # 7 neural inputs for the smart prey (food, threat, herd vector, own energy)
            state = np.array([
                closest_food[0] - prey.pos[0], closest_food[1] - prey.pos[1],
                closest_pred[0] - prey.pos[0], closest_pred[1] - prey.pos[1],
                herd_vector[0], herd_vector[1],
                prey.energy * 0.01
            ]).reshape(1, 7)

            action = prey.forward(state)[0]

            # Manage stamina for fast sprinting when in danger
            dist_to_danger = np.linalg.norm(closest_pred - prey.pos)
            speed_modifier = 0.6
            if dist_to_danger < 3.0 and prey.stamina > 20:
                speed_modifier = 1.1  # adrenaline sprint to escape!
                prey.stamina -= 4.0
            else:
                prey.stamina = min(100.0, prey.stamina + 1.5)  # recover while resting

            move = np.clip(action, -speed_modifier, speed_modifier)
            prey.pos = np.clip(prey.pos + move, -9.5, 9.5)

            # Consume vital energy
            prey.energy -= 0.4 + np.linalg.norm(move) * 0.3

            # Actual eating
            if len(self.foods) > 0:
                dists = np.linalg.norm(self.foods - prey.pos, axis=1)
                eaten = np.where(dists < 0.5)[0]
                if len(eaten) > 0:
                    prey.energy += 35.0
                    self.foods = np.delete(self.foods, eaten, axis=0)

            # Survival and balanced reproduction conditions
            if prey.energy <= 0 or prey.age > 400:
                continue

            if prey.energy >= 160.0 and len(self.preys) < 40:  # population cap to prevent overpopulation
                prey.energy -= 80.0
                child = AdvancedBrain(input_dim=7, pos=prey.pos + np.random.uniform(-0.3, 0.3, size=(2,)))
                child.W1, child.W2 = prey.W1.copy(), prey.W2.copy()
                child.mutate()
                next_preys.append(child)

            next_preys.append(prey)

        self.preys = next_preys if len(next_preys) > 0 else [AdvancedBrain(input_dim=7) for _ in range(5)]

        # 2. Smart predator behavior and update (Stamina Hunting Logic)
        next_predators = []
        for pred in self.predators:
            pred.age += 1

            if len(self.preys) > 0:
                # Target the weakest or closest prey
                dists_to_all_prey = np.linalg.norm([p.pos for p in self.preys] - pred.pos, axis=1)
                closest_prey = self.preys[np.argmin(dists_to_all_prey)].pos
                min_dist = np.min(dists_to_all_prey)
            else:
                closest_prey = pred.pos
                min_dist = 100.0

            # 5 inputs for the predator (prey position, its own stamina)
            pred_state = np.array([
                closest_prey[0] - pred.pos[0], closest_prey[1] - pred.pos[1],
                pred.stamina * 0.01, 1.0, 0.0
            ]).reshape(1, 5)

            pred_action = pred.forward(pred_state)[0]

            # Muscle fatigue mechanic for the predator (it can't run forever)
            if min_dist < 4.0 and pred.stamina > 15:
                pred_speed = 0.95  # burst attack
                pred.stamina -= 5.0
            else:
                pred_speed = 0.45  # slow walk to catch its breath
                pred.stamina = min(100.0, pred.stamina + 2.0)

            pred_move = np.clip(pred_action, -pred_speed, pred_speed)
            pred.pos = np.clip(pred.pos + pred_move, -9.5, 9.5)
            pred.energy -= 0.6 + np.linalg.norm(pred_move) * 0.4

            # Actual hunting
            if len(self.preys) > 0:
                dists_to_prey = np.linalg.norm([p.pos for p in self.preys] - pred.pos, axis=1)
                caught = np.where(dists_to_prey < 0.6)[0]
                if len(caught) > 0:
                    pred.energy += 70.0  # large hunting reward
                    # Remove the eaten prey
                    self.preys = [p for idx, p in enumerate(self.preys) if idx not in caught]

            # Strict predator death and reproduction conditions (ecological balance rule)
            if pred.energy <= 0 or pred.age > 500:
                continue

            # Raise the reproduction threshold significantly to avoid the screen filling with red like before
            if pred.energy >= 320.0 and len(self.predators) < 6:
                pred.energy -= 150.0
                p_child = AdvancedBrain(input_dim=5, pos=pred.pos + np.random.uniform(-0.5, 0.5, size=(2,)), is_predator=True)
                p_child.W1, p_child.W2 = pred.W1.copy(), pred.W2.copy()
                p_child.mutate()
                next_predators.append(p_child)

            next_predators.append(pred)

        self.predators = next_predators if len(next_predators) > 0 else [AdvancedBrain(input_dim=5, is_predator=True)]


# === Run the legendary realistic visual display ===
env = RealEcosystemEnv()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_facecolor('#FCFBF4')  # easy-on-the-eyes environment background

prey_scatter = ax.scatter([], [], c='#00E5FF', s=70, edgecolors='#006064', label='AI Prey (Herd Behavior)')
pred_scatter = ax.scatter([], [], c='#FF1744', marker='v', s=130, edgecolors='#7F0000', label='AI Predators (Stamina Hunter)')
food_scatter = ax.scatter([], [], c='#2E7D32', marker='o', s=40, label='Natural Food')

ax.legend(loc='upper right')
ax.grid(True, color='#E0E0E0', linestyle='--')
title_text = ax.set_title("")

def init():
    prey_scatter.set_offsets(np.empty((0, 2)))
    pred_scatter.set_offsets(np.empty((0, 2)))
    food_scatter.set_offsets(np.empty((0, 2)))
    return prey_scatter, pred_scatter, food_scatter

def animate(frame):
    env.update()

    if len(env.preys) > 0:
        prey_scatter.set_offsets(np.array([p.pos for p in env.preys]))
    else:
        prey_scatter.set_offsets(np.empty((0, 2)))

    if len(env.predators) > 0:
        pred_scatter.set_offsets(np.array([pr.pos for pr in env.predators]))
    else:
        pred_scatter.set_offsets(np.empty((0, 2)))

    food_scatter.set_offsets(env.foods)

    title_text.set_text(
        f"Chaos-Evolve V9 | Balanced Natural Ecosystem\n"
        f"Prey (Herd) Count: {len(env.preys)} | Predators (Red): {len(env.predators)} | Step: {frame}"
    )
    return prey_scatter, pred_scatter, food_scatter

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=2000, interval=35, blit=True, cache_frame_data=False)
plt.show()