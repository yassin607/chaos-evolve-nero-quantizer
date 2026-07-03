import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree

# ==============================================================================
# Chaos-Evolve V10: KD-tree perf, sexual reproduction, adaptive mutation,
# recurrent memory, patchy food, live population chart, no silent respawns
# ==============================================================================

MEMORY_DIM = 2  # small recurrent state each agent carries frame-to-frame
WORLD_LO, WORLD_HI = -9.5, 9.5

# Instinct priors for the founding generation: (input_index, output_index, coeff).
# output_index 0 = move_x, 1 = move_y. Prey state layout is
# [food_dx, food_dy, threat_dx, threat_dy, herd_dx, herd_dy, energy];
# predator state layout is [prey_dx, prey_dy, stamina, bias, bias].
# A negative coeff on the threat channels means "move away from positive
# direction to threat", i.e. flee. This is NOT a hard-coded behavior override —
# it's just a sane starting point for the weights; mutation and reproduction
# still freely reshape it every generation from here.
PREY_INSTINCT = [(0, 0, 0.5), (1, 1, 0.5),      # seek food
                 (2, 0, -0.8), (3, 1, -0.8),     # flee threat
                 (4, 0, 0.15), (5, 1, 0.15)]     # mild herd cohesion
PRED_INSTINCT = [(0, 0, 0.9), (1, 1, 0.9)]       # chase prey


class AdvancedBrain:
    """A tiny recurrent MLP brain. Weights are simulated-INT4 quantized,
    but only re-quantized when they actually change (mutation), not on
    every forward pass."""

    def __init__(self, input_dim, pos=None, is_predator=False, generation=0, instinct=None):
        self.pos = np.array(pos if pos is not None else np.random.uniform(-9, 9, size=(2,)), dtype=np.float32)
        self.energy = 150.0 if is_predator else 90.0
        self.stamina = 100.0
        self.age = 0
        self.is_predator = is_predator
        self.generation = generation
        self.memory = np.zeros(MEMORY_DIM, dtype=np.float32)
        self.bursting = False  # visual flag: sprinting / attacking this frame

        self.input_dim = input_dim
        out_dim = 2 + MEMORY_DIM
        if instinct is not None:
            self.W1, self.b1, self.W2 = self._instinct_weights(input_dim, 12, out_dim, instinct)
        else:
            self.W1 = np.random.randn(input_dim, 12) * 0.4
            self.b1 = np.zeros((1, 12))
            self.W2 = np.random.randn(12, out_dim) * 0.4

        self._quantized = None
        self._quantize()

    @staticmethod
    def _instinct_weights(input_dim, hidden_dim, out_dim, channels, noise=0.05):
        """Build a linear 'reflex' controller inside the MLP using a paired-ReLU
        trick: relu(x) - relu(-x) == x, so two hidden units per channel let a
        chosen input drive a chosen output linearly, on top of small random
        noise for population diversity. Everything is still mutable afterward."""
        W1 = np.random.randn(input_dim, hidden_dim) * noise
        W2 = np.random.randn(hidden_dim, out_dim) * noise
        b1 = np.zeros((1, hidden_dim))
        slot = 0
        for in_idx, out_idx, coeff in channels:
            if slot + 1 >= hidden_dim:
                break
            W1[in_idx, slot] = 1.0
            W1[in_idx, slot + 1] = -1.0
            W2[slot, out_idx] = coeff
            W2[slot + 1, out_idx] = -coeff
            slot += 2
        return W1, b1, W2

    def _quantize(self):
        W1_q = np.clip(np.round(self.W1 * 4), -8, 7) / 4.0
        W2_q = np.clip(np.round(self.W2 * 4), -8, 7) / 4.0
        self._quantized = (W1_q, W2_q)

    def forward(self, state_vec):
        """state_vec: 1D array of length input_dim - MEMORY_DIM (memory gets appended here)."""
        X = np.concatenate([state_vec, self.memory]).reshape(1, -1)
        W1_q, W2_q = self._quantized
        h = np.maximum(0, np.dot(X, W1_q) + self.b1)
        out = np.dot(h, W2_q)[0]
        self.memory = np.tanh(out[2:])
        return out[:2]

    def mutate(self, rate=0.15):
        self.W1 += np.random.randn(*self.W1.shape) * rate
        self.W2 += np.random.randn(*self.W2.shape) * rate
        self._quantize()

    @staticmethod
    def crossover(parent_a, parent_b, pos, mutation_rate, is_predator=False):
        """Sexual reproduction: each weight is inherited from a random parent,
        then the child is mutated. Real genetic recombination instead of
        pure clone-and-mutate."""
        child = AdvancedBrain(parent_a.input_dim, pos=pos, is_predator=is_predator,
                               generation=max(parent_a.generation, parent_b.generation) + 1)
        mask1 = np.random.rand(*parent_a.W1.shape) < 0.5
        mask2 = np.random.rand(*parent_a.W2.shape) < 0.5
        child.W1 = np.where(mask1, parent_a.W1, parent_b.W1).copy()
        child.W2 = np.where(mask2, parent_a.W2, parent_b.W2).copy()
        child.mutate(rate=mutation_rate)
        return child


class RealEcosystemEnv:
    PREY_INPUT_DIM = 7 + MEMORY_DIM      # food(2) + threat(2) + herd(2) + energy(1) + memory
    PRED_INPUT_DIM = 5 + MEMORY_DIM      # prey(2) + stamina(1) + bias(2) + memory

    def __init__(self, num_prey=32, num_pred=2, num_food=40, num_food_clusters=6,
                 target_prey_pop=32, target_pred_pop=2, base_mutation=0.15):
        self.preys = [AdvancedBrain(self.PREY_INPUT_DIM, is_predator=False, instinct=PREY_INSTINCT)
                      for _ in range(num_prey)]
        self.predators = [AdvancedBrain(self.PRED_INPUT_DIM, is_predator=True, instinct=PRED_INSTINCT)
                           for _ in range(num_pred)]

        # Patchy food: a handful of cluster centers that food spawns/regrows around,
        # instead of pure uniform noise. Makes herding near a good patch matter.
        self.food_centers = np.random.uniform(-7, 7, size=(num_food_clusters, 2))
        self.foods = self._spawn_food_near_clusters(num_food)

        self.target_prey_pop = target_prey_pop
        self.target_pred_pop = target_pred_pop
        self.base_mutation = base_mutation

        self.time_step = 0
        self.events = []  # extinction / notable events, logged not silently patched over

        # Rolling history for the live population chart
        self.hist_len = 600
        self.stats = {"t": [], "prey": [], "pred": [], "food": []}

    # ------------------------------------------------------------------ #
    def _spawn_food_near_clusters(self, n):
        centers = self.food_centers[np.random.randint(0, len(self.food_centers), size=n)]
        pts = centers + np.random.normal(scale=1.2, size=(n, 2))
        return np.clip(pts, -9, 9)

    def _adaptive_mutation(self, population, target):
        """When a population is below its target, explore more (higher mutation);
        when it's thriving, exploit what's working (lower mutation)."""
        ratio = target / max(population, 1)
        return float(np.clip(self.base_mutation * ratio, self.base_mutation * 0.5, self.base_mutation * 2.5))

    # ------------------------------------------------------------------ #
    def update(self):
        self.time_step += 1

        # Regrow food near existing clusters rather than pure random scatter
        if len(self.foods) < 40 and np.random.rand() < 0.55:
            new_pt = self._spawn_food_near_clusters(1)
            self.foods = np.vstack([self.foods, new_pt]) if len(self.foods) else new_pt

        self._update_prey()
        self._update_predators()
        self._maybe_immigrate()
        self._record_stats()

    def _maybe_immigrate(self):
        """Real ecosystems aren't closed systems - individuals disperse in from
        neighboring populations. Without this, any small closed population
        WILL eventually go extinct from pure demographic bad luck, no matter
        how well-tuned the energy economy is (a basic, well-known result in
        population ecology). This is deliberately rare, only kicks in when a
        population is already thin, and is logged - it's a trickle of genetic
        rescue, not the old silent full-population reset."""
        if len(self.preys) < 10 and np.random.rand() < (0.02 if self.preys else 0.03):
            was_extinct = len(self.preys) == 0
            edge_pos = np.random.uniform(-9.5, 9.5, size=(2,))
            migrant = AdvancedBrain(self.PREY_INPUT_DIM, pos=edge_pos, is_predator=False, instinct=PREY_INSTINCT)
            self.preys.append(migrant)
            self.events.append((self.time_step, "prey population recolonized" if was_extinct else "prey immigrant arrived"))
        if len(self.predators) < 2 and np.random.rand() < (0.01 if self.predators else 0.015):
            was_extinct = len(self.predators) == 0
            edge_pos = np.random.uniform(-9.5, 9.5, size=(2,))
            migrant = AdvancedBrain(self.PRED_INPUT_DIM, pos=edge_pos, is_predator=True, instinct=PRED_INSTINCT)
            self.predators.append(migrant)
            self.events.append((self.time_step, "predator population recolonized" if was_extinct else "predator immigrant arrived"))

    # ------------------------------------------------------------------ #
    def _update_prey(self):
        n_prey = len(self.preys)
        if n_prey == 0:
            self.events.append((self.time_step, "PREY EXTINCT"))
            return

        prey_positions = np.array([p.pos for p in self.preys])
        # Snapshot food for this frame. We must NOT mutate self.foods mid-loop:
        # the batch KD-tree indices below are computed against this exact array,
        # so shrinking it while iterating would make later lookups point past
        # the end of the array. Eaten food is marked in a mask and only actually
        # removed from self.foods once, after the loop.
        foods_snapshot = self.foods
        food_alive = np.ones(len(foods_snapshot), dtype=bool)
        food_tree = cKDTree(foods_snapshot) if len(foods_snapshot) else None

        pred_positions = np.array([pr.pos for pr in self.predators]) if self.predators else None
        pred_tree = cKDTree(pred_positions) if pred_positions is not None and len(pred_positions) else None
        prey_tree = cKDTree(prey_positions) if n_prey > 1 else None

        # Batch nearest-neighbor queries instead of per-agent O(n) scans
        nearest_food_idx = food_tree.query(prey_positions)[1] if food_tree is not None else None
        nearest_pred_idx = pred_tree.query(prey_positions)[1] if pred_tree is not None else None
        herd_neighbor_lists = prey_tree.query_ball_point(prey_positions, r=4.0) if prey_tree is not None else None

        mutation_rate = self._adaptive_mutation(n_prey, self.target_prey_pop)
        # Density-dependent fertility: a real population-biology mechanism where
        # reproduction gets easier when numbers are low and harder when crowded,
        # instead of a fixed threshold that lets a small population stay stuck.
        repro_threshold = 130.0 * float(np.clip(n_prey / self.target_prey_pop, 0.45, 1.0))

        next_preys = []
        newborns = []
        reproduced_this_frame = set()

        for i, prey in enumerate(self.preys):
            prey.age += 1

            if herd_neighbor_lists is not None:
                neighbor_idx = [j for j in herd_neighbor_lists[i] if j != i]
                herd_vector = np.mean(prey_positions[neighbor_idx] - prey.pos, axis=0) if neighbor_idx else np.zeros(2)
            else:
                herd_vector = np.zeros(2)

            closest_food = foods_snapshot[nearest_food_idx[i]] if nearest_food_idx is not None else prey.pos
            closest_pred = pred_positions[nearest_pred_idx[i]] if nearest_pred_idx is not None else np.array([100.0, 100.0])

            state = np.array([
                closest_food[0] - prey.pos[0], closest_food[1] - prey.pos[1],
                closest_pred[0] - prey.pos[0], closest_pred[1] - prey.pos[1],
                herd_vector[0], herd_vector[1],
                prey.energy * 0.01,
            ])

            action = prey.forward(state)

            dist_to_danger = np.linalg.norm(closest_pred - prey.pos)
            speed_modifier = 0.6
            if dist_to_danger < 4.5 and prey.stamina > 20:
                speed_modifier = 1.1
                prey.stamina -= 4.0
                prey.bursting = True
            else:
                prey.stamina = min(100.0, prey.stamina + 1.5)
                prey.bursting = False

            move = np.clip(action, -speed_modifier, speed_modifier)
            prey.pos = np.clip(prey.pos + move, WORLD_LO, WORLD_HI)
            prey.energy -= 0.28 + np.linalg.norm(move) * 0.22

            alive_idx = np.where(food_alive)[0]
            if len(alive_idx) > 0:
                dists = np.linalg.norm(foods_snapshot[alive_idx] - prey.pos, axis=1)
                within_range = alive_idx[dists < 0.5]
                if len(within_range) > 0:
                    prey.energy += 55.0
                    food_alive[within_range[0]] = False  # one food item per bite

            if prey.energy <= 0 or prey.age > 400:
                continue

            # Sexual reproduction: find a nearby, well-fed mate via the herd tree
            if (prey.energy >= repro_threshold and len(self.preys) + len(newborns) < 40
                    and i not in reproduced_this_frame and herd_neighbor_lists is not None):
                candidates = [j for j in herd_neighbor_lists[i]
                              if j != i and self.preys[j].energy >= 100.0 and j not in reproduced_this_frame]
                if candidates:
                    mate_idx = candidates[np.argmin(np.linalg.norm(prey_positions[candidates] - prey.pos, axis=1))]
                    mate = self.preys[mate_idx]
                    child = AdvancedBrain.crossover(
                        prey, mate, prey.pos + np.random.uniform(-0.3, 0.3, size=(2,)), mutation_rate)
                    prey.energy -= 60.0
                    mate.energy -= 40.0  # mate contributes less than the "carrier"
                    newborns.append(child)
                    reproduced_this_frame.add(i)
                    reproduced_this_frame.add(mate_idx)
                elif prey.energy >= repro_threshold:
                    # No mate nearby: fall back to asexual so isolated survivors
                    # aren't permanently sterile
                    prey.energy -= 60.0
                    child = AdvancedBrain(self.PREY_INPUT_DIM, pos=prey.pos + np.random.uniform(-0.3, 0.3, size=(2,)),
                                           generation=prey.generation + 1)
                    child.W1, child.W2 = prey.W1.copy(), prey.W2.copy()
                    child.mutate(rate=mutation_rate)
                    newborns.append(child)

            next_preys.append(prey)

        self.foods = foods_snapshot[food_alive]
        self.preys = next_preys + newborns
        # No silent auto-respawn: a real die-off is logged, not papered over
        if len(self.preys) == 0:
            self.events.append((self.time_step, "PREY EXTINCT"))

    # ------------------------------------------------------------------ #
    def _update_predators(self):
        n_pred = len(self.predators)
        if n_pred == 0:
            return

        # Snapshot prey for this frame, same reasoning as the food snapshot above:
        # predators eating prey mid-loop must not shrink the array the batch
        # KD-tree indices were computed against.
        prey_snapshot = self.preys
        prey_alive = np.ones(len(prey_snapshot), dtype=bool)
        prey_positions = np.array([p.pos for p in prey_snapshot]) if prey_snapshot else None
        prey_tree = cKDTree(prey_positions) if prey_positions is not None and len(prey_positions) else None
        pred_positions = np.array([pr.pos for pr in self.predators])

        nearest_prey = prey_tree.query(pred_positions) if prey_tree is not None else (None, None)
        nearest_dists, nearest_idx = nearest_prey

        mutation_rate = self._adaptive_mutation(n_pred, self.target_pred_pop)
        # Predator carrying capacity tracks live prey abundance instead of a
        # fixed cap, so predators can't keep growing past what the current
        # prey population can actually support (basic carrying-capacity
        # ecology - this is what stops boom/overshoot/crash cycles from
        # being terminal).
        pred_cap = max(2, len(self.preys) // 6)

        next_predators = []
        for i, pred in enumerate(self.predators):
            pred.age += 1

            if nearest_idx is not None:
                closest_prey = prey_positions[nearest_idx[i]]
                min_dist = nearest_dists[i]
            else:
                closest_prey = pred.pos
                min_dist = 100.0

            pred_state = np.array([
                closest_prey[0] - pred.pos[0], closest_prey[1] - pred.pos[1],
                pred.stamina * 0.01, 1.0, 0.0,
            ])

            pred_action = pred.forward(pred_state)

            if min_dist < 4.0 and pred.stamina > 15:
                pred_speed = 0.95
                pred.stamina -= 5.0
                pred.bursting = True
            else:
                pred_speed = 0.45
                pred.stamina = min(100.0, pred.stamina + 2.0)
                pred.bursting = False

            pred_move = np.clip(pred_action, -pred_speed, pred_speed)
            pred.pos = np.clip(pred.pos + pred_move, WORLD_LO, WORLD_HI)
            # Density-dependent upkeep: more predators competing for the same
            # prey pool costs each of them more, a basic carrying-capacity
            # effect that keeps a predator boom from being self-sustaining.
            competition = 1.0 + 0.18 * max(0, len(self.predators) - 3)
            pred.energy -= (0.85 + np.linalg.norm(pred_move) * 0.4) * competition

            alive_idx = np.where(prey_alive)[0]
            if len(alive_idx) > 0:
                dists_to_prey = np.linalg.norm(prey_positions[alive_idx] - pred.pos, axis=1)
                in_range = alive_idx[dists_to_prey < 0.55]
                if len(in_range) > 0:
                    # Functional response: catch success drops as prey get scarce.
                    # Real predators have a harder time successfully cornering
                    # prey that are rare and scattered, not just harder time
                    # finding them - this is what protects the last stragglers
                    # of a population from being mopped up with certainty.
                    catch_prob = float(np.clip(len(self.preys) / 10.0, 0.15, 1.0))
                    if np.random.rand() < catch_prob:
                        nearest_in_range = in_range[np.argmin(dists_to_prey[dists_to_prey < 0.55])]
                        pred.energy = min(pred.energy + 42.0, 400.0)  # cap: no infinite banking
                        prey_alive[nearest_in_range] = False

            if pred.energy <= 0 or pred.age > 500:
                continue

            # Predator reproduction is coupled to prey availability, not just its
            # own energy reserve - mirrors real predator-prey (Lotka-Volterra)
            # dynamics where predator numbers can't keep climbing once their
            # food source has crashed.
            if pred.energy >= 380.0 and len(self.predators) < pred_cap:
                # Sexual reproduction for predators too, when a mate is close by
                others = [j for j, other in enumerate(self.predators)
                          if j != i and other.energy >= 200.0]
                if others:
                    mate = self.predators[others[np.argmin(
                        np.linalg.norm(pred_positions[others] - pred.pos, axis=1))]]
                    child = AdvancedBrain.crossover(
                        pred, mate, pred.pos + np.random.uniform(-0.5, 0.5, size=(2,)),
                        mutation_rate, is_predator=True)
                else:
                    child = AdvancedBrain(self.PRED_INPUT_DIM,
                                           pos=pred.pos + np.random.uniform(-0.5, 0.5, size=(2,)),
                                           is_predator=True, generation=pred.generation + 1)
                    child.W1, child.W2 = pred.W1.copy(), pred.W2.copy()
                    child.mutate(rate=mutation_rate)
                pred.energy -= 220.0
                next_predators.append(child)

            next_predators.append(pred)

        self.preys = [p for p, alive in zip(prey_snapshot, prey_alive) if alive]
        self.predators = next_predators
        if len(self.predators) == 0:
            self.events.append((self.time_step, "PREDATORS EXTINCT"))

    # ------------------------------------------------------------------ #
    def _record_stats(self):
        self.stats["t"].append(self.time_step)
        self.stats["prey"].append(len(self.preys))
        self.stats["pred"].append(len(self.predators))
        self.stats["food"].append(len(self.foods))
        if len(self.stats["t"]) > self.hist_len:
            for k in self.stats:
                self.stats[k] = self.stats[k][-self.hist_len:]


# === Run the visual display ===
env = RealEcosystemEnv()

fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]})
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_facecolor("#FCFBF4")

prey_scatter = ax.scatter([], [], s=70, edgecolors="#006064", label="AI Prey (energy = color)", cmap="RdYlGn")
pred_scatter = ax.scatter([], [], c="#FF1744", marker="v", s=130, edgecolors="#7F0000", label="AI Predators")
food_scatter = ax.scatter([], [], c="#2E7D32", marker="o", s=40, label="Natural Food")

ax.legend(loc="upper right", fontsize=8)
ax.grid(True, color="#E0E0E0", linestyle="--")
title_text = ax.set_title("")

ax_hist.set_facecolor("#FCFBF4")
ax_hist.grid(True, color="#E0E0E0", linestyle="--")
ax_hist.set_xlabel("time step")
ax_hist.set_ylabel("population")
line_prey, = ax_hist.plot([], [], color="#00838F", label="Prey")
line_pred, = ax_hist.plot([], [], color="#D50000", label="Predators")
line_food, = ax_hist.plot([], [], color="#2E7D32", alpha=0.5, label="Food")
ax_hist.legend(loc="upper right", fontsize=8)


def init():
    prey_scatter.set_offsets(np.empty((0, 2)))
    pred_scatter.set_offsets(np.empty((0, 2)))
    food_scatter.set_offsets(np.empty((0, 2)))
    return prey_scatter, pred_scatter, food_scatter, line_prey, line_pred, line_food


def animate(frame):
    env.update()

    if env.preys:
        prey_scatter.set_offsets(np.array([p.pos for p in env.preys]))
        energies = np.clip(np.array([p.energy for p in env.preys]) / 150.0, 0, 1)
        prey_scatter.set_array(energies)
        prey_scatter.set_cmap("RdYlGn")
    else:
        prey_scatter.set_offsets(np.empty((0, 2)))

    if env.predators:
        pred_scatter.set_offsets(np.array([pr.pos for pr in env.predators]))
        # Bursting predators (mid-attack) render larger & fully opaque; resting ones fade a bit
        sizes = [200 if pr.bursting else 110 for pr in env.predators]
        alphas = 1.0 if any(pr.bursting for pr in env.predators) else 0.85
        pred_scatter.set_sizes(sizes)
        pred_scatter.set_alpha(alphas)
    else:
        pred_scatter.set_offsets(np.empty((0, 2)))

    food_scatter.set_offsets(env.foods if len(env.foods) else np.empty((0, 2)))

    max_gen = max([p.generation for p in env.preys], default=0)
    title_text.set_text(
        f"Chaos-Evolve V10 | Sexual Reproduction + Adaptive Mutation\n"
        f"Prey: {len(env.preys)} | Predators: {len(env.predators)} | "
        f"Max Prey Generation: {max_gen} | Step: {frame}"
    )

    t = env.stats["t"]
    line_prey.set_data(t, env.stats["prey"])
    line_pred.set_data(t, env.stats["pred"])
    line_food.set_data(t, env.stats["food"])
    if t:
        ax_hist.set_xlim(max(0, t[0]), t[-1] + 1)
        ymax = max(max(env.stats["prey"], default=1), max(env.stats["food"], default=1),
                   max(env.stats["pred"], default=1)) + 2
        ax_hist.set_ylim(0, ymax)

    return prey_scatter, pred_scatter, food_scatter, line_prey, line_pred, line_food


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=2000, interval=35,
                               blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()