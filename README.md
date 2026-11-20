# particle_simulation
Proxima Centauri: N-Body Simulation 

An ongoing computational physics project focused on exploring orbital mechanics, gravitational interactions, and multi-body dynamics inspired by the Proxima Centauri stellar system.

The project aims to progressively develop a 3D N-body simulation using numerical methods and scientific computing with Python.

Current Progress

The project currently includes an initial working prototype of a three-particle system.

Implemented Features

- 3D particle motion and trajectory visualization.
- Pairwise Newtonian gravitational interactions.
- Magnetic Lorentz force for charged particles.
- Numerical integration using the Semi-Implicit Euler method.
- Animated particle trajectories using Matplotlib.
- Customizable initial positions, velocities, masses, and charges.

Technologies Used

- Python
- NumPy
- Matplotlib
- Matplotlib Animation

Project Roadmap

- [x] Build the initial three-particle working prototype.
- [x] Implement pairwise gravitational interactions.
- [x] Implement magnetic force interactions.
- [x] Visualize 3D trajectories.
- [ ] Improve numerical integration and stability.
- [ ] Expand the model to include additional celestial bodies.
- [ ] Incorporate the Proxima Centauri planetary system.
- [ ] Explore the Alpha Centauri A–B binary system.
- [ ] Develop a more comprehensive multi-body simulation.
- [ ] Perform numerical validation and conservation analysis.

Project Status

Early Development — Working Prototype

This is an evolving educational and computational physics project. The current implementation is not a validated astronomical model of the Proxima Centauri system.

Motivation

The goal of this project is to explore how mathematical models and computational methods can be used to understand complex gravitational systems and orbital dynamics.
Built through independent research, experimentation, mathematical exploration, and AI-assisted development.

---
Exploring the universe, one simulation at a time.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 1.2
dt = 0.01
total_steps = 3400

p1 = {
    "mass": 19.0,
    "charge": 1.0,
    "pos": np.array([-1.0, -0.5, 0.4], dtype=float),
    "vel": np.array([0.0, 0.0, 0.0], dtype=float)
}

p2 = {
    "mass": 4.0,
    "charge": -1.0,
    "pos": np.array([2.2, 0.5, -0.4], dtype=float),
    "vel": np.array([0.0, 0.0, 0.0], dtype=float)
}

p3 = {
    "mass": 0.5,
    "charge": 0.0,  
    "pos": np.array([6.2, 0.0, 0.9], dtype=float),
    "vel": np.array([0.0, 0.0, 0.0], dtype=float)
}

total_mass_12 = p1["mass"] + p2["mass"]
com_12 = (p1["mass"] * p1["pos"] + p2["mass"] * p2["pos"]) / total_mass_12

r1_com = np.linalg.norm(p1["pos"] - com_12)
r2_com = np.linalg.norm(p2["pos"] - com_12)
init_dist_12 = np.linalg.norm(p2["pos"] - p1["pos"])

f12_mag = G * p1["mass"] * p2["mass"] / (init_dist_12 ** 2)
v1_base = np.sqrt(f12_mag * r1_com / p1["mass"])
v2_base = np.sqrt(f12_mag * r2_com / p2["mass"])

p1["vel"] = np.array([0.05, -v1_base * 0.85, -0.25], dtype=float)
p2["vel"] = np.array([-0.05, v2_base * 0.85,  0.65], dtype=float)

r3_vec = p3["pos"] - com_12
r3_dist = np.linalg.norm(r3_vec)
v3_circ = np.sqrt(G * total_mass_12 / r3_dist)

tangent_xy = np.array([-r3_vec[1], r3_vec[0], 0.0], dtype=float)
tangent_xy = tangent_xy / np.linalg.norm(tangent_xy)
p3["vel"] = tangent_xy * (v3_circ * 0.76) + np.array([0.0, 0.0, -0.35], dtype=float)

B_field = np.array([0.0, 0.0, 0.35], dtype=float)


def pairwise_attraction(a, b, g_const, soften=0.25):
    delta = b["pos"] - a["pos"]
    dist = np.linalg.norm(delta)
    if dist < soften:
        dist = soften
    unit = delta / dist
    mag = g_const * a["mass"] * b["mass"] / (dist ** 2)
    return mag * unit


def step_physics_3body(b1, b2, b3, g_const, b_vec, time_step):
    
    f_12 = pairwise_attraction(b1, b2, g_const)
    f_13 = pairwise_attraction(b1, b3, g_const)
    f_23 = pairwise_attraction(b2, b3, g_const)

    
    tot_f1 = f_12 + f_13 + b1["charge"] * np.cross(b1["vel"], b_vec)
    tot_f2 = -f_12 + f_23 + b2["charge"] * np.cross(b2["vel"], b_vec)
    tot_f3 = -f_13 - f_23 + b3["charge"] * np.cross(b3["vel"], b_vec)

    
    a1 = tot_f1 / b1["mass"]
    a2 = tot_f2 / b2["mass"]
    a3 = tot_f3 / b3["mass"]

    
    b1["vel"] += a1 * time_step
    b2["vel"] += a2 * time_step
    b3["vel"] += a3 * time_step

    b1["pos"] += b1["vel"] * time_step
    b2["pos"] += b2["vel"] * time_step
    b3["pos"] += b3["vel"] * time_step



times = [0.0]
traj1 = [p1["pos"].copy()]
traj2 = [p2["pos"].copy()]
traj3 = [p3["pos"].copy()]

current_time = 0.0
for _ in range(total_steps):
    step_physics_3body(p1, p2, p3, G, B_field, dt)
    current_time += dt
    times.append(current_time)
    traj1.append(p1["pos"].copy())
    traj2.append(p2["pos"].copy())
    traj3.append(p3["pos"].copy())

times = np.array(times)
traj1 = np.array(traj1)
traj2 = np.array(traj2)
traj3 = np.array(traj3)

fig = plt.figure(figsize=(9, 8), facecolor="black")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

limit = 7.5
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)

ax.grid(False)
ax.set_axis_off()

rng = np.random.default_rng(seed=42)
num_bg_particles = 350
bg_stars = rng.uniform(-limit * 1.1, limit * 1.1, size=(num_bg_particles, 3))
bg_sizes = rng.uniform(2.0, 6.5, size=num_bg_particles)
bg_alphas = rng.uniform(0.30, 0.80, size=num_bg_particles)

bg_colors = np.zeros((num_bg_particles, 4))
bg_colors[:, 0] = 1.0
bg_colors[:, 1] = 1.0
bg_colors[:, 2] = 1.0
bg_colors[:, 3] = bg_alphas

ax.scatter(
    bg_stars[:, 0],
    bg_stars[:, 1],
    bg_stars[:, 2],
    c=bg_colors,
    s=bg_sizes,
    edgecolors="none"
)

pt1 = ax.scatter([], [], [], color="cyan", s=120, label="Particle 1 ")
pt2 = ax.scatter([], [], [], color="orange", s=60, label="Particle 2 ")
pt3 = ax.scatter([], [], [], color="#7CFC00", s=45, label="Particle 3 ")

trail1 = ax.scatter([], [], [], color="cyan", s=9, alpha=0.35)
trail2 = ax.scatter([], [], [], color="orange", s=7, alpha=0.35)
trail3 = ax.scatter([], [], [], color="#7CFC00", s=7, alpha=0.30)

time_text = ax.set_title("3D 3-Body Elliptical Orbit Simulation | Time: 0.00 s", color="white")

legend = ax.legend(loc="upper right", facecolor="#111111", edgecolor="#333333")
for text in legend.get_texts():
    text.set_color("white")

trail_length = 450
sample_step = 5

def update(frame):
    start = max(0, frame - trail_length)

    t1 = traj1[start:frame + 1:sample_step]
    t2 = traj2[start:frame + 1:sample_step]
    t3 = traj3[start:frame + 1:sample_step]

    trail1._offsets3d = (t1[:, 0], t1[:, 1], t1[:, 2])
    trail2._offsets3d = (t2[:, 0], t2[:, 1], t2[:, 2])
    trail3._offsets3d = (t3[:, 0], t3[:, 1], t3[:, 2])

    pt1._offsets3d = ([traj1[frame, 0]], [traj1[frame, 1]], [traj1[frame, 2]])
    pt2._offsets3d = ([traj2[frame, 0]], [traj2[frame, 1]], [traj2[frame, 2]])
    pt3._offsets3d = ([traj3[frame, 0]], [traj3[frame, 1]], [traj3[frame, 2]])

    time_text.set_text(f"3D 3-Body Elliptical Orbit Simulation | Time: {times[frame]:.2f} s")
    return pt1, pt2, pt3, trail1, trail2, trail3, time_text


ani = FuncAnimation(fig, update, frames=len(times), interval=15, blit=False)
plt.tight_layout()
plt.show()
              
