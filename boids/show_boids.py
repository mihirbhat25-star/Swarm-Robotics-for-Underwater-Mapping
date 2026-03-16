"""
Shows an animation of the Boids system.
"""
from modules.boids import Boids
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

boids = Boids(
    min_speed=0.0001,
    max_speed=0.01,
    max_force=0.1,
    max_turn=5,
    perception=0.25,
    crowding=0.025,
    n_boids=100,
    dt=1,
    canvas_scale=1,
    boundary_size_pctg=0.2,
    wrap=False,
    show=False,
)

init_config = boids.get_random_init(boids.n_boids)  # Get a random initial configuration
history = boids.generate_trajectory(init_config, loiter=False, time_bool=True)  # Generate a trajectory
positions = history["positions"]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("Boids Simulation")
scat = ax.scatter([], [], s=20)

writer = FFMpegWriter(fps=40)
print("🎬 Saving boids animation to boids_animation_wo_loiter.mp4...")
with writer.saving(fig, "boids_animation_wo_loiter.mp4", dpi=100):
    for frame in range(len(positions)):
        scat.set_offsets(positions[frame])
        ax.set_title(f"Boids Simulation")
        writer.grab_frame()
print("✅ Animation saved as boids_animation_wo_loiter.mp4")