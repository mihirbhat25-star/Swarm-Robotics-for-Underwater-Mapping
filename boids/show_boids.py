"""
Shows an animation of the Boids system.
"""
from modules.boids import Boids
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from modules.boids import make_dataset

boids = Boids(
    min_speed=0.0001,  # Min speed of the boids
    max_speed=0.01,  # Max speed of the boids
    max_force=0.1,  # Max amount of steering that any single update is allowed to add
    max_turn=5,  # How many degrees is a boid allowed to turn
    perception=0.1,  # How distant must two boids be in order to be neighbors
    crowding=0.020,  # How much groups are pushed apart (lower = tighter groups)
    n_boids=100,  # How many boids in the environment
    dt=1,  # Size of a time step (lower = more precise simulation)
    canvas_scale=1,  # Canvas is rescaled by this amount (used to control size)
    boundary_size_pctg=0.2,  # Relative size of the soft boundary
    wrap=False,  # If True, wrap around instead of avoiding boundary
    limits=True,  # If True, enforce speed and turn limits
    show=False,  # Show an animated plot of the boid everytime update_boids is called
)

history = boids.generate_trajectory()  # Generate a trajectory
# positions = history["positions"]

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# ax.set_title("Boids Simulation")
# scat = ax.scatter([], [], s=20)

# writer = FFMpegWriter(fps=40)
# print("🎬 Saving boids animation to boids_animation_wo_loiter.mp4...")
# with writer.saving(fig, "boids_animation_wo_loiter.mp4", dpi=100):
#     for frame in range(len(positions)):
#         scat.set_offsets(positions[frame])
#         ax.set_title(f"Boids Simulation")
#         writer.grab_frame()
# print("✅ Animation saved as boids_animation_wo_loiter.mp4")