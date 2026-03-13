import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from boids.forward import forward
from modules.boids import convert_to_tf_sparse

def visualize_autoregressive_trajectory(model_path, loader_te, boids_te, save_path="auto_vis_gnca.mp4"):
    """
    Load the latest model and visualize the autoregressive trajectory.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Initialize the autoregressive trajectory
    boid_trajectory_auto = []

    # Generate the trajectory
    for sample in loader_te:
        inputs, x_next = sample

        if len(boid_trajectory_auto) == 0:
            boid_trajectory_auto.append(x_next)
        else:
            x_last = boid_trajectory_auto[-1]
            a_scipy = boids_te.get_neighbors(x_last[:, :2])
            a = convert_to_tf_sparse(a_scipy)
            inputs_auto = [x_last, a, inputs[-1]]
            x_next_auto = forward(model, *inputs_auto, training=False)
            boid_trajectory_auto.append(x_next_auto)

    # Convert trajectory to numpy array
    boid_trajectory_auto = np.array(boid_trajectory_auto)

    # Visualize the trajectory
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=10)

    def update(frame):
        positions = boid_trajectory_auto[frame, :, :2]
        scat.set_offsets(positions)
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=len(boid_trajectory_auto), interval=50, blit=True
    )

    # Save the animation
    ani.save(save_path, writer="ffmpeg")
    print(f"Animation saved as {save_path}")

# Example usage
visualize_autoregressive_trajectory("gnca_model", loader_te, boids_te)