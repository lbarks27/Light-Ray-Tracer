"""Demo: launch several rays through a continuous refractive-index field and plot with matplotlib 3D."""
import numpy as np
from raytracer.refractive_index import RefractiveIndexField
from raytracer.ray import integrate_ray
import matplotlib.pyplot as plt


def main():
    # Create a field: background 1.0 with a Gaussian bump at origin
    field = RefractiveIndexField(
        n0=1.0,
        gaussians=[{"amplitude": 10.0, "center": np.array([0.0, 0.0, 0.0]), "sigma": 0.6}],
        linear_grad=np.array([0.0, 0.0, 0.0]),
    )

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    bounds = np.array([[-2.5, 2.5], [-4.0, 2.0], [-2.0, 2.0]])

    # Visualize the refractive-index field as a semi-transparent scatter cloud
    grid_x = np.linspace(bounds[0][0], bounds[0][1], 12)
    grid_y = np.linspace(bounds[1][0], bounds[1][1], 12)
    grid_z = np.linspace(bounds[2][0], bounds[2][1], 12)
    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z)
    sample_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    n_values = np.array([field.n(p) for p in sample_points])
    scatter = ax.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        sample_points[:, 2],
        c=n_values,
        cmap='magma',
        alpha=0.55,
        s=12,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Refractive index', color='white')

    # Create a grid of ray origins at y = -3, pointing towards +y
    xs = np.linspace(-1.5, 1.5, 5)
    zs = np.linspace(-1.0, 1.0, 3)
    origins = [np.array([x, -3.0, z]) for x in xs for z in zs]

    def sensor_plane(pos):
        return pos[1] >= 1.5

    for o in origins:
        traj = integrate_ray(
            o,
            np.array([1.0, 1.0, 1.0]),
            field,
            ds=0.02,
            steps=600,
            adaptive=True,
            domain_bounds=bounds,
            surfaces=[sensor_plane],
        )
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=0.9, color='lightgray')

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('white')
        axis.set_tick_params(colors='white')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ray paths through a Gaussian refractive-index bump', color='white')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
