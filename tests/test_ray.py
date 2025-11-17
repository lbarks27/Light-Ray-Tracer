import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raytracer.refractive_index import RefractiveIndexField
from raytracer.ray import integrate_ray


def test_straight_in_constant_field():
    # In a constant refractive index field (no gradients), rays must be straight lines.
    field = RefractiveIndexField(n0=1.33, gaussians=[], linear_grad=np.zeros(3))
    r0 = np.array([0.0, 0.0, 0.0])
    dir0 = np.array([0.3, 0.4, 0.5])
    traj = integrate_ray(r0, dir0, field, ds=0.05, steps=100, adaptive=True)

    # The direction should remain constant. Compare trajectory with the expected line
    # parameterized by cumulative arclength obtained from the adaptive integrator.
    u = dir0 / np.linalg.norm(dir0)
    deltas = np.diff(traj, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    s_values = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    expected = r0[np.newaxis, :] + np.outer(s_values, u)

    max_err = np.max(np.linalg.norm(traj - expected, axis=1))
    assert max_err < 1e-8


def test_straight_line_with_fixed_step():
    field = RefractiveIndexField(n0=1.0, gaussians=[], linear_grad=np.zeros(3))
    r0 = np.zeros(3)
    dir0 = np.array([0.0, 1.0, 0.0])
    ds = 0.1
    steps = 50
    traj = integrate_ray(r0, dir0, field, ds=ds, steps=steps, adaptive=False)

    assert traj.shape == (steps + 1, 3)
    assert np.allclose(traj[:, 0], 0.0)
    assert np.allclose(traj[:, 2], 0.0)
    assert np.allclose(traj[:, 1], np.linspace(0.0, ds * steps, steps + 1))


def test_stop_when_leaving_domain():
    field = RefractiveIndexField(n0=1.0, gaussians=[], linear_grad=np.zeros(3))
    domain = np.array([[-1.0, 1.0], [-1.0, 0.5], [-1.0, 1.0]])
    traj = integrate_ray(
        np.array([0.0, -0.5, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        field,
        ds=0.05,
        steps=200,
        adaptive=True,
        domain_bounds=domain,
    )

    assert traj[-1, 1] >= domain[1, 1]
    assert traj.shape[0] < 201


def test_stop_on_surface_callable():
    field = RefractiveIndexField(n0=1.0, gaussians=[], linear_grad=np.zeros(3))

    def plane_surface(pos):
        return pos[1] >= 0.25

    traj = integrate_ray(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        field,
        ds=0.05,
        steps=200,
        adaptive=False,
        surfaces=[plane_surface],
    )

    assert traj[-1, 1] >= 0.25
    assert traj.shape[0] < 201
