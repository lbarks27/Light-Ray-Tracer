import numpy as np


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length vector")
    return v / n


def integrate_ray(
    r0,
    dir0,
    field,
    ds=0.01,
    steps=1000,
    *,
    adaptive=True,
    tol=1e-5,
    min_step=None,
    max_step=None,
    domain_bounds=None,
    surfaces=None,
    stop_on_exit=True,
):
    """Integrate a ray through a continuous refractive-index field.

    We use the first-order system with state [r, p] where p = n(r) * dr/ds.

    Equations:
      dr/ds = p / n(r)
      dp/ds = grad_n(r)

    Parameters
    ----------
    r0 : array-like (3,)   initial position
    dir0: array-like (3,)  initial direction (need not be unit length)
    field: RefractiveIndexField instance with n(x) and grad_n(x)
    ds: float initial step-size in path-length s
    steps: int maximum number of steps
    adaptive: bool enabling adaptive RK4 with automatic step control
    tol: float error tolerance for adaptive stepping
    min_step, max_step: optional absolute step-size limits (default derived from ``ds``)
    domain_bounds: optional array-like with shape (3, 2) giving (min, max) bounds for
        x/y/z. Integration stops if the position leaves these bounds and
        ``stop_on_exit`` is True.
    surfaces: iterable of callables ``surface(position) -> bool`` that return True when
        the ray should stop (e.g., when reaching a surface).
    stop_on_exit: bool controlling whether leaving ``domain_bounds`` stops the ray.

    Returns
    -------
    traj : ndarray shape (N, 3) positions along ray. ``N`` may be smaller than
        ``steps + 1`` if adaptive stepping or stopping conditions terminate early.
    """
    r = np.asarray(r0, dtype=float)
    u0 = _normalize(dir0)
    n0 = field.n(r)
    p = n0 * u0

    # state = [r_x, r_y, r_z, p_x, p_y, p_z]
    state = np.empty(6, dtype=float)
    state[0:3] = r
    state[3:6] = p

    def deriv(sstate):
        r = sstate[0:3]
        p = sstate[3:6]
        n = field.n(r)
        drds = p / n
        dpds = field.grad_n(r)
        return np.concatenate([drds, dpds])

    def rk4_step(sstate, step):
        k1 = deriv(sstate)
        k2 = deriv(sstate + 0.5 * step * k1)
        k3 = deriv(sstate + 0.5 * step * k2)
        k4 = deriv(sstate + step * k3)
        return sstate + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def adaptive_step(sstate, step):
        full = rk4_step(sstate, step)
        half = rk4_step(rk4_step(sstate, 0.5 * step), 0.5 * step)
        err = np.linalg.norm(half[0:3] - full[0:3])
        return half, err

    bounds = None
    if domain_bounds is not None:
        bounds = np.asarray(domain_bounds, dtype=float)
        if bounds.shape != (3, 2):
            raise ValueError("domain_bounds must have shape (3, 2)")

    surface_tests = list(surfaces or [])

    def should_stop(pos):
        if bounds is not None and stop_on_exit:
            if np.any(pos < bounds[:, 0]) or np.any(pos > bounds[:, 1]):
                return True
        for surface in surface_tests:
            if surface(pos):
                return True
        return False

    if not adaptive:
        traj = np.zeros((steps + 1, 3), dtype=float)
        traj[0] = state[0:3].copy()
        valid = 1
        for _ in range(steps):
            state = rk4_step(state, ds)
            traj[valid] = state[0:3]
            valid += 1
            if should_stop(state[0:3]):
                return traj[:valid]
        return traj

    # Adaptive stepping path storage
    traj = [state[0:3].copy()]
    h = ds
    min_h = ds / 32.0 if min_step is None else float(min_step)
    max_h = ds * 5.0 if max_step is None else float(max_step)
    h = np.clip(h, min_h, max_h)

    for _ in range(steps):
        next_state, err = adaptive_step(state, h)
        if err > tol and h > min_h:
            h = max(h * 0.5, min_h)
            continue
        state = next_state
        traj.append(state[0:3].copy())
        if should_stop(state[0:3]):
            break
        if err < tol / 4.0 and h < max_h:
            h = min(h * 2.0, max_h)
    return np.vstack(traj)
