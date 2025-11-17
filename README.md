# Light Ray Tracer (continuous refractive index)

This small project integrates rays through a continuously-varying refractive-index field in 3D and visualizes them with matplotlib's 3D tools. This is an exercise to help learn and prepare for my mach cutoff simulation project in which I will be propagating shockwave geometry via raytracing. 

Quick start
----------

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the demo:

```bash
python -m run_demo
```

3. Run tests:

```bash
pytest -q
```

What is implemented
--------------------

- `RefractiveIndexField` with background `n0`, optional Gaussian perturbations, and optional linear gradient.
- RK4 integrator for the ray equations using the formulation d/ds(n dr/ds) = grad n (implemented as first-order system in (r, p)).
- Adaptive ray integration with automatic step control plus optional stopping conditions (domain bounds and arbitrary user-specified surfaces).
- `run_demo.py` to visualize rays through a Gaussian bump in `n`.

Goals
------------------
- Add surfaces / interfaces with Snell's law.
- Add a visualization of the refractive-index field in the plot. 