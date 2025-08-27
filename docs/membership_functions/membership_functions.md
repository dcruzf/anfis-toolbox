# Membership Functions (Guide)

Membership functions (MFs) define fuzzy sets over each input dimension. They map crisp inputs x to degrees of membership μ(x) ∈ [0, 1] and are the learnable “premise” parameters in ANFIS.

This guide helps you choose the right MF type, understand parameters and constraints, and use them effectively in models.

## What you’ll find here

- Quick chooser: which MF to pick and why
- Shape catalog with intuition and parameters
- Constraints and common pitfalls
- Code examples: building inputs with different MFs
- Links to full API reference

See also: API reference for details and signatures in [Membership Functions](../api/membership-functions.md).

---

## Quick chooser

- Smooth and robust (default): GaussianMF or BellMF
- Sharp transitions, interpretable breakpoints: TriangularMF or TrapezoidalMF
- Monotonic ramps: SigmoidalMF (S-like), SShapedMF (soft step up), ZShapedMF (soft step down)
- S-shaped with both rise and fall (two transitions): PiMF

Rules of thumb:

- Prefer differentiable MFs (Gaussian, Bell, Sigmoidal, S/Z/Pi) for gradient-based training.
- Use piecewise-linear (Triangular, Trapezoidal) for crisp partitions and interpretability with linear slopes.
- Start simple (Gaussian) and evolve to Bell/Pi for flexibility if needed.

---

## Catalog of MF types

### GaussianMF

- Shape: Smooth bell curve.
- Params: mean (center), sigma > 0 (width).
- Use when: You want smooth, unimodal fuzzy sets; great default.
- Notes: Larger sigma = wider, smaller sigma = sharper.
- [more](gaussianmf.md)

### BellMF

- Shape: Generalized bell, adjustable shoulders and peak flatness.
- Params: a > 0 (width), b (slope/shape), c (center).
- Use when: Gaussian feels too restrictive; need heavier shoulders or flatter peak.
- [more](bellmf.md)

### SigmoidalMF

- Shape: Monotonic S-curve (increasing if a > 0; decreasing if a < 0).
- Params: a ≠ 0 (slope), c (center/inflection).
- Use when: Threshold-like behavior; left/right open fuzzy sets.

### SShapedMF

- Shape: Smooth step up from 0 to 1 between [a, b].
- Params: a < b (transition interval).
- Use when: You need a soft lower-bound transition; compact support mostly on right.

### ZShapedMF

- Shape: Smooth step down from 1 to 0 between [a, b].
- Params: a < b (transition interval).
- Use when: You need a soft upper-bound transition; compact support mostly on left.

### PiMF

- Shape: S-shaped rise (a → b), plateau, then Z-shaped fall (c → d).
- Params: a < b ≤ c < d (two transitions with optional plateau).
- Use when: You want a “top hat” with smooth edges.

### TriangularMF

- Shape: Piecewise linear triangle.
- Params: a ≤ b ≤ c (vertices; b is the peak).
- Use when: You want simple, interpretable partitions with linear slopes.

### TrapezoidalMF

- Shape: Piecewise linear trapezoid with a flat plateau.
- Params: a ≤ b ≤ c ≤ d (corners; [b, c] is plateau).
- Use when: You want robust coverage with flat central region.

---

## Parameter constraints and tips

- Ordering:
	- Triangular: a ≤ b ≤ c
	- Trapezoidal: a ≤ b ≤ c ≤ d
	- Pi: a < b ≤ c < d
	- S/Z: a < b
- Positivity:
	- Gaussian: sigma > 0
	- Bell: a > 0; b can be any real; c free
- Monotonicity:
	- Sigmoid a > 0 → increasing; a < 0 → decreasing
- Degenerate cases are validated and guarded against; avoid a == b for S/Z and a == b or c == b for triangular, etc., unless explicitly supported by your use case.

Training tips:

- Initialize centers (Gaussian/Bell/Sigmoid c) near data clusters or quantiles.
- Initialize widths (sigma/a) proportional to input range (e.g., range/3).
- For piecewise-linear MFs, place breakpoints (a, b, c, d) to cover the domain with overlap.

---

## Examples

### Using different MF types on an input

```python
import numpy as np
from anfis_toolbox.membership import (
		GaussianMF, BellMF, SigmoidalMF,
		SShapedMF, ZShapedMF, PiMF,
		TriangularMF, TrapezoidalMF,
)

x = np.linspace(-5, 5, 201)

mfs = [
		GaussianMF(mean=0.0, sigma=1.0),
		BellMF(a=1.5, b=2.0, c=0.0),
		SigmoidalMF(a=2.0, c=0.0),
		SShapedMF(a=-2.0, b=2.0),
		ZShapedMF(a=-2.0, b=2.0),
		PiMF(a=-3.0, b=-1.0, c=1.0, d=3.0),
		TriangularMF(a=-3.0, b=0.0, c=3.0),
		TrapezoidalMF(a=-4.0, b=-2.0, c=2.0, d=4.0),
]

mus = [mf.forward(x) for mf in mfs]
# mus is a list of arrays, each in [0, 1]
```

### Building an input partition

```python
import numpy as np
from anfis_toolbox.membership import GaussianMF

def gaussian_partition(xmin, xmax, n):
		centers = np.linspace(xmin, xmax, n)
		sigma = (xmax - xmin) / (3 * n)
		return [GaussianMF(mean=c, sigma=sigma) for c in centers]

partition = gaussian_partition(-5, 5, n=3)
```

---

## Choosing and tuning MFs

- Start with 2–4 MFs per input; increase only if underfitting.
- Prefer smooth MFs when optimizing continuously; consider piecewise-linear if you need crisp regions.
- Monitor overlap: adjacent MFs should overlap enough to avoid dead regions but not so much that they become redundant.
- Regularize widths (sigma/a) to prevent collapsing (too narrow) or washing out (too wide).

---

## API reference

For class signatures, attributes, and full docstrings, see:

- [Membership Functions API](../api/membership-functions.md)
	- [GaussianMF](../api/membership-functions.md#gaussianmf)
	- [BellMF](../api/membership-functions.md#bellmf)
	- [SigmoidalMF](../api/membership-functions.md#sigmoidalmf)
	- [SShapedMF](../api/membership-functions.md#sshapedmf)
	- [ZShapedMF](../api/membership-functions.md#zshapedmf)
	- [PiMF](../api/membership-functions.md#pimf)
	- [TriangularMF](../api/membership-functions.md#triangularmf)
	- [TrapezoidalMF](../api/membership-functions.md#trapezoidalmf)

---

## Troubleshooting

- NaNs during training: check parameter constraints (e.g., sigma > 0), and ensure forwards are called before backwards when using low-level APIs.
- Flat gradients: increase slopes (e.g., sigmoid a) or reduce widths (sigma/a); or space centers further apart.
- Overlapping too much: reduce widths or spread centers.
- Dead regions (μ ≈ 0): increase overlap or relocate centers toward data density.
