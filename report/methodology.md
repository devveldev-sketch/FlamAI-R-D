# Assignment 1 — Parameter Estimation Methodology

## 1. Problem Statement

The objective is to recover the three unknown parameters `theta`, `M`, and `X` from the observed `(x, y)` points supplied in `xy_data.csv`.

The parametric curve is:

```text
x = t*cos(theta)
    - exp(M*|t|)*sin(0.3*t)*sin(theta) + X

y = 42 + t*sin(theta)
    + exp(M*|t|)*sin(0.3*t)*cos(theta)
```

The parameter constraints are:

```text
0 < theta < 50 degrees
-0.05 < M < 0.05
0 < X < 100
6 < t < 60
```

The supplied `xy_data.csv` contains 1,500 observed `(x, y)` points.

The assignment evaluates the solution using the L1 distance between uniformly sampled points on the expected and predicted curves. It also awards marks for explaining the process and for the submitted code/repository.

---

## 2. Approach

A direct optimization over every point's unknown `t` value would introduce a large number of additional variables. Instead, the equations can be transformed so that `t` can be recovered directly from `(x, y)` once `theta` and `X` are specified.

This reduces the numerical optimization problem to three unknown parameters:

```text
theta
M
X
```

---

## 3. Mathematical Transformation

Starting from the original equations:

```text
x - X = t*cos(theta)
        - exp(M*|t|)*sin(0.3*t)*sin(theta)

y - 42 = t*sin(theta)
         + exp(M*|t|)*sin(0.3*t)*cos(theta)
```

Project the coordinates onto the direction defined by `theta`:

```text
(x - X)*cos(theta) + (y - 42)*sin(theta)
```

Substituting the equations causes the oscillating terms to cancel:

```text
-exp(M*|t|)*sin(0.3*t)*sin(theta)*cos(theta)
+
exp(M*|t|)*sin(0.3*t)*cos(theta)*sin(theta)
= 0
```

Therefore:

```text
t = (x - X)*cos(theta) + (y - 42)*sin(theta)
```

This provides a direct recovered `t` value for every observed point.

---

## 4. Perpendicular Projection

A perpendicular projection isolates the oscillating component:

```text
-(x - X)*sin(theta) + (y - 42)*cos(theta)
```

The linear `t` terms cancel, giving:

```text
-(x - X)*sin(theta) + (y - 42)*cos(theta)
=
exp(M*|t|)*sin(0.3*t)
```

Because the valid range is `6 < t < 60`, all valid recovered `t` values are positive, so:

```text
|t| = t
```

The residual for each observed point is therefore:

```text
observed_perpendicular
-
exp(M*t)*sin(0.3*t)
```

The optimization minimizes these residuals.

---

## 5. Numerical Optimization

The implementation uses bounded nonlinear least squares from:

```text
scipy.optimize.least_squares
```

The optimization variables are:

```text
[theta, M, X]
```

The parameter bounds enforce the ranges specified in the assignment.

Multiple starting points are used to reduce dependence on a single initial guess.

For each candidate parameter set:

1. Recover `t` from the parallel projection.
2. Calculate the perpendicular observed component.
3. Calculate the expected oscillating component.
4. Compute the residual.
5. Minimize the residual across all 1,500 observations.
6. Validate that every recovered `t` satisfies `6 < t < 60`.

---

## 6. Curve Reconstruction

After estimating the parameters, the original equations are evaluated using the recovered `t` values:

```text
x_pred =
t*cos(theta)
- exp(M*t)*sin(0.3*t)*sin(theta)
+ X

y_pred =
42
+ t*sin(theta)
+ exp(M*t)*sin(0.3*t)*cos(theta)
```

The reconstructed coordinates are compared with the supplied observations.

---

## 7. Experimental Results

Running:

```bash
python3 solve.py
```

on the supplied `xy_data.csv` produced:

```text
theta = 29.9999729322 degrees
M     = 0.0299999969
X     = 54.9999982128
```

These estimates are effectively:

```text
theta = 30 degrees
M     = 0.03
X     = 55
```

The small differences are numerical precision effects.

---

## 8. Parameter Constraint Validation

| Parameter | Estimated value | Required range |
|-----------|-----------------|----------------|
| theta | 29.9999729322° | 0° < theta < 50° |
| M | 0.0299999969 | -0.05 < M < 0.05 |
| X | 54.9999982128 | 0 < X < 100 |

All three parameter constraints were satisfied.

The recovered `t` values were:

```text
Minimum t = 6.0494054727
Maximum t = 59.9951707023
```

Therefore all supplied points satisfy:

```text
6 < t < 60
```

---

## 9. Residual Validation

The optimization produced:

```text
Mean squared residual:
1.215331957292e-11

Mean absolute residual:
2.559805438534e-06

Maximum absolute residual:
1.761505485520e-05
```

These very small residuals indicate that the estimated parameters closely reproduce the structure of the supplied curve.

---

## 10. Coordinate Reconstruction Validation

Using the estimated parameters and recovered `t` values:

```text
Mean |x error| = 1.279901671940e-06
Mean |y error| = 2.216857142855e-06

Maximum |x error| = 8.807520217147e-06
Maximum |y error| = 1.525508915279e-05
```

The mean pointwise coordinate L1 error from the current reconstruction is:

```text
3.496758814795e-06
```

This is a pointwise reconstruction diagnostic and is reported separately from the uniformly sampled curve evaluation.

---

## 11. Uniformly Sampled Curve Evaluation

To align with the assignment's stated L1 evaluation criterion, the implementation also performs a uniform sampling comparison.

The recovered `t` values are sorted to establish the parameter order of the supplied observations. The observed curve is then represented at uniformly spaced `t` values using interpolation.

The predicted curve is evaluated directly at the same uniformly spaced `t` values.

For each sampled point, the coordinate-wise L1 error is:

```text
|expected_x - predicted_x|
+
|expected_y - predicted_y|
```

The current implementation uses:

```text
Number of evaluation points = 1500
```

The resulting diagnostic values are:

```text
Uniform curve L1 distance (sum):
2.583681874781e-01

Uniform curve L1 distance (mean):
1.722454583187e-04

Maximum sampled L1 error:
9.082030708996e-03
```

The uniformly sampled comparison is saved to:

```text
outputs/uniform_curve_comparison.csv
```

This file contains the sampled `t` values, expected coordinates, predicted coordinates, and coordinate-wise errors.

---

## 12. Reproducibility

The complete analysis can be reproduced with:

```bash
python3 solve.py
```

The script:

- loads and validates the CSV,
- estimates the unknown parameters,
- validates the recovered `t` range,
- reconstructs the curve,
- calculates residual and reconstruction statistics,
- performs the uniformly sampled curve comparison,
- saves numerical results, and
- generates a fitted-curve plot.

Generated outputs are:

```text
outputs/results.txt
outputs/fitted_curve.png
outputs/uniform_curve_comparison.csv
```

---

## 13. Final Answer

The recovered unknown parameters are:

```text
theta = 30 degrees
M     = 0.03
X     = 55
```

Therefore, the recovered parametric curve is:

```text
x = t*cos(30°)
    - exp(0.03*t)*sin(0.3*t)*sin(30°) + 55

y = 42 + t*sin(30°)
    + exp(0.03*t)*sin(0.3*t)*cos(30°)

6 < t < 60
```

The estimated parameters produce an extremely close reconstruction of the 1,500 supplied points.
