import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# =========================================================
# 1. Configuration
# =========================================================

DATA_FILE = "xy_data.csv"
OUTPUT_DIR = "outputs"

# Number of uniformly sampled points used for curve evaluation.
N_EVAL_POINTS = 1500


# =========================================================
# 2. Load and validate the data
# =========================================================

data = pd.read_csv(DATA_FILE)

required_columns = {"x", "y"}

if not required_columns.issubset(data.columns):
    raise ValueError("CSV must contain 'x' and 'y' columns.")

if len(data) == 0:
    raise ValueError("CSV file is empty.")

x = data["x"].to_numpy(dtype=float)
y = data["y"].to_numpy(dtype=float)

print(f"Loaded {len(data)} points.")
print(f"Columns: {list(data.columns)}")


# =========================================================
# 3. Recover t and calculate residuals
# =========================================================

def recover_t_and_residuals(params):
    """
    Recover t using the projection along the direction theta.

    t = (x-X)cos(theta) + (y-42)sin(theta)

    The perpendicular projection should satisfy:

    -(x-X)sin(theta) + (y-42)cos(theta)
        = exp(M*t) * sin(0.3*t)

    Since t > 6, |t| = t.
    """

    theta, M, X = params

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    t = (
        (x - X) * cos_theta
        + (y - 42.0) * sin_theta
    )

    observed_perpendicular = (
        -(x - X) * sin_theta
        + (y - 42.0) * cos_theta
    )

    predicted_perpendicular = (
        np.exp(M * t)
        * np.sin(0.3 * t)
    )

    residuals = (
        observed_perpendicular
        - predicted_perpendicular
    )

    return t, residuals


# =========================================================
# 4. Optimization objective
# =========================================================

def objective(params):
    """
    Objective function for bounded nonlinear least squares.

    Solutions producing t values outside the required
    interval (6, 60) receive a penalty.
    """

    t, residuals = recover_t_and_residuals(params)

    penalty = np.zeros_like(residuals)

    invalid_low = t <= 6
    invalid_high = t >= 60

    penalty[invalid_low] += (
        100.0 * (6.0 - t[invalid_low])
    )

    penalty[invalid_high] += (
        100.0 * (t[invalid_high] - 60.0)
    )

    return residuals + penalty


# =========================================================
# 5. Parameter bounds
# =========================================================

theta_min = np.deg2rad(0.000001)
theta_max = np.deg2rad(49.999999)

M_min = -0.049999999
M_max = 0.049999999

X_min = 0.000001
X_max = 99.999999

lower_bounds = [
    theta_min,
    M_min,
    X_min
]

upper_bounds = [
    theta_max,
    M_max,
    X_max
]


# =========================================================
# 6. Multiple starting points
# =========================================================

initial_points = [
    [np.deg2rad(30.0), 0.03, 55.0],
    [np.deg2rad(20.0), 0.00, 50.0],
    [np.deg2rad(40.0), -0.02, 20.0],
    [np.deg2rad(10.0), 0.02, 30.0],
    [np.deg2rad(45.0), -0.03, 70.0]
]


# =========================================================
# 7. Run optimization
# =========================================================

best_result = None
best_error = np.inf

for initial in initial_points:

    result = least_squares(
        objective,
        initial,
        bounds=(lower_bounds, upper_bounds),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=10000
    )

    t_candidate, residuals_candidate = (
        recover_t_and_residuals(result.x)
    )

    valid_t = (
        np.all(t_candidate > 6)
        and np.all(t_candidate < 60)
    )

    error = np.mean(residuals_candidate ** 2)

    if valid_t and error < best_error:
        best_error = error
        best_result = result


if best_result is None:
    raise RuntimeError(
        "No valid parameter solution was found."
    )


# =========================================================
# 8. Extract final parameters
# =========================================================

theta, M, X = best_result.x

theta_degrees = np.rad2deg(theta)

t, residuals = recover_t_and_residuals(
    best_result.x
)


# =========================================================
# 9. Reconstruct observed points
# =========================================================

predicted_x = (
    t * np.cos(theta)
    - np.exp(M * np.abs(t))
    * np.sin(0.3 * t)
    * np.sin(theta)
    + X
)

predicted_y = (
    42.0
    + t * np.sin(theta)
    + np.exp(M * np.abs(t))
    * np.sin(0.3 * t)
    * np.cos(theta)
)


# =========================================================
# 10. Pointwise reconstruction errors
# =========================================================

x_error = np.abs(x - predicted_x)
y_error = np.abs(y - predicted_y)

mean_x_error = np.mean(x_error)
mean_y_error = np.mean(y_error)

max_x_error = np.max(x_error)
max_y_error = np.max(y_error)

pointwise_l1 = np.mean(
    np.abs(x - predicted_x)
    + np.abs(y - predicted_y)
)


# =========================================================
# 11. Uniformly sampled curve evaluation
# =========================================================

def curve_from_t(t_values, theta_value, M_value, X_value):
    """
    Evaluate the parametric curve for an array of t values.
    """

    t_values = np.asarray(t_values, dtype=float)

    curve_x = (
        t_values * np.cos(theta_value)
        - np.exp(M_value * np.abs(t_values))
        * np.sin(0.3 * t_values)
        * np.sin(theta_value)
        + X_value
    )

    curve_y = (
        42.0
        + t_values * np.sin(theta_value)
        + np.exp(M_value * np.abs(t_values))
        * np.sin(0.3 * t_values)
        * np.cos(theta_value)
    )

    return curve_x, curve_y


# Sort recovered t values so that the observed data follows
# the natural parameter direction.

sort_order = np.argsort(t)

t_sorted = t[sort_order]

# Use the actual recovered t interval.
t_start = t_sorted[0]
t_end = t_sorted[-1]

# Uniform sampling in the parameter t.
uniform_t = np.linspace(
    t_start,
    t_end,
    N_EVAL_POINTS
)


# ---------------------------------------------------------
# Expected curve
# ---------------------------------------------------------
#
# The supplied points are treated as the expected/observed
# curve. We obtain its uniformly sampled representation by
# interpolating the sorted observed coordinates against t.
#

observed_x_sorted = x[sort_order]
observed_y_sorted = y[sort_order]

uniform_expected_x = np.interp(
    uniform_t,
    t_sorted,
    observed_x_sorted
)

uniform_expected_y = np.interp(
    uniform_t,
    t_sorted,
    observed_y_sorted
)


# ---------------------------------------------------------
# Predicted curve
# ---------------------------------------------------------

uniform_predicted_x, uniform_predicted_y = curve_from_t(
    uniform_t,
    theta,
    M,
    X
)


# ---------------------------------------------------------
# L1 curve distance
# ---------------------------------------------------------

l1_x = np.abs(
    uniform_expected_x - uniform_predicted_x
)

l1_y = np.abs(
    uniform_expected_y - uniform_predicted_y
)

uniform_l1_total = np.sum(
    l1_x + l1_y
)

uniform_l1_mean = np.mean(
    l1_x + l1_y
)

uniform_l1_max = np.max(
    l1_x + l1_y
)


# =========================================================
# 12. Print final results
# =========================================================

print("\n" + "=" * 65)
print("FINAL PARAMETER ESTIMATION")
print("=" * 65)

print(f"Theta (degrees): {theta_degrees:.10f}")
print(f"M:                {M:.10f}")
print(f"X:                {X:.10f}")

print("\nRounded underlying parameters:")
print(f"Theta = {theta_degrees:.6f} degrees")
print(f"M     = {M:.8f}")
print(f"X     = {X:.8f}")


print("\n" + "-" * 65)
print("PARAMETER CONSTRAINT VALIDATION")
print("-" * 65)

print(
    f"Theta valid: "
    f"{0 < theta_degrees < 50}"
)

print(
    f"M valid: "
    f"{-0.05 < M < 0.05}"
)

print(
    f"X valid: "
    f"{0 < X < 100}"
)


print("\nRecovered t range:")
print(f"Minimum t: {t.min():.10f}")
print(f"Maximum t: {t.max():.10f}")

print(
    f"t constraint satisfied: "
    f"{np.all(t > 6) and np.all(t < 60)}"
)


print("\n" + "-" * 65)
print("RESIDUAL STATISTICS")
print("-" * 65)

print(
    f"Mean squared residual: "
    f"{best_error:.12e}"
)

print(
    f"Mean absolute residual: "
    f"{np.mean(np.abs(residuals)):.12e}"
)

print(
    f"Maximum absolute residual: "
    f"{np.max(np.abs(residuals)):.12e}"
)


print("\n" + "-" * 65)
print("POINTWISE RECONSTRUCTION ERROR")
print("-" * 65)

print(
    f"Mean |x error|: "
    f"{mean_x_error:.12e}"
)

print(
    f"Mean |y error|: "
    f"{mean_y_error:.12e}"
)

print(
    f"Max |x error|: "
    f"{max_x_error:.12e}"
)

print(
    f"Max |y error|: "
    f"{max_y_error:.12e}"
)

print(
    f"Mean pointwise L1 error: "
    f"{pointwise_l1:.12e}"
)


print("\n" + "-" * 65)
print("UNIFORMLY SAMPLED CURVE L1 EVALUATION")
print("-" * 65)

print(f"Number of evaluation points: {N_EVAL_POINTS}")

print(
    f"Evaluation t range: "
    f"{t_start:.10f} to {t_end:.10f}"
)

print(
    f"Uniform curve L1 distance (sum): "
    f"{uniform_l1_total:.12e}"
)

print(
    f"Uniform curve L1 distance (mean): "
    f"{uniform_l1_mean:.12e}"
)

print(
    f"Maximum sampled L1 error: "
    f"{uniform_l1_max:.12e}"
)


# =========================================================
# 13. Create output directory
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 14. Save numerical results
# =========================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "results.txt"
)

with open(results_file, "w") as file:

    file.write(
        "Assignment 1 - Parameter Estimation\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Theta (degrees): "
        f"{theta_degrees:.10f}\n"
    )

    file.write(
        f"M: {M:.10f}\n"
    )

    file.write(
        f"X: {X:.10f}\n"
    )

    file.write("\nRounded parameters:\n")

    file.write(
        "Theta = 30 degrees\n"
    )

    file.write(
        "M = 0.03\n"
    )

    file.write(
        "X = 55\n"
    )

    file.write("\nParameter validation:\n")

    file.write(
        f"Theta valid: "
        f"{0 < theta_degrees < 50}\n"
    )

    file.write(
        f"M valid: "
        f"{-0.05 < M < 0.05}\n"
    )

    file.write(
        f"X valid: "
        f"{0 < X < 100}\n"
    )

    file.write("\nRecovered t range:\n")

    file.write(
        f"Minimum t: {t.min():.10f}\n"
    )

    file.write(
        f"Maximum t: {t.max():.10f}\n"
    )

    file.write(
        f"t constraint satisfied: "
        f"{np.all(t > 6) and np.all(t < 60)}\n"
    )

    file.write("\nResidual statistics:\n")

    file.write(
        f"Mean squared residual: "
        f"{best_error:.12e}\n"
    )

    file.write(
        f"Mean absolute residual: "
        f"{np.mean(np.abs(residuals)):.12e}\n"
    )

    file.write(
        f"Maximum absolute residual: "
        f"{np.max(np.abs(residuals)):.12e}\n"
    )

    file.write("\nPointwise reconstruction:\n")

    file.write(
        f"Mean |x error|: "
        f"{mean_x_error:.12e}\n"
    )

    file.write(
        f"Mean |y error|: "
        f"{mean_y_error:.12e}\n"
    )

    file.write(
        f"Max |x error|: "
        f"{max_x_error:.12e}\n"
    )

    file.write(
        f"Max |y error|: "
        f"{max_y_error:.12e}\n"
    )

    file.write(
        f"Mean pointwise L1 error: "
        f"{pointwise_l1:.12e}\n"
    )

    file.write("\nUniformly sampled curve evaluation:\n")

    file.write(
        f"Number of evaluation points: "
        f"{N_EVAL_POINTS}\n"
    )

    file.write(
        f"Evaluation t range: "
        f"{t_start:.10f} to {t_end:.10f}\n"
    )

    file.write(
        f"Uniform curve L1 distance (sum): "
        f"{uniform_l1_total:.12e}\n"
    )

    file.write(
        f"Uniform curve L1 distance (mean): "
        f"{uniform_l1_mean:.12e}\n"
    )

    file.write(
        f"Maximum sampled L1 error: "
        f"{uniform_l1_max:.12e}\n"
    )


# =========================================================
# 15. Save fitted curve plot
# =========================================================

plot_file = os.path.join(
    OUTPUT_DIR,
    "fitted_curve.png"
)

plt.figure(figsize=(10, 7))

plt.scatter(
    x,
    y,
    s=8,
    alpha=0.5,
    label="Observed data"
)

plt.plot(
    uniform_predicted_x,
    uniform_predicted_y,
    linewidth=2,
    label="Reconstructed curve"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "Observed Data vs Reconstructed Parametric Curve"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


# =========================================================
# 16. Save uniformly sampled comparison
# =========================================================

comparison_file = os.path.join(
    OUTPUT_DIR,
    "uniform_curve_comparison.csv"
)

comparison_data = pd.DataFrame({
    "t": uniform_t,
    "expected_x": uniform_expected_x,
    "expected_y": uniform_expected_y,
    "predicted_x": uniform_predicted_x,
    "predicted_y": uniform_predicted_y,
    "absolute_x_error": l1_x,
    "absolute_y_error": l1_y,
    "point_l1_error": l1_x + l1_y
})

comparison_data.to_csv(
    comparison_file,
    index=False
)


# =========================================================
# 17. Completion message
# =========================================================

print("\n" + "=" * 65)
print("OUTPUT FILES")
print("=" * 65)

print(f"Saved: {results_file}")
print(f"Saved: {plot_file}")
print(f"Saved: {comparison_file}")

print("\nAssignment 1 processing completed successfully.")