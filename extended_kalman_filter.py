import numpy as np
import matplotlib.pyplot as plt

def h(x):        # Nonlinear measurement function
    return x**2

def H_jacobian(x):  # Derivative of h(x) w.r.t. x
    return 2 * x

N=50
# Initial state
x = 1.0
P = 1.0
Q = 0.01
R = 1.0  # Sensor noise

true_x = np.linspace(1, 5, N)
measurements = h(true_x) + np.random.normal(0, np.sqrt(R), N)

ekf_estimates = []

for z in measurements:
    # Prediction step (x doesn't change here, for simplicity)
    P = P + Q

    # Update step (EKF)
    H = H_jacobian(x)
    z_pred = h(x)
    K = P * H / (H**2 * P + R)  # EKF Gain
    x = x + K * (z - z_pred)
    P = (1 - K * H) * P

    ekf_estimates.append(x)

# Plotting
plt.plot(true_x, label="True x")
plt.plot(np.sqrt(measurements), label="Measurements (sqrt)", linestyle='dotted')
plt.plot(ekf_estimates, label="EKF Estimate")
plt.legend()
plt.title("Extended Kalman Filter (Nonlinear Sensor)")
plt.xlabel("Time Step")
plt.ylabel("State Estimate")
plt.savefig('images/extended_kalman.png')
plt.show()
