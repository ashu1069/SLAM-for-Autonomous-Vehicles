import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D:
    def __init__(self, dt, process_noise_var_p, process_noise_var_v, measurement_noise_var_p, measurement_noise_var_v):
        self.dt = dt

        # Initial state: [position, velocity]
        self.x = np.zeros((2, 1))  # [x, v]
        self.P = np.eye(2)         # Initial uncertainty or covariance matrix (correlation between position and velocity)

        # State transition model
        self.F = np.array([[1, dt],
                           [0, 1]])

        # Process noise covariance
        self.Q = np.array([[process_noise_var_p * dt**3 / 3, process_noise_var_p * dt**2 / 2],
                           [process_noise_var_p * dt**2 / 2, process_noise_var_v * dt]])

        # Measurement model: both position and velocity
        self.H = np.array([[1, 0],  # GPS (position)
                           [0, 1]]) # Encoder (velocity)

        # Measurement noise covariance
        self.R = np.array([[measurement_noise_var_p, 0],
                           [0, measurement_noise_var_v]])

    def predict(self, u=0):
        B = np.array([[0.5 * self.dt**2],
                      [self.dt]])
        self.x = self.F @ self.x + B * u 
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        # Innovation
        y = z - self.H @ self.x
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update state
        self.x = self.x + K @ y
        # Update covariance
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        return self.x


dt = 1.0
N = 50
true_velocity = 1.0
true_positions = [i * true_velocity * dt for i in range(N)]
true_velocities = [true_velocity] * N

# Simulated measurements
gps_noise_std = 0.5
encoder_noise_std = 0.2
gps_measurements = [pos + np.random.normal(0, gps_noise_std) for pos in true_positions]
encoder_measurements = [vel + np.random.normal(0, encoder_noise_std) for vel in true_velocities]


kf = KalmanFilter1D(
    dt=dt,
    process_noise_var_p=0.1,
    process_noise_var_v=0.1,
    measurement_noise_var_p=gps_noise_std**2,
    measurement_noise_var_v=encoder_noise_std**2
)

position_estimates = []
velocity_estimates = []

for i in range(N):
    kf.predict()
    z = np.array([[gps_measurements[i]],
                  [encoder_measurements[i]]])
    x_est = kf.update(z)
    position_estimates.append(x_est[0, 0])
    velocity_estimates.append(x_est[1, 0])


plt.figure(figsize=(12, 5))

# Position plot
plt.subplot(1, 2, 1)
plt.plot(true_positions, label="True Position")
plt.plot(gps_measurements, label="GPS (noisy)", linestyle="dotted")
plt.plot(position_estimates, label="KF Estimate", linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("Position Estimation")
plt.legend()
plt.grid(True)

# Velocity plot
plt.subplot(1, 2, 2)
plt.plot(true_velocities, label="True Velocity")
plt.plot(encoder_measurements, label="Encoder (noisy)", linestyle="dotted")
plt.plot(velocity_estimates, label="KF Estimate", linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Velocity")
plt.title("Velocity Estimation")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/1D_kalman.png')
plt.show()
