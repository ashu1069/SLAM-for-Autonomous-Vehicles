import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter2D:
    def __init__(self, dt, process_noise_pos, process_noise_vel, measurement_noise_pos, measurement_noise_vel):
        self.dt = dt

        # State vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

        # State transition model
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Process noise covariance
        q = process_noise_pos
        r = process_noise_vel
        dt2 = dt ** 2
        dt3 = dt ** 3 / 2
        dt4 = dt ** 4 / 4

        self.Q = np.array([
            [dt4*q, 0,    dt3*q, 0],
            [0,    dt4*q, 0,    dt3*q],
            [dt3*q, 0,    dt2*q, 0],
            [0,    dt3*q, 0,    dt2*q]
        ])

        # Measurement model: directly observe [x, y, vx, vy]
        self.H = np.eye(4)

        # Measurement noise covariance
        self.R = np.diag([measurement_noise_pos, measurement_noise_pos,
                          measurement_noise_vel, measurement_noise_vel])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.x.shape[0]) - K @ self.H) @ self.P
        return self.x

# --------------------------
# Simulate 2D motion
# --------------------------

dt = 1.0
N = 50
true_vx, true_vy = 1.0, 0.5

true_positions = [(i * true_vx, i * true_vy) for i in range(N)]
true_velocities = [(true_vx, true_vy) for _ in range(N)]

# Simulate noisy measurements
gps_noise_std = 0.5
encoder_noise_std = 0.2

gps_measurements = [(x + np.random.normal(0, gps_noise_std),
                     y + np.random.normal(0, gps_noise_std)) for x, y in true_positions]

encoder_measurements = [(vx + np.random.normal(0, encoder_noise_std),
                         vy + np.random.normal(0, encoder_noise_std)) for vx, vy in true_velocities]

# --------------------------
# Run Kalman Filter
# --------------------------

kf = KalmanFilter2D(
    dt=dt,
    process_noise_pos=0.1,
    process_noise_vel=0.1,
    measurement_noise_pos=gps_noise_std**2,
    measurement_noise_vel=encoder_noise_std**2
)

est_x, est_y, est_vx, est_vy = [], [], [], []

for i in range(N):
    z = np.array([
        [gps_measurements[i][0]],
        [gps_measurements[i][1]],
        [encoder_measurements[i][0]],
        [encoder_measurements[i][1]]
    ])
    kf.predict()
    x_est = kf.update(z)
    est_x.append(x_est[0, 0])
    est_y.append(x_est[1, 0])
    est_vx.append(x_est[2, 0])
    est_vy.append(x_est[3, 0])

# --------------------------
# Plotting
# --------------------------

plt.figure(figsize=(12, 6))

# Position
plt.subplot(1, 2, 1)
true_x = [p[0] for p in true_positions]
true_y = [p[1] for p in true_positions]
gps_x = [m[0] for m in gps_measurements]
gps_y = [m[1] for m in gps_measurements]

plt.plot(true_x, true_y, label="True Position", linewidth=2)
plt.scatter(gps_x, gps_y, label="GPS", s=20, c='gray', alpha=0.5)
plt.plot(est_x, est_y, label="KF Estimate", linestyle="--")
plt.title("2D Position Tracking")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis("equal")
plt.grid(True)

# Velocity
plt.subplot(1, 2, 2)
plt.plot([true_vx]*N, label="True vx")
plt.plot([true_vy]*N, label="True vy")
plt.plot(est_vx, label="Estimated vx", linestyle="--")
plt.plot(est_vy, label="Estimated vy", linestyle="--")
plt.title("Velocity Tracking")
plt.xlabel("Time Step")
plt.ylabel("Velocity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/2D_kalman.png')
plt.show()
