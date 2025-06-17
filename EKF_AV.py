'''
Initialize:
  x = initial_state_vector  # e.g. [x, y, vx, vy, theta]
  P = initial_covariance_matrix
  Q = process_noise_covariance
  R = measurement_noise_covariance

Loop every time step (dt):
  # 1. Prediction Step (Motion Model)
  x_pred = f(x, u, dt)          # Non-linear motion model, e.g. bicycle model
  F_jacobian = Jacobian_of_f_at(x, u, dt)
  P_pred = F_jacobian * P * F_jacobian.T + Q

  # 2. Measurement Update (Sensor Fusion)
  z = get_sensor_measurements()  # GPS, IMU, encoders, etc.
  
  # Predict measurement from predicted state
  z_pred = h(x_pred)             # Measurement function (may be non-linear)
  H_jacobian = Jacobian_of_h_at(x_pred)
  
  y = z - z_pred                # Innovation (measurement residual)
  S = H_jacobian * P_pred * H_jacobian.T + R    # Innovation covariance
  K = P_pred * H_jacobian.T * inverse(S)        # Kalman gain

  # Update state estimate and covariance
  x = x_pred + K * y
  P = (I - K * H_jacobian) * P_pred

  # (Optional) Normalize angle theta in x if present to [-pi, pi]

Return x, P as updated state and covariance

'''

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilterAV:
    def __init__(self, dt):
        self.dt = dt
        
        # State vector [x,y,v,theta]
        self.x = np.zeros([4,1])

        # Covariance matrix
        self.P = np.eye(4) + 0.1

        # Process Noise Covariance
        self.Q = np.diag([0.1,0.1,0.2,0.05])

        # Measurement noise covariance
        self.R = np.diag([1.0,1.0])

    def predict(self, u):
        v = self.x[2,0]
        theta = self.x[3,0]
        a = u[0] # acceleration
        omega = u[1] # angular velocity

        dt = self.dt

        # Motion Model
        self.x[0,0] += v * np.cos(theta) * dt
        self.x[1,0] += v * np.sin(theta) * dt
        self.x[2,0] += a * dt
        self.x[3,0] += omega * dt

        # Jacobian of the Motion Model
        F = np.array([
            [1, 0, np.cos(theta) * dt, -v * np.sin(theta) * dt],
            [0, 1, np.sin(theta) * dt,  v * np.cos(theta) * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        # Measurement Model: Only GPS (x,y)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z_pred = H @ self.x
        y = z - z_pred # innovation

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4)- K @ H) @ self.P

    def get_state(self):
        return self.x
    
if __name__ == "__main__":
    ekf = KalmanFilterAV(dt=0.1)

    true_path = []
    measured_path = []
    estimated_path = []

    # Simulate 50 steps
    for t in range(100):
        # Simulate control input: accelerate forward, small right turn
        u = np.array([0.1, 0.01])  # [acceleration, yaw rate]

        # Predict step
        ekf.predict(u)

        # Simulate GPS measurement with noise
        true_x = ekf.x[0, 0] + np.random.normal(0, 1.0)
        true_y = ekf.x[1, 0] + np.random.normal(0, 1.0)
        z = np.array([[true_x], [true_y]])

        # Update step
        ekf.update(z)

        # Store data
        true_path.append([ekf.x[0, 0] - np.random.normal(0, 1.0), ekf.x[1, 0] - np.random.normal(0, 1.0)])
        measured_path.append([true_x, true_y])
        estimated_path.append([ekf.x[0, 0], ekf.x[1, 0]])

        # Print filtered state
        print(f"Step {t+1}: x={ekf.x[0,0]:.2f}, y={ekf.x[1,0]:.2f}, v={ekf.x[2,0]:.2f}, theta={ekf.x[3,0]:.2f}")

    # Convert paths to arrays for plotting
    true_path = np.array(true_path)
    measured_path = np.array(measured_path)
    estimated_path = np.array(estimated_path)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(true_path[:, 0], true_path[:, 1], label='True Path', linestyle='--', alpha=0.5)
    plt.plot(measured_path[:, 0], measured_path[:, 1], label='Noisy GPS Measurements', marker='x', linestyle='')
    plt.plot(estimated_path[:, 0], estimated_path[:, 1], label='EKF Estimated Path', linewidth=2)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.title('Extended Kalman Filter - 2D Localization')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('images/ekf_av.png')
    plt.show()